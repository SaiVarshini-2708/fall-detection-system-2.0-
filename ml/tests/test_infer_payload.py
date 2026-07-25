import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infer import send_alert


def test_send_alert_includes_severity_proxy_when_provided():
    captured = {}

    def fake_post(url, json=None, timeout=5):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout

        class Response:
            status_code = 201
            text = "ok"

        return Response()

    with patch("infer.requests.post", side_effect=fake_post):
        send_alert(
            {"fall_type": "slip", "pre_activity": "walking", "confidence": 0.91},
            "stunned",
            confirmation_window_ms=1200,
            location="Kitchen",
            severity_proxy="high",
        )

    assert captured["url"].endswith("/api/alert")
    payload = captured["json"]
    assert payload["fall_type"] == "slip"
    assert payload["pre_activity"] == "walking"
    assert payload["post_state"] == "stunned"
    assert payload["confidence"] == 0.91
    assert payload["severity_proxy"] == "high"


def test_send_alert_derives_severity_proxy_from_window_when_missing():
    captured = {}

    def fake_post(url, json=None, timeout=5):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout

        class Response:
            status_code = 201
            text = "ok"

        return Response()

    window = np.zeros((200, 6), dtype=np.float32)
    window[:, 0] = np.linspace(0, 8, 200, dtype=np.float32)

    with patch("infer.requests.post", side_effect=fake_post):
        send_alert(
            {"fall_type": "slip", "pre_activity": "walking", "confidence": 0.91},
            "stunned",
            window=window,
        )

    assert captured["url"].endswith("/api/alert")
    payload = captured["json"]
    assert payload["severity_proxy"] in {"low", "medium", "high"}
