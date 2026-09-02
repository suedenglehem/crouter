"""Tool-call transparency (PRD 25)."""

import json


def test_tools_and_tool_choice_forwarded_verbatim():
    """The echo backend returns the exact request it received."""
    from fastapi.testclient import TestClient

    from router.api import create_app
    from conftest import make_config

    cfg = make_config(**{"backends": {"fast": {"type": "mock", "name": "local-fast", "model": "local-fast", "behavior": "echo"}}})
    with TestClient(create_app(cfg)) as c:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "description": "Run the test suite",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                },
            }
        ]
        payload = {
            "model": "local-fast",
            "messages": [{"role": "user", "content": "run the tests"}],
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "run_tests"}},
            "temperature": 0.2,
        }
        r = c.post("/v1/chat/completions", json=payload)
        assert r.status_code == 200
        received = json.loads(r.json()["choices"][0]["message"]["content"])["received"]

        # Tool schemas and choice must pass through unmodified.
        assert received["tools"] == tools
        assert received["tool_choice"] == payload["tool_choice"]
        assert received["messages"] == payload["messages"]
        assert received["temperature"] == 0.2
        # The model field is rewritten to the backend's real model name.
        assert received["model"] == "local-fast"


def test_tool_call_response_passthrough(client):
    from fastapi.testclient import TestClient

    from router.api import create_app
    from conftest import make_config

    cfg = make_config(**{"backends": {"fast": {"type": "mock", "name": "local-fast", "model": "local-fast", "behavior": "tool_call"}}})
    with TestClient(create_app(cfg)) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "local-fast", "messages": [{"role": "user", "content": "weather in paris?"}]},
        )
        assert r.status_code == 200
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        tc = choice["message"]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        # Arguments arrive as a JSON string, uninterpreted.
        json.loads(tc["function"]["arguments"])


def test_tool_call_streaming_chunks(client):
    from fastapi.testclient import TestClient

    from router.api import create_app
    from conftest import make_config

    cfg = make_config(**{"backends": {"fast": {"type": "mock", "name": "local-fast", "model": "local-fast", "behavior": "tool_call"}}})
    with TestClient(create_app(cfg)) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "local-fast", "stream": True, "messages": [{"role": "user", "content": "weather?"}]},
        )
        assert r.status_code == 200
        tool_chunks = []
        for line in r.text.splitlines():
            if line.startswith("data:") and "[DONE]" not in line:
                ev = json.loads(line[len("data:"):].strip())
                for ch in ev.get("choices", []):
                    delta = ch.get("delta", {})
                    if "tool_calls" in delta:
                        tool_chunks.append(delta["tool_calls"])
        assert len(tool_chunks) == 1
        assert tool_chunks[0][0]["function"]["name"] == "get_weather"


def test_assistant_tool_message_round_trip():
    """A follow-up request containing a tool result message is forwarded as-is."""
    from fastapi.testclient import TestClient

    from router.api import create_app
    from conftest import make_config

    cfg = make_config(**{"backends": {"fast": {"type": "mock", "name": "local-fast", "model": "local-fast", "behavior": "echo"}}})
    with TestClient(create_app(cfg)) as c:
        messages = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        r = c.post("/v1/chat/completions", json={"model": "local-fast", "messages": messages})
        received = json.loads(r.json()["choices"][0]["message"]["content"])["received"]
        assert received["messages"] == messages
