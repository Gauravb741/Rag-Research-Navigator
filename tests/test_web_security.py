import http.client
import json
import threading
from pathlib import Path

from src.interface.web import GraphReasoner, SecureHandler, validate_payload

ROOT = Path(__file__).resolve().parents[1]
STATE = json.loads((ROOT / "data/knowledge/knowledge_state.json").read_text(encoding="utf-8"))


def test_web_input_validation_limits_and_shape():
    assert validate_payload(b'{"idea":"A structured multi-hop RAG research idea."}')
    for raw in (b'{"idea":7}', b'{"idea":"short"}', b'{"idea":"valid enough idea","extra":"reject"}'):
        try:
            validate_payload(raw)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe or invalid payload was accepted")
    try:
        validate_payload(json.dumps({"idea": "x" * 5001}).encode())
    except ValueError:
        pass
    else:
        raise AssertionError("oversized idea was accepted")
    try:
        validate_payload(b'{"idea":"valid idea with surrogate \\ud800"}')
    except ValueError:
        pass
    else:
        raise AssertionError("invalid Unicode was accepted")


def test_web_routes_are_same_origin_and_safe_errors():
    from http.server import ThreadingHTTPServer

    SecureHandler.reasoner = GraphReasoner(STATE)
    server = ThreadingHTTPServer(("127.0.0.1", 0), SecureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        connection = http.client.HTTPConnection(host, port, timeout=5)
        connection.request("GET", "/api/health")
        response = connection.getresponse()
        body = json.loads(response.read())
        assert response.status == 200
        assert body["status"] == "ok"
        assert response.getheader("X-Content-Type-Options") == "nosniff"
        assert "default-src 'self'" in response.getheader("Content-Security-Policy")

        connection.request("GET", "/api/graph")
        response = connection.getresponse()
        graph = json.loads(response.read())
        assert response.status == 200
        from tests.test_system import STATE as _STATE
        assert graph["metadata"]["entity_count"] == len(_STATE["entities"])
        assert graph["metadata"]["relationship_count"] == len(_STATE["relationships"])
        assert len(graph["entities"]) == len(_STATE["entities"])
        assert len(graph["relationships"]) == len(_STATE["relationships"])
        assert "mapping_rule" not in graph["entities"][0]

        connection.request("POST", "/api/reason", body=json.dumps({"idea": "x"}), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.loads(response.read())
        assert response.status == 400
        assert "stack" not in json.dumps(body).lower()
        assert "traceback" not in json.dumps(body).lower()

        connection.request("GET", "/../data/knowledge/knowledge_state.json")
        response = connection.getresponse()
        response.read()
        assert response.status in {400, 404}
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
