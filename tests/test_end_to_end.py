"""Run and score against a fake OpenAI-compatible server: no API key or network needed."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benchpress.cli import main


class FakeModel(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        letter = "B" if len(prompt) % 3 else "A"
        text = f"{letter} - A fake rationale." if "<LETTER>" in prompt else letter
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
               "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": text}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_run_resume_and_score(server, tmp_path):
    out = tmp_path / "run"
    args = ["run", "--model", "custom:fake", "--base-url", server, "--out", str(out),
            "--cells", "afp-editorial-standards-and-best-practices__T01",
            "--arms", "letter", "rationale", "--replicates", "2", "--workers", "4", "--yes"]
    main(args)
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4 * 10 * 2 * 2           # 4 pairs x 10 stimuli x 2 arms x 2 replicates
    assert all(r["chosen_option"] in ("A", "B") and r["error"] is None for r in rows)
    assert all(r["rationale"] for r in rows if r["arm"] == "rationale")

    main(args)                                   # resume: nothing left to do
    assert len((out / "results.jsonl").read_text(encoding="utf-8").splitlines()) == len(rows)

    main(["score", str(out), "--boot", "50"])
    summary = json.loads((out / "report" / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["models"]["fake"]) == {"letter", "rationale"}
    assert (out / "report" / "report.md").exists()


def test_l2_scored_against_core(server, tmp_path):
    out = tmp_path / "run"
    cell = ["--cells", "afp-editorial-standards-and-best-practices__T01"]
    base = ["run", "--model", "custom:fake", "--base-url", server, "--out", str(out), "--yes", *cell]
    main(base + ["--configurations", "1", "4"])
    main(base + ["--track", "l2", "--l2-all"])
    main(["score", str(out), "--boot", "0"])
    assert (out / "report" / "l2_shifts.csv").exists()
