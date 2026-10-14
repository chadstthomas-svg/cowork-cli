"""cowork-serve: run the companion as an OpenAI-compatible endpoint.

Wraps llama-server with the model so ANY OpenAI-compatible client
(LM Studio frontends, open-webui, LangChain, curl) can use it:

    cowork-serve --gguf cowork-companion-350M-Q8_0.gguf --port 8080 [--vulkan]

Then point a client at http://localhost:8080/v1/chat/completions with:
    system: (the coworker system prompt - printed at startup, also in
            curiousgene/data/cowork_tasks.SYSTEM_PROMPT)
    user:   "<instruction>\\n\\n=== Provided material ===\\n<file text>"

The envelope JSON comes back in message.content; `cowork --from-json` style
tooling can render it. Note: for exact spreadsheet totals also enable the
harness ledger (cowork CLI --compute) - a raw endpoint serves model output
only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from curiousgene.data.cowork_tasks import SYSTEM_PROMPT
from curiousgene.product.cowork_cli import LlamaServer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True,
                    help="path to a merged GGUF, e.g. "
                         "cowork-companion-350M-Q8_0.gguf from "
                         "https://huggingface.co/tablinum/cowork-companion")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--vulkan", action="store_true")
    args = ap.parse_args()

    gguf = Path(args.gguf)
    if not gguf.exists():
        print(f"[cowork-serve] model not found: {gguf}")
        return 1

    print("[cowork-serve] system prompt (send as the system message):")
    print(SYSTEM_PROMPT)
    print(f"[cowork-serve] endpoint: http://127.0.0.1:{args.port}/v1/chat/completions")
    print(f"[cowork-serve] model: {gguf}")
    print("[cowork-serve] starting llama-server (Ctrl+C to stop)...")
    with LlamaServer(str(gguf), args.ctx, args.vulkan) as endpoint:
        print(f"[cowork-serve] ready: {endpoint}")
        try:
            while True:
                import time
                time.sleep(3600)
        except KeyboardInterrupt:
            print("[cowork-serve] stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
