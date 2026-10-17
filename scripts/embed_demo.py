"""
A stand-in "existing website" that embeds the trendify viewer in an iframe, for trying the
embedding in a real browser. Run it, open the printed address, and work through the checklist.

    python scripts/embed_demo.py                 # viewer mounted at /viz on the same site
    python scripts/embed_demo.py --csp           # plus a strict Content-Security-Policy on both
    python scripts/embed_demo.py --cross-origin  # viewer on a different origin from the page

Cross-origin mode uses one port: the page is served as "localhost" and the viewer as
"127.0.0.1", which a browser treats as different sites. That keeps it working through port
forwarding (VS Code Remote SSH, `ssh -L`), where a second port might not map through.
"""

import argparse
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from starlette.routing import Host

from trendify.viewer.app import create_app, make_server

DEFAULT_DB = Path(__file__).parent / "example_data" / "trendify" / "trendify.db"
VIEWER_HOST = "127.0.0.1"

# The policy the user guide recommends for the viewer's own pages.
VIEWER_CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; "
    "font-src 'self' data:; worker-src 'self' blob:"
)

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>My existing site</title></head>
<body style="margin:0;font-family:sans-serif">
<header style="height:40px;box-sizing:border-box;padding:10px 16px;background:#222;color:#fff;display:flex;gap:24px">
  <b>My existing site</b><span>mode: {mode}</span><span id="beat">host /health: ...</span>
</header>
<iframe src="{src}" title="Trendify viewer" allow="fullscreen; clipboard-write"
        style="width:100%;height:calc(100vh - 40px);border:0"></iframe>
<script>
  // If the viewer ever blocks this page's server, these numbers jump.
  let worst = 0, since = performance.now();
  setInterval(async () => {{
    const start = performance.now();
    try {{ await fetch("/health", {{cache: "no-store"}}); }} catch (e) {{}}
    const last = performance.now() - start;
    if (start - since > 5000) {{ worst = 0; since = start; }}
    worst = Math.max(worst, last);
    document.getElementById("beat").textContent =
      `host /health: ${{last.toFixed(0)}} ms (worst of last 5 s: ${{worst.toFixed(0)}} ms)`;
  }}, 100);
</script></body></html>"""

CHECKLIST = """
Open {open_url}

Look for:
  1. The tag sidebar loads; clicking a plot tag draws it, clicking a table tag shows the table.
  2. DevTools > Network: no 404s or failed requests. DevTools > Console: no errors{csp_note}.
  3. The header's "worst of last 5 s" stays small while a heavy plot loads (it used to jump to ~1000+ ms).
  4. Toggle the theme and drag the sidebar edge; reload; they should be remembered.
  5. The fullscreen and copy buttons work from inside the frame.
{cross_note}"""


def viewer_origin(request: Request) -> str:
    """The viewer's address as the browser sees it: same scheme and port, other hostname."""
    port = f":{request.url.port}" if request.url.port else ""
    return f"{request.url.scheme}://{VIEWER_HOST}{port}"


def build_site(args: argparse.Namespace) -> tuple[FastAPI, FastAPI]:
    viewer = create_app(args.db, show_db_path=False)
    site = FastAPI()

    @site.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @site.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        origin = viewer_origin(request)
        src = f"{origin}/" if args.cross_origin else "/viz/"
        mode = (
            "viewer on another origin"
            if args.cross_origin
            else "viewer mounted at /viz"
        )
        response = HTMLResponse(PAGE.format(src=src, mode=mode))
        if args.csp:
            frame_src = origin if args.cross_origin else "'self'"
            response.headers["Content-Security-Policy"] = (
                f"default-src 'self'; script-src 'self' 'unsafe-inline'; "
                f"style-src 'self' 'unsafe-inline'; frame-src {frame_src}"
            )
        return response

    if args.csp:

        @viewer.middleware("http")
        async def viewer_csp(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            response = await call_next(request)
            if response.headers.get("content-type", "").startswith("text/html"):
                response.headers["Content-Security-Policy"] = VIEWER_CSP
            return response

    if args.cross_origin:
        # First, so the viewer owns every path under its hostname, including "/".
        site.router.routes.insert(0, Host(VIEWER_HOST, app=viewer))
    else:
        site.mount("/viz", viewer)
    return site, viewer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help="trendify.db to show"
    )
    parser.add_argument("--port", type=int, default=8000, help="port to serve on")
    parser.add_argument(
        "--csp", action="store_true", help="send Content-Security-Policy headers"
    )
    parser.add_argument(
        "--cross-origin",
        action="store_true",
        help="serve the viewer from a different origin than the page",
    )
    args = parser.parse_args()
    if not args.db.exists():
        raise SystemExit(f"No database at {args.db}. Pass one with --db.")

    site, viewer = build_site(args)
    cross_note = (
        "  6. Open the page as localhost, not 127.0.0.1 (that name serves the viewer alone). In Chrome,\n"
        "     block third-party cookies (Settings > Privacy), reload: the viewer should still work, it\n"
        "     just forgets the theme and sidebar width. No red errors in the Console.\n"
        if args.cross_origin
        else ""
    )
    print(
        CHECKLIST.format(
            open_url=f"http://localhost:{args.port}",
            csp_note=" (any 'Content Security Policy' line is a finding)"
            if args.csp
            else "",
            cross_note=cross_note,
        )
    )
    make_server(
        site, host="127.0.0.1", port=args.port, log_level="warning", viewers=(viewer,)
    ).run()


if __name__ == "__main__":
    main()
