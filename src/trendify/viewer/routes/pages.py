"""
HTML page routes for the dashboard. The index page server-renders the full tag tree (see
`trendify.viewer.tag_tree`) directly into the sidebar via a recursive Jinja2 macro, which is
simpler and more robust than fetching it client-side and building the recursive structure in
Alpine, which has no solid native support for recursive components.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from trendify.viewer.hydration import RunnerClosed
from trendify.viewer.tag_tree import build_tag_tree

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    # The tag tree is built on the store's worker thread, off the event loop (see
    # `viewer.app.create_app`). `root_path` is the prefix this app is mounted under ("" when it
    # is served at the root), so every URL in the page can be built from it.
    try:
        tag_tree = await request.app.state.runner.run(build_tag_tree)
    except RunnerClosed as error:
        raise HTTPException(
            status_code=503, detail="The viewer is shutting down."
        ) from error
    root_path = request.scope.get("root_path", "").rstrip("/")
    return request.app.state.templates.TemplateResponse(
        request, "index.html", {"tag_tree": tag_tree, "root_path": root_path}
    )
