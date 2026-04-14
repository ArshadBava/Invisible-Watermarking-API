from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import STATIC_DIR, TEMPLATES_DIR
from app.routers.dwt import router as dwt_router
from app.routers.dct import router as dct_router
from app.routers.dft import router as dft_router
from app.routers.robustness import router as robustness_router


def create_app() -> FastAPI:
    app = FastAPI()

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request):
        return templates.TemplateResponse("embed.html", {"request": request})

    @app.get("/embed", response_class=HTMLResponse)
    async def get_embed_ui(request: Request):
        return templates.TemplateResponse("embed.html", {"request": request})

    @app.get("/extract", response_class=HTMLResponse)
    async def get_extract_ui(request: Request):
        return templates.TemplateResponse("extract.html", {"request": request})

    @app.get("/robustness", response_class=HTMLResponse)
    async def get_robustness_ui(request: Request):
        return templates.TemplateResponse("robustness.html", {"request": request})

    app.include_router(dwt_router)
    app.include_router(dct_router)
    app.include_router(dft_router)
    app.include_router(robustness_router)
    return app


app = create_app()

