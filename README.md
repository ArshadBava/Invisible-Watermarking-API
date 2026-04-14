# FastAPI Watermarking Service (DWT + DCT + DFT)

This project provides a small FastAPI app with:

- **DWT watermarking**: `/embed`, `/extract` (used by the included UI pages)
- **DCT watermarking**: `/dct/embed`, `/dct/extract`
- **DFT watermarking**: `/dft/embed`, `/dft/extract`
- **UI**: `/`, `/embed`, `/extract`

## Run

From `d:\Final_Project`:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

## Project structure

```
Main_Project/
  api.py                 # compatibility entrypoint (exports `app`)
  app/
    main.py              # FastAPI app factory + router wiring
    core/config.py       # constants + templates/static paths
    routers/
      dwt.py             # /embed, /extract
      dct.py             # /dct/embed, /dct/extract
      dft.py             # /dft/embed, /dft/extract
    services/
      dwt_watermark.py   # DWT embed/extract logic
      dct_watermark.py   # DCT embed/extract logic
      dft_watermark.py   # DFT embed/extract logic
    utils/               # validation + metrics + text watermark helper
  templates/
  static/
```

