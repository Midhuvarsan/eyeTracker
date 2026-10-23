"""
Web app for the retinal screening pipeline (backend + frontend in one server).

Put this file, the static/ folder, main.py and disease_info.py in the same folder, then:

    pip install -r requirements.txt
    python app.py

and open http://127.0.0.1:5000 in your browser.

Endpoints
    GET  /                 the web page
    GET  /api/health       model status (loading / ready / error)
    GET  /api/samples      images in the project folder you can try with one click
    GET  /samples/<name>   one of those images
    POST /api/predict      multipart: image  OR  left + right ; tta=1|0
    POST /api/explain      json: {token, code, grid}  -> region heatmaps (slow on CPU)
"""
import base64
import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict

import cv2
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

# Load .env file
load_dotenv()

import main as pipeline  # your existing pipeline: load_model, occlusion_map, overlay
from disease_info import F1, INFO
from llm_explain import explain_prediction

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
SAMPLE_EXT = (".png", ".jpg", ".jpeg")
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", 16))
MAX_SIDE = int(os.environ.get("MAX_SIDE", 1600))  # downscale huge uploads; the model works at 448 px anyway

app = Flask(__name__, static_folder=STATIC, static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

# One model, one inference at a time (the model is not thread-safe on CPU/GPU).
infer_lock = threading.Lock()
_model = {"obj": None, "error": None, "loading": False}
SESSIONS = OrderedDict()  # token -> uploaded images + results, so /api/explain can reuse them


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def load_model_once():
    with infer_lock:
        if _model["obj"] is not None:
            return _model["obj"]
        _model["loading"], _model["error"] = True, None
        try:
            _model["obj"] = pipeline.load_model()
        except Exception as e:  # noqa: BLE001
            _model["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            _model["loading"] = False
        return _model["obj"]


def read_image(file_storage):
    try:
        img = Image.open(file_storage.stream)
        img.load()
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise ApiError(f"“{file_storage.filename}” isn’t a readable image. Use a JPG, PNG or TIFF photo.")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def to_data_uri(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ApiError("Could not encode the heatmap image.", 500)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def info_for(code):
    base = INFO.get(code, {"name": code, "what": "", "signs": "", "risk": "", "next": ""})
    return {
        "code": code,
        "name": base["name"],
        "what": base["what"],
        "signs": base["signs"],
        "risk": base["risk"],
        "next": base["next"],
        "f1": F1.get(code),
        "unreliable": F1.get(code, 1.0) < 0.5,
    }


def list_samples():
    out = []
    for name in sorted(os.listdir(BASE)):
        low = name.lower()
        if low.endswith(SAMPLE_EXT) and low != "result.png":
            label = os.path.splitext(name)[0].replace("_", " ").replace("-", " ").strip()
            out.append({"name": name, "label": label[:1].upper() + label[1:]})
    return out


def remember(session):
    token = uuid.uuid4().hex
    SESSIONS[token] = session
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return token


# ----------------------------------------------------------------------------
# routes
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/health")
def health():
    return jsonify(
        ready=_model["obj"] is not None,
        loading=_model["loading"],
        error=_model["error"],
        device=pipeline.DEVICE,
    )


@app.get("/api/samples")
def samples():
    return jsonify(list_samples())


@app.get("/samples/<path:name>")
def sample_file(name):
    if name not in {s["name"] for s in list_samples()}:
        raise ApiError("That sample doesn’t exist.", 404)
    return send_from_directory(BASE, name)


@app.post("/api/predict")
def api_predict():
    files = request.files
    if files.get("left") and files.get("right"):
        left, right, single = read_image(files["left"]), read_image(files["right"]), False
    else:
        one = files.get("image") or files.get("left") or files.get("right")
        if one is None:
            raise ApiError("Add a fundus photo before analyzing.")
        left = right = read_image(one)
        single = True
    tta = request.form.get("tta", "1") != "0"

    model = load_model_once()
    t0 = time.time()
    with infer_lock:
        probs, preds = model.predict(left, right, tta=tta)
    seconds = time.time() - t0
    probs = {k: float(v) for k, v in probs.items()}
    preds = {k: bool(v) for k, v in preds.items()}

    ranked = sorted(probs, key=probs.get, reverse=True)
    flagged = [c for c in ranked if preds.get(c) and c != "N"]
    normal = bool(preds.get("N"))
    verdict = "disease" if flagged else ("normal" if normal else "inconclusive")
    detail_codes = flagged if flagged else (["N"] if normal else [])

    token = remember({"left": left, "right": right, "single": single,
                      "probs": probs, "flagged": flagged, "cache": {}})
    return jsonify(
        token=token,
        verdict=verdict,
        single=single,
        conflict=bool(normal and flagged),
        seconds=seconds,
        device=pipeline.DEVICE,
        classes=[{**info_for(c), "prob": probs[c], "flagged": preds.get(c, False)} for c in ranked],
        details=[info_for(c) for c in detail_codes],
        explain_code=flagged[0] if flagged else None,
        eyes=["left"] if single else ["left", "right"],
    )


@app.post("/api/explain")
def api_explain():
    data = request.get_json(silent=True) or {}
    session = SESSIONS.get(data.get("token"))
    if session is None:
        raise ApiError("This analysis has expired. Analyze the image again.", 404)
    code = data.get("code") or (session["flagged"][0] if session["flagged"] else None)
    if code not in session["probs"]:
        raise ApiError("Nothing was flagged, so there is no region map to show.")
    try:
        grid = min(10, max(4, int(data.get("grid", 6))))
    except (TypeError, ValueError):
        grid = 6

    key = (code, grid)
    if key not in session["cache"]:
        model = load_model_once()
        eyes = ["left"] if session["single"] else ["left", "right"]
        result = []
        for which in eyes:
            with infer_lock:
                heat, arr = pipeline.occlusion_map(model, session["left"], session["right"], code, which, grid)
            original, mixed = pipeline.overlay(arr, heat)
            result.append({"eye": which, "original": to_data_uri(original), "overlay": to_data_uri(mixed)})
        session["cache"][key] = result
    return jsonify(code=code, name=info_for(code)["name"], eyes=session["cache"][key])


@app.post("/api/llm-explain")
def api_llm_explain():
    data = request.get_json(silent=True) or {}
    session = SESSIONS.get(data.get("token"))
    if session is None:
        raise ApiError("This analysis has expired. Analyze the image again.", 404)
    code = data.get("code")
    if not code or code not in session["probs"]:
        raise ApiError("Unknown disease code.", 400)
    prob = session["probs"][code]
    single = session.get("single", False)
    result = explain_prediction(code, prob, session["probs"], single_eye=single)
    return jsonify(result)


# ----------------------------------------------------------------------------
# errors always come back as JSON so the page can show a clear message
# ----------------------------------------------------------------------------
@app.errorhandler(ApiError)
def handle_api_error(e):
    return jsonify(error=e.message), e.status


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="That file is too large. The limit is 25 MB."), 413


@app.errorhandler(Exception)
def handle_unexpected(e):
    if isinstance(e, HTTPException):
        return e
    traceback.print_exc()
    return jsonify(error=f"The analysis failed: {type(e).__name__}: {e}"), 500


def _preload():
    try:
        load_model_once()
    except Exception:  # noqa: BLE001  (error is stored and shown in the page)
        traceback.print_exc()


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t", "yes")

    threading.Thread(target=_preload, daemon=True).start()
    print(f"\nOpen http://{host}:{port} in your browser (the model loads in the background).\n")
    app.run(host=host, port=port, debug=debug, threaded=True)
