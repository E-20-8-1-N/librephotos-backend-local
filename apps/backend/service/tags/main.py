import gc
import time

import gevent
from flask import Flask, request
from gevent.pywsgi import WSGIServer
from places365.places365 import Places365
from siglip2.siglip2 import SigLIP2

app = Flask(__name__)

places365_instance = None
siglip2_instance = None
last_request_time = None

# Unload models after 2 minutes of inactivity
IDLE_TIMEOUT_SECONDS = 120


def log(message):
    print(f"tags: {message}")


def _unload_models():
    """Free all loaded tagging models."""
    global places365_instance, siglip2_instance
    if places365_instance is not None:
        log("unloading places365 model to free memory")
        places365_instance.unload()
        places365_instance = None
    if siglip2_instance is not None:
        log("unloading siglip2 model to free memory")
        siglip2_instance.unload()
        siglip2_instance = None
    gc.collect()


def _idle_unloader():
    """Background greenlet that unloads models after idle timeout."""
    global last_request_time
    while True:
        gevent.sleep(30)
        if (
            (places365_instance is not None or siglip2_instance is not None)
            and last_request_time is not None
            and time.time() - last_request_time > IDLE_TIMEOUT_SECONDS
        ):
            _unload_models()


@app.route("/generate-tags", methods=["POST"])
def generate_tags():
    global last_request_time
    last_request_time = time.time()

    try:
        data = request.get_json()
        image_path = data["image_path"]
        confidence = data.get("confidence", 0.4)
        tagging_model = data.get("tagging_model", "places365")
    except Exception as e:
        print(str(e))
        return "", 400

    try:
        if tagging_model == "siglip2":
            global siglip2_instance
            if siglip2_instance is None:
                siglip2_instance = SigLIP2()
            # SigLIP 2 uses cosine similarity (range -1 to 1), not probability scores.
            # Always return the top 10 most relevant tags above a minimum threshold.
            result = siglip2_instance.predict(image_path, threshold=0.05, max_tags=10)
            return {"tags": result}, 201
        else:
            global places365_instance
            if places365_instance is None:
                places365_instance = Places365()
            result = places365_instance.inference_places365(image_path, confidence)
            return {"tags": result}, 201
    except Exception as e:
        print(f"tags: Error processing image {image_path}: {e}")
        return {"error": "Failed to process image"}, 500


@app.route("/health", methods=["GET"])
def health():
    return {"last_request_time": last_request_time}, 200


if __name__ == "__main__":
    log("service starting")
    server = WSGIServer(("0.0.0.0", 8011), app)
    server_thread = gevent.spawn(server.serve_forever)
    idle_thread = gevent.spawn(_idle_unloader)
    gevent.joinall([server_thread, idle_thread])
