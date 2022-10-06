"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image

import torch
from flask import Flask, render_template, request, redirect

RESULT_FOLDER = os.path.join('static', 'result_photo')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = RESULT_FOLDER
i=0
@app.route("/", methods=["GET", "POST"])
def predict():
    global i
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])

        results.render()  # updates results.imgs with boxes and labels
        results.save(save_dir=os.path.join(app.config['UPLOAD_FOLDER']))
        name_image=f"image0.jpg"
        if i==0: result_image = os.path.join(f"/static/result_photo", name_image)
        else: result_image = os.path.join(f"/static/result_photo{i+1}", name_image)
        i+=1
        return render_template("result.html", result_image=result_image)

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()
    os.environ['FLASK_ENV']='development'
    app.run(host="0.0.0.0", port=args.port, debug=True)  # debug=True causes Restarting with stat