"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image

import torch
from flask import Flask, render_template, request, redirect, flash

RESULT_FOLDER = os.path.join('static', 'result_photo')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = RESULT_FOLDER
i=0
@app.route("/", methods=["GET", "POST"])
def predict():
    global i
    if request.method == "POST":
        if "file" not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files["file"]
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('File not allowed')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            #To resize image
            wpercent = (640/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((640,hsize), Image.ANTIALIAS)

            results = model([img])
            results.render()  # updates results.imgs with boxes and labels
            results.save(save_dir=os.path.join(app.config['UPLOAD_FOLDER']))
            name_image="image0.jpg"
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
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    os.environ['FLASK_ENV']='development'
    app.run(host="0.0.0.0", port=args.port, debug=True)  # debug=True causes Restarting with stat