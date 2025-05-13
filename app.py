import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import predict_flower  # Use external prediction logic

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/why')
def why():
    return render_template('why.html')


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            label = predict_flower(file_path)
            return render_template("index.html", prediction=label,
                                   image_path=file_path)

    return render_template("index.html", prediction=None, image_path=None)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/how")
def how():
    return render_template("how.html")


if __name__ == "__main__":
    app.run(debug=True)
