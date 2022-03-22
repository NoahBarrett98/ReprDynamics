from back_end.UNNTF import UNNTF

from flask import Flask, jsonify, request, render_template
import atexit
import tempfile
import os
import click
import numpy as np
from PIL import Image
import glob

app = Flask(__name__)

MAX_IMGS = 100

def OnExitApp(temp_dir):
    temp_dir.cleanup()

def init(save_dir):
    if not os.path.isdir("static/temp"):
        os.mkdir("static/temp")
    f = tempfile.TemporaryDirectory(dir = "static/temp")
    atexit.register(OnExitApp, temp_dir=f)

    # read np files and save into temp dir #
    images = np.load(os.path.join(save_dir, "data", "data.npy"))
    if not os.path.isdir(os.path.join(f.name, "img")):
        os.mkdir(os.path.join(f.name, "img"))
    for i, img in enumerate(images):
        im = Image.fromarray(img)
        im.save(os.path.join(f.name, "img", f"{i}.png"))
    app.config['temp_f'] = f
    app.config['img_dir'] = os.path.join(f.name, "img")

    # initalize UNNTF
    # INIT UNNTF
    unntf = UNNTF(
                save_path=save_dir,
                load_path=os.path.join(save_dir, "snapshots.json")
                )
    app.config['unntf'] = unntf

    app.config["cur_mat"] = -1
    
    
@app.route("/", methods=['GET', 'POST'])
def index():
    samples = [f for f in glob.glob(os.path.join(app.config['img_dir'], "*"))]
    ids = [os.path.basename(f).split(".")[0] for f in samples]
    return render_template('index.html', len=len(ids), ids = ids, samples = samples, max_imgs=MAX_IMGS)

@app.route("/array_post",methods=['GET','POST'])
def array_post():
    if request.method=='POST':
        samples = request.form.getlist("selected[]")
        if len(samples):
            mat = app.config["unntf"].distance_matrix(neighbours=os.path.join(app.config["save_dir"], "neighbours.json"), 
                    sample_ids=samples,
                    num_neighbours=10,
                    distance_metric="euclidean")
            app.config["cur_mat"] = mat.tolist()
        return 'Sucesss', 200
    
    if request.method == 'GET':
        message = {'mat':app.config["cur_mat"]}
        return jsonify(message)
    

@click.command()
@click.option("--save_dir", type=str, default=None, help="directory of saved outcomes")
def run(save_dir):
    f = init(save_dir)
    app.config["save_dir"] = save_dir
    app.run(debug=True) 






    