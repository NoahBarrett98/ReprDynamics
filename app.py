from back_end.ReprD import ReprD
from flask import Flask, jsonify, request, render_template
import os
import click
import numpy as np
from PIL import Image
import glob
import time
import shutil

app = Flask(__name__)

MAX_IMGS = 5000

def OnExitApp(temp_dir):
    temp_dir.cleanup()

def init(save_dir):
    if not os.path.isdir("static/temp"):
        os.mkdir("static/temp")
   
    # read np files and save into temp dir #
    images = np.load(os.path.join(save_dir, "data", "data.npy"))
    if os.path.isdir(os.path.join("static/temp", "img")):
        shutil.rmtree(os.path.join("static/temp", "img"), ignore_errors=True)
    os.mkdir(os.path.join("static/temp", "img"))

    for i, img in enumerate(images):
        im = Image.fromarray(img)
        im.save(os.path.join("static/temp", "img", f"{i}.png"))
    app.config['temp_f'] = "static/temp"
    app.config['img_dir'] = os.path.join("static/temp", "img")

    # initalize ReprD
    # INIT ReprD
    reprd = ReprD(
                save_path=save_dir,
                load_path=os.path.join(save_dir, "snapshots.json")
                )
    app.config['reprd'] = reprd

    app.config["cur_mat"] = -1
    
    
@app.route("/", methods=['GET', 'POST'])
def index():
    samples = [f for f in glob.glob(os.path.join(app.config['img_dir'], "*"))][:MAX_IMGS]
    ids = [os.path.basename(f).split(".")[0] for f in samples][:MAX_IMGS]
    return render_template('index.html', len=len(ids), ids = ids, samples = samples, max_imgs=MAX_IMGS)

@app.route("/array_post",methods=['GET','POST'])
def array_post():
    if request.method=='POST':
        samples = request.form.getlist("selected[]")
        if len(samples):
            i = time.time()
            # compute neighbours #
            neighbours = app.config["reprd"].compute_neighbours(max_neighbours=15,
                                                                knn_algo="ball_tree",
                                                                sample_ids=samples
                                                                  )
            # compute distance matrix #
            mat = app.config["reprd"].distance_matrix(neighbours=neighbours, 
                                                        sample_ids=samples,
                                                        num_neighbours=4,
                                                        distance_metric="euclidean")

            
            app.config["cur_mat"] = mat
            app.config["cur_neighbours"] = app.config["reprd"].get_neighbours(neighbours=neighbours,
                                                                                sample_ids=samples)

        return 'Sucesss', 200
    
    if request.method == 'GET':
        message = {'mat':app.config["cur_mat"],
                    'neighbours':app.config["cur_neighbours"]}
        return jsonify(message)
    

@click.command()
@click.option("--save_dir", type=str, default=None, help="directory of saved outcomes")
def run(save_dir):
    f = init(save_dir)
    app.config["save_dir"] = save_dir
    app.run(debug=False) 






    