
import copy
import random
import io
import base64
from PIL import Image
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from Generators.RandomImageGenerator import NumpyRandomImageGenerator, RandImageGenerator
from Generators.EnemyGenerator import NCAEnemyGenerator

app = Flask(__name__)
CORS(app)

generatorModels = [NumpyRandomImageGenerator()]
enemyModels = [NCAEnemyGenerator()]

def encode_image(img : Image):
    buffer = io.BytesIO()
    img.save(buffer, 'png')

    buffer.seek(0)
    
    data = buffer.read()
    data = base64.b64encode(data).decode()
    return data

def encode_image_lst(imgs):
    data_lst = []

    for img in imgs:
        data_lst += [encode_image(img)]

    return data_lst

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/SingleImage", methods = ["GET"])
def SingleImage():
    model = random.choice(generatorModels)
    img = model.Generate(width = 100, height = 100, black_white = False)

    return jsonify({
            'msg': 'success', 
            'size': [img.width, img.height], 
            'format': img.format,
            'img': encode_image(img)
        })


@app.route("/ImageBatch/<batch_size>", methods = ["GET"])
def ImageBatchGET(batch_size):
    model = random.choice(generatorModels)

    if int(batch_size) <= 0:
        return jsonify({
            'msg': 'success', 
            'size': [], 
            'format': None,
            'img': []
        })

    imgs = model.GenerateBatch(width = 100, height = 100, black_white = False, batch_size = int(batch_size))

    data_lst = encode_image_lst(imgs)
    return jsonify({
            'msg': 'success', 
            'size': [imgs[0].width, imgs[0].height], 
            'format': imgs[0].format,
            'imgs': data_lst
        })

@app.route("/ImageBatch", methods = ["POST"])
def ImageBatchPOST():
    model = random.choice(generatorModels)
    files = request.files.getlist("selected_images")
    batch_size = int(request.form['batch_size'])

    for i, file in enumerate(files):
        
        img = Image.open(file.stream)
        files[i] = img

    model = random.choice(generatorModels)
    imgs = model.GenerateBatch(width = 100, height = 100, black_white = False, batch_size = batch_size, selected = files)
        
    data_lst = encode_image_lst(imgs)

    return jsonify({
            'msg': 'success', 
            'size': [imgs[0].width, imgs[0].height], 
            'format': imgs[0].format,
            'imgs': data_lst
        })



@app.route("/SingleImageBrowser", methods = ["GET"])
def SingleImageBrowser():
    model = random.choice(generatorModels)
    img = model.Generate(width = 100, height = 100, black_white = False)
    return f'<img src="data:image/png;base64,{encode_image(img)}">'

@app.route("/ImageBatchBrowser/<batch_size>", methods = ["GET"])
def ImageBatchBrowserGET(batch_size):
    model = random.choice(generatorModels)
    batch_size = int(batch_size)

    data_request = request.get_json()
    if batch_size <= 0:
        return jsonify({
            'msg': 'success', 
            'size': [], 
            'format': None,
            'img': []
        })

    imgs = model.GenerateBatch(width = 100, height = 100, black_white = False, batch_size = batch_size)
     
    data_lst = encode_image_lst(imgs)
    return "".join([f'<img src="data:image/png;base64,{data_}">' for data_ in data_lst])

@app.route("/ImageBatchBrowser", methods = ["POST"])
def ImageBatchBrowserPOST():
    model = random.choice(generatorModels)

    files = request.files.getlist("selected_images")
    batch_size = int(request.form['batch_size'])

    for i, file in enumerate(files):
        
        img = Image.open(file.stream)
        files[i] = img

    model = random.choice(generatorModels)
    imgs = model.GenerateBatch(width = 100, height = 100, black_white = False, batch_size = batch_size, selected = files)
        
    data_lst = encode_image_lst(imgs)
    return "".join([f'<img src="data:image/png;base64,{data_}">' for data_ in data_lst])


@app.route("/Enemies", methods = ["POST"])
def Enemies():
    dataRequest = request.get_json()
    idToImgMap = enemyModels[0].GenerateBatch(enemyTypes = dataRequest["enemyTypes"])
    for key in idToImgMap.keys():
        img = idToImgMap[key] 
        idToImgMap[key] = encode_image(img)
    
    return jsonify({
        'msg': 'success', 
        'enemies': idToImgMap 
    })

@app.route("/EnemiesBrowser", methods = ["POST"])
def EnemiesBrowser():
    dataRequest = request.get_json()
    idToImgMap = enemyModels[0].GenerateBatch(enemyTypes = dataRequest["enemyTypes"])
    for key in idToImgMap.keys():
        img = idToImgMap[key] 
        idToImgMap[key] = encode_image(img)
    
    return "".join([f'<img src="data:image/png;base64,{idToImgMap[key]}">' for key in idToImgMap.keys()])

if __name__ == '__main__':
    app.run(debug=True)
    