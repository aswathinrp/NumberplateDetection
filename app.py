from flask import Flask,render_template,Response,request
from camera import Video 
import os
# from deeplearning import OCR
import numberplatedetection

app = Flask(__name__)

BASE_PATH=os.getcwd()
UPLOAD_PATH=os.path.join(BASE_PATH,'static/upload/')

@app.route('/',methods=['POST','GET'])

def index():
    if request.method=='POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename 
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        # text = OCR(path_save,filename)
        return render_template('index.html')
    return render_template('index.html')

# def gen(camera):
#     while True:
#         frame=camera.get_frame()
#         yield(b'--frame\r\n'
#               b'Content-Type: image/jpeg\r\n\r\n'+ frame +
#               b'\r\n\r\n')
# @app.route('/video')
# def video():
#     return Response(gen(Video())),
#     mimetype = 'multipart/x-mixed-replace;boundary=frame'
    
if __name__ == "__main__":
    app.run(debug=True)

