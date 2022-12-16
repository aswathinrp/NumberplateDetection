import cv2
class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def _del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame = self.video.read()
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()
    