import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import json, time
from flask import Flask, render_template, request, jsonify, Response
import requests
import base64

def gen(camera):
	return camera.getFrame()

class VideoCamera(object):
	def __init__(self):
		self.appClosed = False;
		self.cap = cv2.VideoCapture(0)

		#Getting the classifier
		self.face_cascade = cv2.CascadeClassifier('../cascades/data/haarcascade_frontalface_alt2.xml')
		self.recognizer = cv2.face.LBPHFaceRecognizer_create()
		self.recognizer.read("../emotion_training.yml")

		self.emotions = "Neutral", "Anger", "Contempt", "Disgust", "Fear", "Happy", "Sadness", "Surprise"
		self.emotion_values = [0]*len(self.emotions)

		#Getting the ditionary back in order to get the names 
		self.labels ={}
		#Loads the file with "dictionary" for the label
		with open("../emotion_labels.pkl", "rb") as f:
			old_labels = pickle.load(f)
			#Old_labels were in the form {"person_name": 1} need to invert (invert the key-value pairs)
			self.labels = {v:k for k,v in old_labels.items()}

	def __del__(self):
		self.cap.release()
		cv2.destroyAllWindows()

	def getFrame(self):
		while (True):
			#reads the code frame-by-frame
			ret, frame = self.cap.read()
			#Haarcascade only works with grayscale
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			#To set the size and to detect the faces
			faces=self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

			#Defines the region of interest
			#x, y and width and height
			for (x,y,w,h) in faces:
				#print(x,y,w,h)
				region_of_interest_gray = gray[y:y+h, x:x+w]
				region_of_interest_colour = frame[y:y+h, x:x+w]

				#Begins the predictions
				id_, conf = self.recognizer.predict(region_of_interest_gray)
				if conf >= 25 and conf <= 85:
					#Prints the name
					print(self.labels[id_])

					self.emotion_values[id_] += 1
					#Putting the name in the OpenCV rectangle
					font = cv2.FONT_HERSHEY_SIMPLEX

					name=self.labels[id_]	#Name is the prediction
					colour = (255,255,255)
					stroke = 2

					# 1 is the font size
					cv2.putText(frame, name, (x,y), font, 1, colour, stroke, cv2.LINE_AA)

				#Make the rectangle
				colour = (255,0,0) #BGR
				stroke = 2 #thickness

				#defining starting and ending coordinates of the rectangle
				x_end = x + w
				y_end = y + h
				cv2.rectangle(frame, (x,y), (x_end,y_end), colour, stroke)


			#Show on the screen (both gray and colour)
			#cv2.imshow('frame', frame)
			retVal, jpeg = cv2.imencode('.jpg', frame)
			data = []
			data.append(jpeg.tobytes())
			toYield = data[0]
			yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + toYield + b'\r\n\r\n')
			#Pass to another function which will display this image


			#End when the q key is pressed
			if cv2.waitKey(20) & 0xFF == ord('q'):
				self.appClosed = True;
				break

		#del self
			
app = Flask(__name__)
output=[]

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/detect')
def  detector():
	#Display the new image when we get it
	return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
	app.run(debug=True)

'''
def custom_autopct (pct):
	if pct > 4:
		return ('%.2f%%' % pct)
	else:
		return ''


plt.pie(emotion_values, labels=emotions, autopct=custom_autopct)
plt.show()
'''

