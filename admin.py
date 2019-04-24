import cv2
import os
import sys
import numpy as np
from datetime import datetime
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi
import subprocess

class NEW(QDialog):        #new user
    def __init__(self):
        super(NEW, self).__init__()
        loadUi("new_user.ui", self)

    def get_user(self):
    	first_name = self.f_name_label.text()
    	last_name = self.l_name_label.text()
    	#male_radio = self.male_radio.text()
    	#female_radio = self.female_radio.text()
    	#job_title = self.job_title_label.text()
    	#department = self.dpt_combobox.text()
    	return first_name, last_name

class USER(QDialog):        
# Dialog box for entering the users name and key of new dataset.
    def __init__(self):
        super(USER, self).__init__()
        loadUi("user_info.ui", self)

    def get_name_key(self):
        name = self.name_label.text()
        key = int(self.key_label.text())
        return name, key


class ADMIN(QMainWindow):
	def __init__(self):
		super(ADMIN, self).__init__()
		loadUi("mainwindow.ui", self)
		self.image = cv2.imread("sys1.png")
		self.camera_id=0
		self.ret = False
		self.trained_model = 0
		self.dataset_per_subject = 50
		self.draw_text("FACESEC ADMIN CONSOLE", 40, 30, 1, (255,255,255))
		self.display()

		self.face_classifier = cv2.CascadeClassifier("/home/davie/pentest/pentest/obj/davie_project/main/aufr/classifiers/haarcascade_frontalface_default.xml")

		#actions
		self.new_user_btn.setCheckable(True)
		self.generate_btn.setCheckable(True)
		self.extract_btn.setCheckable(True)
		self.training_btn.setCheckable(True)
		self.save_btn.setCheckable(True)

		#events
		self.new_user_btn.clicked.connect(self.new)
		self.generate_btn.clicked.connect(self.generate)
		self.extract_btn.clicked.connect(self.extractor)
		self.training_btn.clicked.connect(self.train)
		self.save_btn.clicked.connect(self.save_image)

		#Algorithms
		self.algo_radio_group.buttonClicked.connect(self.algorithm_radio_changed)
		self.face_rect_radio.setChecked(True)

		#Recognizers
		self.update_recognizer()
		self.assign_algorithms()



	def start_timer(self):
		# start the timer for execution of the OpenCV.
		self.capture = cv2.VideoCapture(self.camera_id)
		self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
		self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		self.timer = QtCore.QTimer()
		if self.generate_btn.isChecked():
			self.timer.timeout.connect(self.save_dataset)
		self.timer.start(5)

	def stop_timer(self):
		# stop timer or come out of the loop.
		self.timer.stop()
		self.ret = False
		self.capture.release()

	def save_image(self):
		# Save image captured using the save button.
		location = "pictures"
		file_type = ".jpg"
		file_name = self.time()+file_type
		os.makedirs(os.path.join(os.getcwd(),location), exist_ok=True)
		cv2.imwrite(os.path.join(os.getcwd(),location,file_name), self.image)
		QMessageBox().about(self, "Image Saved", "Image saved successfully at "+location+"/"+file_name)

	def save_dataset(self):
		#Save images of new dataset generated using generate dataset button.
		location = os.path.join(self.current_path, str(self.dataset_per_subject)+".jpg")
		if self.dataset_per_subject < 1:
			QMessageBox().about(self, "Dataset Generated", "Your response is recorded now you can train the Model \n or Generate New Dataset.")
			self.generate_btn.setText("Generate Dataset")
			self.generate_btn.setChecked(False)
			self.stop_timer()
			self.dataset_per_subject = 50 # again setting max datasets

		if self.generate_btn.isChecked():
			self.ret, self.image = self.capture.read()
			self.image = cv2.flip(self.image, 1)
			faces = self.get_faces()
			self.draw_rectangle(faces)
			if len(faces) is not 1:
				self.draw_text("Only One Person at a time")
			else:
				for (x, y, w, h) in faces:
					cv2.imwrite(location, self.resize_image(self.get_gray_image()[y:y+h, x:x+w], 92, 112))
					self.draw_text("/".join(location.split("/")[-3:]), 20, 20+ self.dataset_per_subject)
					self.dataset_per_subject -= 1
					self.progress_bar_generate.setValue(100 - self.dataset_per_subject*2 % 100)
		self.display()

	def display(self):
		#Display in the canvas, video feed.
		pixImage = self.pix_image(self.image)
		self.video_feed.setPixmap(QtGui.QPixmap.fromImage(pixImage))
		self.video_feed.setScaledContents(True)

	def pix_image(self, image):
		#Converting image from OpenCv to PyQT compatible image.
		qformat = QtGui.QImage.Format_RGB888  # only RGB Image
		if len(image.shape) >= 3:
			r, c, ch = image.shape
		else:
			r, c = image.shape
			qformat = QtGui.QImage.Format_Indexed8
		pixImage = QtGui.QImage(image, c, r, image.strides[0], qformat)
		return pixImage.rgbSwapped()

	def new(self):
		if self.new_user_btn.isChecked():
			register = NEW()
			register.exec_()
			#first_name, last_name, male_radio, female_radio, job_title, department = register.get_user()
			first_name, last_name = register.get_user()
			
		



	def generate(self):
		#Envoke user dialog and enter name and key.
	        if self.generate_btn.isChecked():
	            try:
	                user = USER()
	                user.exec_()
	                name, key = user.get_name_key()
	                self.current_path = os.path.join(os.getcwd(),"datasets",str(key)+"-"+name)
	                os.makedirs(self.current_path, exist_ok=True)
	                self.start_timer()
	                self.generate_btn.setText("Generating")
	            except:
	                msg = QMessageBox()
	                msg.about(self, "User Information", '''Provide Information Please! \n name[string]\n key[integer]''')
	                self.generate_btn.setChecked(False)

	def extractor(self):
		#Start extracting unique facial features using DNN
		if self.extract_btn.isChecked():
			self.extract_btn.setText("Stop Extraction")
			subprocess.run(["gnome-terminal", "-e", "python3 /home/davie/pentest/pentest/obj/davie_project/main/proj1/opencvrec/facial_embeddings.py -i /home/davie/pentest/pentest/obj/davie_project/main/proj1/opencvrec/shithole/ -e output/embeddings.pickle"])
		else:
			self.extract_btn.setText("Extract")
			self.extract_btn.isChecked(False)

	def algorithm_radio_changed(self):
		# When radio button change, either model is training or recognizing in respective algorithm.
		self.assign_algorithms()                                # 1. update current radio button
		self.update_recognizer()                                # 2. update face Recognizer
		#self.read_model()                                       # 3. read trained data of recognizer set in step 2
		if self.training_btn.isChecked():
			self.train()

	def update_recognizer(self):
		#whenever algoritm radio buttons changes this function need to be invoked.
		if self.eigen_algo_radio.isChecked():
			self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
		elif self.fisher_algo_radio.isChecked():
			self.face_recognizer = cv2.face.FisherFaceRecognizer_create()
		else:
			self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()

	def assign_algorithms(self):
		#Assigning anyone of algorithm to current woring algorithm.
		if self.eigen_algo_radio.isChecked():
			self.algorithm = "EIGEN"
		elif self.fisher_algo_radio.isChecked():
			self.algorithm = "FISHER"
		else:
			self.algorithm = "LBPH"

	def save_model(self):
		#Save anyone model.
		try:
			self.face_recognizer.save("training/"+self.algorithm.lower()+"_trained_model.yml")
			msg = self.algorithm+" model trained, stop training or train another model"
			self.trained_model += 1
			self.progress_bar_train.setValue(self.trained_model)
			QMessageBox().about(self, "Training Completed", msg)
		except Exception as e:
			self.print_custom_error("Unable to save Trained Model due to")
			print(e)

	def train(self):
		#When train button is clicked.
		if self.training_btn.isChecked():
			button = self.algo_radio_group.checkedButton()
			button.setEnabled(False)
			self.training_btn.setText("Stop Training")
			os.makedirs("training", exist_ok=True)
			labels, faces = self.get_labels_and_faces()
			try:
				msg = self.algorithm+" model training started"
				QMessageBox().about(self, "Training Started", msg)
				self.face_recognizer.train(faces, np.array(labels))
				self.save_model()
			except Exception as e:
				self.print_custom_error("Unable To Train the Model Due to: ")
				print(e)

			else:
				self.eigen_algo_radio.setEnabled(True)
				self.fisher_algo_radio.setEnabled(True)
				self.lbph_algo_radio.setEnabled(True)
			self.training_btn.setChecked(False)
			self.training_btn.setText("Train Model")

	def get_all_key_name_pairs(self):
		# Get all (key, name) pair of datasets present in datasets.
		return dict([subfolder.split('-') for _, folders, _ in os.walk(os.path.join(os.getcwd(), "datasets")) for subfolder in folders],)

	def absolute_path_generator(self):
		#Generate all path in dataset folder.
		separator = "-"
		for folder, folders, _ in os.walk(os.path.join(os.getcwd(),"datasets")):
			for subfolder in folders:
				subject_path = os.path.join(folder,subfolder)
				key, _ = subfolder.split(separator)
				for image in os.listdir(subject_path):
					absolute_path = os.path.join(subject_path, image)
					yield absolute_path,key

	def get_labels_and_faces(self):
		#Get label and faces.
		labels, faces = [],[]
		for path,key in self.absolute_path_generator():
			faces.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
			labels.append(int(key))
		return labels,faces

	def get_gray_image(self):
		#Convert BGR image to GRAY image.
		return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

	def get_faces(self):
		#Get all faces in a image.
		#variables
		scale_factor = 1.1
		min_neighbors = 8
		min_size = (100, 100) 
		faces = self.face_classifier.detectMultiScale(
							self.get_gray_image(),
							scaleFactor = scale_factor,
							minNeighbors = min_neighbors,
							minSize = min_size)
		return faces

	def draw_rectangle(self, faces):
		#Draw rectangle.
		for (x, y, w, h) in faces:
			roi_gray_original = self.get_gray_image()[y:y + h, x:x + w]
			roi_gray = self.resize_image(roi_gray_original, 92, 112)
			roi_color = self.image[y:y+h, x:x+w]
			cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	def time(self):
		#Get current time.
		return datetime.now().strftime("%d-%b-%Y:%I-%M-%S")

	def draw_text(self, text, x=20, y=20, font_size=2, color = (0, 255, 0)):
		 #Draw text in current image in particular color.
		 cv2.putText(self.image, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.6, color, font_size)

	def resize_image(self, image, width=280, height=280):
		 #Resize image before storing.
		 return cv2.resize(image, (width,height), interpolation = cv2.INTER_CUBIC)

	def print_custom_error(self, msg):
		#Print custom error message/
		print("="*100)
		print(msg)
		print("="*100)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = ADMIN()         
    ui.show()
    sys.exit(app.exec_())  