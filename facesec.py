import cv2
import os
import sys
import numpy as np
from datetime import datetime
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi
import subprocess

class USER(QDialog):        
    #Dialog box for entering name and key of new dataset.
    """USER Dialog """
    def __init__(self):
        super(USER, self).__init__()
        loadUi("user_info.ui", self)

    def get_name_key(self):
        name = self.name_label.text()
        key = int(self.key_label.text())
        return name, key 
        
class FACESEC(QMainWindow):       
# Main application Class
    def __init__(self):
        super(FACESEC, self).__init__()
        loadUi("dave.ui", self)
        self.face_classifier = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml") 
        self.image = cv2.imread("/home/davie/pentest/pentest/obj/davie_project/main/proj1/opencvrec/africa-blue.jpg")
        self.camera_id = 0
        self.ret = False
        self.draw_text("FACESEC BIOMETRIC AUTHENTICATION", 40, 30, 1, (255,255,255))
        self.display()

        # Actions 
        self.recognize_btn.setCheckable(True)
        self.authentication_btn.setCheckable(True)
        
        # Algorithms
        self.algo_radio_group.buttonClicked.connect(self.algorithm_radio_changed)
        

        # Recangle
        self.face_rect_radio.setChecked(True)

        #more
        self.help_btn.setCheckable(True)
        self.about_btn.setCheckable(True)
        self.exit_btn.setCheckable(True)

        # Events
        self.recognize_btn.clicked.connect(self.recognize)
        self.authentication_btn.clicked.connect(self.authenticate)
        #self.admin_btn.clicked.connect(self.login)
        self.help_btn.clicked.connect(self.help_info)
        self.about_btn.clicked.connect(self.about_info)

        # Recognizers
        self.update_recognizer()
        self.assign_algorithms()

    def start_timer(self):      # start the timeer for execution.
        self.capture = cv2.VideoCapture(self.camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.timer = QtCore.QTimer()
        if self.recognize_btn.isChecked():
            self.timer.timeout.connect(self.update_image)
        self.timer.start(5)

    def stop_timer(self):       # stop timer or come out of the loop.
        self.timer.stop()
        self.ret = False
        self.capture.release()
        
    def update_image(self):     # update canvas every time according to time set in the timer.
        if self.recognize_btn.isChecked():
            self.ret, self.image = self.capture.read()
            self.image = cv2.flip(self.image, 1)
            faces = self.get_faces()
            self.draw_rectangle(faces)
        self.display()

    def save_image(self):       # Save image captured using the save button.
        location = "pictures"
        file_type = ".jpg"
        file_name = self.time()+file_type # a.jpg
        os.makedirs(os.path.join(os.getcwd(),location), exist_ok=True)
        cv2.imwrite(os.path.join(os.getcwd(),location,file_name), self.image)
        QMessageBox().about(self, "Image Saved", "Image saved successfully at "+location+"/"+file_name)


    def display(self):      # Display in the canvas, video feed.
        pixImage = self.pix_image(self.image)
        self.video_feed.setPixmap(QtGui.QPixmap.fromImage(pixImage))
        self.video_feed.setScaledContents(True)

    def pix_image(self, image): # Converting image from OpenCv to PyQT compatible image.
        qformat = QtGui.QImage.Format_RGB888  # only RGB Image
        if len(image.shape) >= 3:
            r, c, ch = image.shape
        else:
            r, c = image.shape
            qformat = QtGui.QImage.Format_Indexed8
        pixImage = QtGui.QImage(image, c, r, image.strides[0], qformat)
        return pixImage.rgbSwapped()

    
    def algorithm_radio_changed(self):      # When radio button change, either model is training or recognizing in respective algorithm.
        self.assign_algorithms()                                # 1. update current radio button
        self.update_recognizer()                                # 2. update face Recognizer
        self.read_model()                                       # 3. read trained data of recognizer set in step 2

    def update_recognizer(self):                                # whenever algoritm radio buttons changes this function need to be invoked.
        if self.eigen_algo_radio.isChecked():
            self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
        elif self.fisher_algo_radio.isChecked():
            self.face_recognizer = cv2.face.FisherFaceRecognizer_create()
        else:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    def assign_algorithms(self):        # Assigning anyone of algorithm to current woring algorithm.
        if self.eigen_algo_radio.isChecked():
            self.algorithm = "EIGEN"
        elif self.fisher_algo_radio.isChecked():
            self.algorithm = "FISHER"
        else:
            self.algorithm = "LBPH"

    def read_model(self):       # Reading trained model.
        if self.recognize_btn.isChecked():
            try:                                       # Need to to invoked when algoritm radio button change
                self.face_recognizer.read("training/"+self.algorithm.lower()+"_trained_model.yml")
            except Exception as e:
                self.print_custom_error("Unable to read Trained Model due to")
                print(e)
    
    def save_model(self):       # Save anyone model.
        try:
            self.face_recognizer.save("training/"+self.algorithm.lower()+"_trained_model.yml")
            msg = self.algorithm+" model trained, stop training or train another model"
            self.trained_model += 1
            self.progress_bar_train.setValue(self.trained_model)
            QMessageBox().about(self, "Training Completed", msg)
        except Exception as e:
            self.print_custom_error("Unable to save Trained Model due to")
            print(e)
    
    def recognize(self):        # When recognized button is called.
        if self.recognize_btn.isChecked():
            self.start_timer()
            self.recognize_btn.setText("Stop Recognition")
            self.read_model()
        else:
            self.recognize_btn.setText("Recognize Face")
            self.stop_timer()

    def authenticate(self):
        if self.authentication_btn.isChecked():
            #subprocess.run(["gnome-terminal", "-e", "python3 eyedetect.py"])
            subprocess.run(["gnome-terminal", "-e", "python3 /home/davie/pentest/pentest/obj/davie_project/security/whatsapp_msg.py"])


        else:
            #self.authentication_btn.isChecked(False)
            self.authentication_btn.setText("Stop Authentication")
            #subprocess.run(["gnome-terminal", "-e", "python3 /home/davie/pentest/pentest/obj/davie_project/security/whatsapp_msg.py"])
            subprocess.run(["gnome-terminal", "-e", "python3 eyedetect.py"])
    
    def get_all_key_name_pairs(self):       # Get all (key, name) pair of datasets present in datasets.
        return dict([subfolder.split('-') for _, folders, _ in os.walk(os.path.join(os.getcwd(), "datasets")) for subfolder in folders],)
        
    def absolute_path_generator(self):      # Generate all path in dataset folder.
        separator = "-"
        for folder, folders, _ in os.walk(os.path.join(os.getcwd(),"datasets")):
            for subfolder in folders:
                subject_path = os.path.join(folder,subfolder)
                key, _ = subfolder.split(separator)
                for image in os.listdir(subject_path):
                    absolute_path = os.path.join(subject_path, image)
                    yield absolute_path,key

    def get_labels_and_faces(self):     # Get label and faces.
        labels, faces = [],[]
        for path,key in self.absolute_path_generator():
            faces.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))
            labels.append(int(key))
        return labels,faces

    def get_gray_image(self):       # Convert BGR image to GRAY image.
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def get_faces(self):        # Get all faces in a image.
        # variables
        scale_factor = 1.1
        min_neighbors = 8
        min_size = (100, 100) 

        faces = self.face_classifier.detectMultiScale(
        					self.get_gray_image(),
        					scaleFactor = scale_factor,
        					minNeighbors = min_neighbors,
        					minSize = min_size)

        return faces

    

    def draw_rectangle(self, faces):        # Draw rectangle on the face
        for (x, y, w, h) in faces:
            roi_gray_original = self.get_gray_image()[y:y + h, x:x + w]
            roi_gray = self.resize_image(roi_gray_original, 92, 112)
            roi_color = self.image[y:y+h, x:x+w]
            if self.recognize_btn.isChecked():
                try:
                    predicted, confidence = self.face_recognizer.predict(roi_gray)
                    name = self.get_all_key_name_pairs().get(str(predicted))
                    self.text2("Recognizing using: "+self.algorithm, 70,50)
                    if self.lbph_algo_radio.isChecked():
                        if confidence > 80:
                            msg = "Unknown"
                        else:
                            confidence = "{:.2f}".format(100 - confidence)
                            msg = name
                        self.progress_bar_recognize.setValue(float(confidence))
                    else:
                        msg = name
                        self.progress_bar_recognize.setValue(int(confidence%100))
                        confidence = "{:.2f}".format(confidence)

                    self.draw_text(msg, x-5,y-5)
                except Exception as e:
                    self.print_custom_error("Unable to Pridict due to")
                    print(e)

                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    def time(self):     # Get current time.
        return datetime.now().strftime("%d-%b-%Y:%I-%M-%S")

    def draw_text(self, text, x=20, y=20, font_size=2, color = (255, 0, 0)): # Draw text in current image in particular color.
        cv2.putText(self.image, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.6, color, font_size)
        f = open("name.txt", "w")
        f.write(text)

    def text2(self, text, x=20, y=20, font_size=2, color=(0,0,255)):
        cv2.putText(self.image, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.6, color, font_size)

    def resize_image(self, image, width=280, height=280): # Resize image before storing.
        return cv2.resize(image, (width,height), interpolation = cv2.INTER_CUBIC)

    def print_custom_error(self, msg):      # Print custom error message/
        print("="*100)
        print(msg)
        print("="*100)

    def recording(self):        # Record Video when either recognizing or generating.
        if self.ret:
            self.video_output.write(self.image)

   #Menu information of the about button
    def about_info(self):
        msg_box = QMessageBox()
        msg_box.setText('''
            ML-based face recognition system, is a security application
            that is used to grant access to highly classified points in an organization.
            it uses computer vision, machine learning and
            deep learning to acheive its purpose. 
            This project has been developed by 
            two AI addicts and cybersecurity students:
            Davie and Sharlet!
            ''')
        msg_box.setInformativeText('''
            Jaramogi Oginga Odinga University of Science and Technology(jooust).
            school of informatics and innovative systems.
            Department of computer science and software engineering.
            Team: Davie K. and Sharlet K.
            Supervisor: Dr. Richard O.
            Date:  9th April 2019
            ''')
        msg_box.setWindowTitle("About ML-Based Face Biometric")
        msg_box.exec_()

    #help information button
    def help_info(self):
        msg_box = QMessageBox()
        msg_box.setText('''
        This system is capable of detecting human faces, generating datasets from detected face, applying DNN to extract unique facial features and machine learning classifier to distiguish these features and perform facial recognition based of machine learning classifier results.
        Only the recognied individual are authenticated to access a specified point by the system!.
            ''')
        msg_box.setInformativeText('''
        To use facial authentication system:
        [1] Click the Recognize face button.
        [2] Once recognized click stop recognition.
        [3] Click Authentication button.
        [4] Finally click Access button to gain entry.
        [5] Incase of any problem and authentication issues, kindly
            contact the system adminstrator for further assistance.
            +254724************
            ''')
        msg_box.setWindowTitle("Help")
        msg_box.exec_()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = FACESEC()
    ui.show()
    sys.exit(app.exec_())
