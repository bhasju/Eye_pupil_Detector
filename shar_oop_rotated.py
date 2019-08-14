import numpy as np
import cv2
import math
location = '/home/bhaskar/opencv-3.4.2/data/haarcascades/'
face_cascade =cv2.CascadeClassifier(location+'/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(location+'/haarcascade_eye.xml')


class sharingan():
	def __init__(self,img):
		self.rows, self.cols, a = img.shape
		self.M = cv2.getRotationMatrix2D((self.cols/2,self.rows/2),90,1)
		self.kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(4,9))
		self.kernel2= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
		self.symmetryfactor = 0.6
		self.fillfactor = 0.2
		self.positionfactor = .3
		self.eye_h_r,self.eye_w_r=range(70,90),range(70,90)
		return
	# def true_pupil(self,contours,roi_gray_eye):
	# 	eye_center = roi_gray_eye.shape[0]
	# 	return contours[1]
	
	def get_eyes(self,gray):
		eyes = eye_cascade.detectMultiScale(gray,2,7)
		##print (eyes)
		true_eyes = self.elm_fp(eyes)
		return true_eyes
	def elm_fp(self,eyes):
		# eye_area_dict = {}
		# for (x,y,w,h) in eyes:
		# 	eye_area_dict[(w*h)] = (x,y,w,h)
		# sorted_dict = sorted(eye_area_dict.keys())
		# for i in range(len(eyes)):
		# 	print (eye_area_dict[sorted_dict[i-1]])
		# 	if sorted_dict[i-1]<3600:
		# 		del eye_area_dict[sorted_dict[i-1]]
		# print (eye_area_dict)		

		# if len(eyes)>2:
		# 	eyes = (eye_area_dict[sorted_dict[0]],eye_area_dict[sorted_dict[1]])	
		# return eyes
		true_eyes = []
		i=0
		for (x,y,w,h) in eyes:
		 	if (w in self.eye_w_r and h in self.eye_h_r):
		 		true_eyes.append((x,y,w,h))
		 		i = i+1
		 	if(i==2):
		 		break
		return true_eyes			
	def pupil_finder(self,img,roi_gray_eye,roi_bgr_eye,eye_center,eye_area):
		ret, thresh_gray = cv2.threshold(roi_gray_eye,80,255, cv2.THRESH_BINARY)
		#thresh_gray = cv2.adaptiveThreshold(roi_gray_eye,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,20)
		
		thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, self.kernel1)
		thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, self.kernel2)
		_,contours,_ = cv2.findContours(thresh_gray, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contours=[cv2.convexHull(c) for c in contours]
		contours=[cv2.convexHull(c) for c in contours]
		# im3 = cv2.drawContours(roi_bgr_eye, contours, -1, (0,255,0), 1)
		i = 0	
		for contour in contours:
			area = cv2.contourArea(contour)
			rect = cv2.boundingRect(contour)
			x,y,w,h = rect
			xe,ye = eye_center
			xp,yp = x+w/2,y+h/2


			radius = 0.25 * (w+h)
			roi_hsv_eye= cv2.cvtColor(roi_bgr_eye,cv2.COLOR_BGR2HSV)
			mask = np.zeros(roi_gray_eye.shape,np.uint8)
			cv2.drawContours(mask,[contour],0,255,-1)
			pixelpoints = np.transpose(np.nonzero(mask))
			mean_val = cv2.mean(roi_hsv_eye,mask = mask)
			colour_condition = (((mean_val[0]<=90)&(mean_val[0]>=30)))

			
			area_condition = ((eye_area/(w*h)>8)&(eye_area/(w*h)<40))
			# position_condition = ((abs(xe-xp)<50000)&(abs(ye-yp)<1000))
			# symmetry_condition = (abs(1 - float(w)/float(h)) <= self.symmetryfactor)
			# fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0))))<= self.fillfactor)
			if  area_condition or colour_condition:
				cv2.drawContours(roi_bgr_eye, contours, i, (0,255,0), 1)
				# print(mean_val)
				# print (colour_condition)
				#print (contour)
				break
			i = i+1	

		return #im3
		
	def get_pupil(self,img):
		img = self.rotate(img)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		eyes = self.get_eyes(gray)
		for(x1,y1,w1,h1) in eyes:
			roi_gray_eye = gray[(y1):(y1+h1),(x1):(x1+w1)]
			roi_bgr_eye = img[(y1):(y1+h1),(x1):(x1+w1)]
			eye_center = (x1+w1/2,y1+h1/2)
			eye_area = w1*h1
			# img = 
			self.pupil_finder(img,roi_gray_eye,roi_bgr_eye, eye_center,eye_area)
		return img
	def mod(self, img):
		img = self.get_pupil(img)
		return img
	def rotate(self,frame):
		
		frame = cv2.warpAffine(frame,self.M,(self.cols,self.rows))
		return frame



#main

if __name__=="__main__":
	ret,cap = False,cv2.VideoCapture('ee.mp4')
	while not ret:
		ret,frame=cap.read()
	itachi = sharingan(frame)
	
	while True:
		ret, frame = cap.read()
		if not ret: continue
		img = itachi.get_pupil(frame)
		cv2.imshow('frame',img)
		if cv2.waitKey(0) & 0xFF==ord("q"):
			break
		elif ret==False :
			break	
	cap.release()
	cv2.destroyAllWindows()
			