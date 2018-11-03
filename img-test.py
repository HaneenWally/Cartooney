import numpy as np
import cv2


#img = cv2.imread('1528671407831.jpg')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.CascadeClassifier face_cascade;

face_classifier = cv2.CascadeClassifier('C://Users//Dalia//Anaconda3//Lib//site-packages//cv2//data//haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier ("C://Users//Dalia//Anaconda3//Lib//site-packages//cv2//data//haarcascade_eye.xml")
mouth_classifier=cv2.CascadeClassifier("C://Users//Dalia//Anaconda3//Lib//site-packages//cv2//data//haarcascade_smile.xml")
nose_classifier=cv2.CascadeClassifier("C://Users//Dalia//Anaconda3//Lib//site-packages//cv2//data//haarcascade_nose.xml")


if face_classifier.empty():
   raise IOError('Unable to load the face cascade classifier xml file')
if eye_classifier.empty():
   raise IOError('Unable to load the eye cascade classifier xml file')

if mouth_classifier.empty():
   raise IOError('Unable to load the mouth cascade classifier xml file')
   
if nose_classifier.empty():
   raise IOError('Unable to load the nose cascade classifier xml file')
   
#faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#-----------------------------------------------------------------------------
#       Load and configure mustache (.png with alpha transparency)
#-----------------------------------------------------------------------------
 
# Load our overlay image: mustache.png
imgMustache = cv2.imread('tests//mustache.png',-1)
 
# Create the mask for the mustache
orig_mask = imgMustache[:,:,3]
 
# Create the inverted mask for the mustache
orig_mask_inv = cv2.bitwise_not(orig_mask)
 
# Convert mustache image to BGR
# and save the original image size (used later when re-sizing the image)
imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]
 
#-----------------------------------------------------------------------------



def face_detector (img):
    # Convert Image to Grayscale
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype='uint8')

    faces = face_classifier.detectMultiScale (gray,1.3, 3)

    if faces is ():
        return img
    
    # Given coordinates to detect face and eyes location from ROI
    for (x, y, w, h) in faces:
        #(x,y) : upper-left corner of the face 
        #(x+w,y+h) : bottom-right corner of the face
        #
        cv2.rectangle (img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #roi for eyes
        #detect eyes
        roiE_gray = gray[y: y+h, x: x+w]
        roiE_color = img[y: y+h, x: x+w]
        eyes = eye_classifier.detectMultiScale (roiE_gray,1.8,5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roiE_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            #roi_color = cv2.flip (roi_color, 1)
    #return roi_color

# =============================================================================
#         #roi of nose
#         #detect nose
        roiN_gray = gray[y: y+h, x: x+w]
        roiN_color = img[y: y+h, x: x+w]
        nose = nose_classifier.detectMultiScale(roiN_gray,1.5,5)
		
		
        for (nx, ny, nw, nh) in nose:
            #my=int(my-0.15*h)
            cv2.rectangle(roiN_color,(nx,ny),(nx+nw,ny+nh),(0,255,255),2)
            mustacheWidth =  1.5 * nw
            mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth
 
            # Center the mustache on the bottom of the nose
            x1 = nx - (mustacheWidth/4)
            x2 = nx + nw + (mustacheWidth/4)
            y1 = ny + nh - (mustacheHeight/2)
            y2 = ny + nh + (mustacheHeight/2)
 
            # Check for clipping
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h
 
            # Re-calculate the width and height of the mustache image
            mustacheWidth = int(x2 - x1)
            mustacheHeight =int( y2 - y1)
 
            # Re-size the original image and the masks to the mustache sizes
            # calcualted above
            mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
 
            # take ROI for mustache from background equal to size of mustache image
            roi = roiN_color[int(y1):int(y2), int(x1):int(x2)]
 
            # roi_bg contains the original image only where the mustache is not
            # in the region that is the size of the mustache.
            roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
 
            # roi_fg contains the image of the mustache only where the mustache is
            roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
 
            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg,roi_fg)
 
            # place the joined image, saved to dst back over the original image
            roiN_color[int(y1):int(y2), int(x1):int(x2)] = dst
 
            break
 
# =============================================================================
	
# =============================================================================
#         #roi of mouth
#         #detect mouth
        roiM_gray = gray[y: y+h, x: x+w]
        roiM_color = img[y: y+h, x: x+w]
        mouth = mouth_classifier.detectMultiScale(roiM_gray,1.8,5)
		
		
        for (mx, my, mw, mh) in mouth:
            #my=int(my-0.15*h)
            cv2.rectangle(roiM_color,(mx,my),(mx+mw,my+mh),(255,255,0),2)
            
# =============================================================================
    cv2.imshow('img',img)
    
img=cv2.imread('tests//twins.jpg')

if img is None:
   raise IOError('Unable to load image file')

face_detector(img)
cv2.waitKey(0)
cv2.destroyAllWindows()