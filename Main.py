import numpy as np
import cv2
import serial
import struct
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys
from PyQt4 import QtGui
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import threading
from collections import deque
class computationThread(QThread):
    def __init__(self):
        QThread.__init__(self)
    def __del__(self):
        self.wait()
    def run(self):
        self.turning = False
        self.record = False
        self.rightVotes = 0
        self.enabled = False
        self.frameNumber = 0
        self.leftVotes = 0
        self.leftMotorSpeed = 50
        self.rightMotorSpeed = 50
        self.ser = serial.Serial('COM3', 12000000, timeout=0.1)
        self.ser.write('E+\nE+\n!M+\n!M+')
        self.image = QtGui.QImage(128, 128, QtGui.QImage.Format_RGB32)
        self.image2 = QtGui.QImage(128, 128, QtGui.QImage.Format_RGB32)
        self.painter = QPainter()
        self.stop()
        self.q = deque([])
        self.totalX = 0
        self.totalY = 0
        self.countX = 0
        self.countY = 0
        self.COGx = 0
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QColor(qRgb(0,255,0)))
        self.painter.begin(self.image)
        self.painter.setPen(pen)
        self.refreshDisplay()
        self.runAlgorithms()
        while self.ser.is_open:
            
            validEvent = 0
            while(validEvent == 0):
                firstByte = self.ser.read(1)
                if(len(firstByte) != 1):
                    continue
                byteOne = struct.unpack('B', firstByte)[0]
                validEvent = (byteOne & 0x80) >> 7;

                
            secondByte = self.ser.read(1)
            byteTwo = struct.unpack('B', secondByte)[0]
                
            x = byteTwo & 0x7f;
            y = byteOne & 0x7f;
                
            if y < 100:
                continue
            self.q.append((x, y))

            self.centroids(x)

            if(len(self.q) > 200):
                (oldx, _) = self.q.popleft()
                self.removeFromCentroidBuffer(oldx)

    def removeEventFromQueue(self):
        if(len(self.q) > 0):
            (oldx, _) = self.q.popleft()
        
    def saveFrame(self, image):
        '''  Saves the image on disk '''

        cv2.imwrite(str(self.frameNumber) +'.png',image)
        self.frameNumber += 1
    def refreshDisplay(self):
        '''  Is called 30 times per second, refreshes the bot display screen'''
        threading.Timer(1.0/30, self.refreshDisplay).start()
        self.image.fill(QtGui.qRgb(255,255,255))
        for x,y in list(self.q):
            self.image.setPixel(x,y, QtGui.qRgb(254,0,0))        
        self.emit(SIGNAL('reset'), self.image)

    def removeFromCentroidBuffer(self, x):
        self.countX-=1
        self.totalX-=(x-63)
        self.COGx = self.totalX / self.countX
        self.COGx += 63
    def centroids(self, newPixelX):
        '''  Determines the centroid of pixels'''
        self.totalX += (newPixelX-63)
        self.countX += 1
        if self.countX > 0 and self.turning == False:
            self.COGx = self.totalX / self.countX
            if self.COGx > 0:
                self.adjustRight()
            if self.COGx < 0:
                self.adjustLeft()
            self.COGx += 63
        self.drawCentroid()

    def drawCentroid(self):
        self.painter.drawLine(63,63,self.COGx,63)
    def detectTurnsWithTemplateMatching(self, array):
        templates = ["left_hand_curve.png", "right_hand_curve.png"]
        (startX, startY, endX, endY) = self.templateMatch(array, templates[0])
        self.painter.drawRect(startX, startY, endX-startX, endY-startY)
        if startX > 0:
            self.leftVotes += 1
            if self.leftVotes >= 2:    
                threading.Timer(1.5, self.turnLeft90Degrees).start()
            else:
                threading.Timer(1.5, self.clearLeftVotes).start()

        (startX, startY, endX, endY) = self.templateMatch(array, templates[1])
        self.painter.drawRect(startX, startY, endX-startX, endY-startY)

        if startX > 0:
            self.rightVotes += 1
            if self.rightVotes >= 2:
                t = threading.Timer(1.5, self.turnRight90Degrees).start()
            else:
                t = threading.Timer(1.5, self.clearRightVotes).start()
        
    def runAlgorithms(self):
        threading.Timer(1.0/30, self.runAlgorithms).start()
#       LANE LINE SIMULATION
#       self.painter.drawLine(100,100,120,127)
#        self.drawCentroid()
        array = self.QImageToCvMat(self.image)
        crop_img = array[100:127, 63:127]
            
#        if(self.turning == False):    
#            self.detectTurnsWithTemplateMatching(array)
            
#        if self.record == True:
#            array = self.QImageToCvMat(self.image)
#
#            self.saveFrame(array)
#        self.detectLines(array)

        self.emit(SIGNAL('reset'), self.image)
    def clearLeftVotes(self):
        self.leftVotes = 0
    def clearRightVotes(self):
        self.rightVotes = 0
    def templateMatch(self, image, template):
        curve = cv2.imread(template)
        template = cv2.cvtColor(curve, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]
        startX = 0
        startY = 0
        endX = 0
        endY = 0
        scale = 1    
        #cv2.imshow("Template", template)
        # loop over the images to find the template in
        # load the image, convert it to grayscale, and initialize the
        # bookkeeping variable to keep track of the matched region
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = None
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            return (startX, startY, endX, endY)
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        
        # check to see if the iteration should be visualized
        # draw a bounding box around the detected region
        clone = np.dstack([edged, edged, edged])
        cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
            	(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
        
        # if we have found a new maximum correlation value, then ipdate
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
            if maxVal < 1500000:
                return (0,0,0,0)
            # unpack the bookkeeping varaible and compute the (x, y) coordinates
            # of the bounding box based on the resized ratio
            (_, maxLoc, r) = found
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
            # draw a bounding box around the detected result and display the image
        return (startX, startY, endX, endY)
          
          
    def QImageToCvMat(self, incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = incomingImage.convertToFormat(4)
        
        width = incomingImage.width()
        height = incomingImage.height()
        
        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        return arr

    def toQImage(self, im, copy=False):
        if im is None:
            return QImage()
    
        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(gray_color_table)
                return qim.copy() if copy else qim
    
            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888);
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32);
                    return qim.copy() if copy else qim
    
        raise NotImplementedException

        
    def detectLines(self, im):
        '''  Applies contours and hough transform to detect lines '''
#        homographyMatrix = np.matrix('-8.42014397e-01  -9.13836156e-01   1.10386987e+02;1.22808744e-02  -3.46093934e+00   3.46141924e+02;-5.21625321e-05  -1.55665641e-02   1.00000000e+00')
#        im = cv2.warpPerspective(img, homographyMatrix, (128,128))
#        rows,cols = im.shape[:2]
#        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#        ret,thresh = cv2.threshold(imgray,125,255,0)
#        thresh = (255-thresh)
#        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#        contnumber=0
#        image = np.zeros((128, 128, 3), np.uint8)
#        image[:] = (255, 255, 255)
#        for contour in contours:
#            if(len(contour) >= 10):
#                cv2.drawContours(image, contours, contnumber, (0,0,255), 4) #draw only contour contnumber
#            contnumber+=1
#        kernel = np.ones((5,5),np.float32)/25
#        dst = cv2.filter2D(image,-1,kernel)
        shapes_grayscale = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        # blur image (this will help clean up noise for Canny Edge Detection)
        shapes_blurred = cv2.GaussianBlur(shapes_grayscale, (5, 5), 1.5)

#       find Canny Edges and show resulting image
        canny_edges = cv2.Canny(shapes_blurred, 100, 200)
        minLineLength = 10
        maxLineGap = 2
        lines = cv2.HoughLines(canny_edges,1,np.pi/90,20,np.array([]), minLineLength,maxLineGap)
        self.drawLinesFromThetas(lines)

    def drawLinesFromThetas(self, lines):
        if lines is not None:
            
            x = 4
            for line in (lines):
                if x < 0:
                    return
                x-=1
                rho = line[0][0]
                theta = line[0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                # these are then scaled so that the lines go off the edges of the image
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                slope = float(y2-y1)/float(x2-x1)
                # y = mx+b
                b = y2-(slope*x2)
                x = (128-b)/slope
            
                self.steer(int(x) , np.rad2deg(theta))
                self.painter.drawLine(x1, y1, x2, y2);
    def steer(self, intersection, angle):
#        print "line meets at", intersection
#        print "theta", angle
        if angle > 90:
#            right lane
            if intersection < 85:
                self.adjustLeft()
            elif intersection > 100:
                if intersection > 140:
                    return
                self.adjustRight()
                
        if angle < 90:
#            left lane                    
            if intersection < 0:
                if intersection < -10:
                    return                                
                self.adjustLeft()
            elif intersection > 20:
                self.adjustRight()
        if angle == 90:
            self.moveForward()
#        else:
#            self.moveForward()
    def drawLines(self,lines):
        if lines is not None:
            a,b,c = lines.shape
            for i in range(a):
                self.painter.drawLine(lines[i][0][0],lines[i][0][1]+100,lines[i][0][2],lines[i][0][3]+100)

        
#    MV0 is left, MV1 is right
    def adjustLeft(self):
        if self.turning == True:
            return
        self.ser.write('\n!MV0=10\n!!MV0=10\n!MV1=20\n!!MV1=20')
    def adjustRight(self):
        if self.turning == True:
            return
        self.ser.write('\n!MV0=20\n!!MV0=20\n!MV1=10\n!!MV1=10')
    def moveForward(self):
        if self.turning == True:
            return
        self.ser.write('\n!MV0=20\n!!MV0=20\n!MV1=20\n!!MV1=20')
    def stop(self):
        self.record = False
        self.ser.write('\n!M-\n!M-')
    def startBot(self):
        self.record = True
        self.ser.write('\n!M+\n!M+')
        self.moveForward()
    def turnLeft90Degrees(self):
        print "Left turn start"
        self.leftVotes = 0
        self.turning = True
        self.ser.write('\n!MV0=0\n!!MV0=0\n!MV1=70\n!!MV1=70')
        t = threading.Timer(0.5, self.endTurning)
        t.start()
    def turnRight90Degrees(self):
        self.rightVotes = 0
        self.turning = True        
        self.ser.write('\n!MV0=70\n!!MV0=70\n!MV1=0\n!!MV1=0')
        t = threading.Timer(0.5, self.endTurning)
        t.start()


    def endTurning(self):
        self.turning = False
        self.moveForward()
        
        


class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50,50,128*7,128*7)
        self.setWindowTitle("eDVS Line detector")
        self.scale = 5
        self.button = QtGui.QPushButton('Stop', self)
        self.button.setGeometry(128*6,128*6,100,30)
        self.buttonStart = QtGui.QPushButton('Start', self)
        self.buttonStart.setGeometry(128*6,128*6+40,100,30)

        self.graphics = QtGui.QGraphicsView(self)
        self.graphics.setGeometry(0,0,128*6,128*6)
        self.scene = QtGui.QGraphicsScene()
        self.image = QtGui.QImage(128, 128, QtGui.QImage.Format_RGB32)
        self.image2 = QtGui.QImage(128, 128, QtGui.QImage.Format_RGB32)
        self.pixmap = QtGui.QGraphicsPixmapItem()
        self.image.fill(QtGui.qRgb(127,127,127))
        tempPixmap = QtGui.QPixmap(1,1)
        tempPixmap.convertFromImage(self.image)
        tempPixmap = tempPixmap.scaled(128*self.scale,128*self.scale)
        self.pixmap.setPixmap(tempPixmap)
        self.scene.addItem(self.pixmap)    
        self.graphics.setScene(self.scene)
        self.show()
        self.get_thread = computationThread()
        self.button.clicked.connect(self.get_thread.stop)
        self.buttonStart.clicked.connect(self.get_thread.startBot)
        self.get_thread.start()
        self.connect(self.get_thread, SIGNAL("reset"), self.reset)
    def reset(self, image):
        tempPixmap = QtGui.QPixmap(1,1)
        tempPixmap.convertFromImage(image)
        tempPixmap = tempPixmap.scaled(128*self.scale,128*self.scale)
        self.pixmap.setPixmap(tempPixmap)

def main():
    app = QtGui.QApplication(sys.argv)
    form = Window()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()




#####HOMOGRAPHY MATRIX
#homographyMatrix = [[ -8.42014397e-01  -9.13836156e-01   1.10386987e+02] [  1.22808744e-02  -3.46093934e+00   3.46141924e+02] [ -5.21625321e-05  -1.55665641e-02   1.00000000e+00]]
#homographyMatrix = np.matrix('-8.42014397e-01  -9.13836156e-01   1.10386987e+02;1.22808744e-02  -3.46093934e+00   3.46141924e+02;-5.21625321e-05  -1.55665641e-02   1.00000000e+00')

### PLAYGROUND

###HOUGH

#imgpath = 'C:/Users/sayegh/warpView.jpg'
#img = cv2.imread(imgpath)
#
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,150,apertureSize = 3)
#print img.shape[1]
#print img.shape
#minLineLength=img.shape[1]-300
#lines = cv2.HoughLinesP(image=edges,rho=0.02,theta=np.pi/500, threshold=10,lines=np.array([]), minLineLength=minLineLength,maxLineGap=100)
#print lines
#a,b,c = lines.shape
#for i in range(a):
#    cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
#
#
#cv2.imshow('result', img)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()



#####CONTOURS
#im = cv2.imread('C:/Users/sayegh/testImage.jpg')
#rows,cols = im.shape[:2]
#imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(imgray,125,255,0)
#thresh = (255-thresh)
#thresh2=thresh.copy()
#im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
##cv2.imshow('image1',im)
##cv2.imshow('image3',thresh2)
##cv2.drawContours(im, contours, -1, (0,255,0), 1) #draw all contours
#contnumber=0
#image = np.zeros((128, 128, 3), np.uint8)
#image[:] = (255, 255, 255)
#
#
#for contour in contours:
#    if(len(contour) >= 5):
#        print "Contour:", contour
#        cv2.drawContours(image, contours, contnumber, (0,0,255), 1) #draw only contour contnumber
#
#    contnumber+=1
#    
#cv2.imwrite('res.jpg',image)
#
#
#imgpath = 'res.jpg'
#img = cv2.imread(imgpath)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray,50,100,apertureSize = 3)
#cv2.imwrite('gray.jpg',edges)
#
#minLineLength = 10
#maxLineGap = 2
#lines = cv2.HoughLinesP(edges,0.01,np.pi/180,15,minLineLength,maxLineGap)
#print len(lines)
#for line in lines:
#    for x1,y1,x2,y2 in line:
#        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
#        print x1, y1, x2, y2
#cv2.imwrite('res3.jpg',img)
#
##
##minLineLength = 5
##maxLineGap = 20
##gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
##
##lines = cv2.HoughLinesP(gray,0.1,np.pi/180,20,minLineLength,maxLineGap)
##for line in lines:
##    for x1,y1,x2,y2 in line:
##        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),1)
##        print x1,  y1, x2, y2
##cv2.imwrite('res2.jpg',image)
##index = 0
##for contour in contours:
##    [vx,vy,x,y] = cv2.fitLine(contours[index], cv2.DIST_L2,0,0.01,0.01)
##    lefty = int((-x*vy/vx) + y)
##    index+=1
##    righty = int(((cols-x)*vy/vx)+y)
##    cv2.line(im,(cols-1,righty),(0,letfty),(0,255,255),2)
#
