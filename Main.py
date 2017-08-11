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
from threading import Timer

class computationThread(QThread):
    def __init__(self):
        QThread.__init__(self)
    def __del__(self):
        self.wait()
    def run(self):
        self.turning = False
        self.rightVotes = 0
        self.leftVotes = 0
        self.leftMotorSpeed = 50
        self.rightMotorSpeed = 50
        self.ser = serial.Serial('COM3', 12000000, timeout=0.1)
        self.ser.write('E+\nE+\n!M+\n!M+')
        self.image = QtGui.QImage(128, 128, QtGui.QImage.Format_RGB32)
        self.image2 = QtGui.QImage(128, 128, QtGui.QImage.Format_RGB32)
        self.painter = QPainter()
        self.painter2 = QPainter()
        self.moveForward()

        while self.ser.is_open:
            line = self.ser.read(4096)
            self.image.fill(QtGui.qRgb(255,255,255))
            self.image2.fill(QtGui.qRgb(255,255,255))
            skipNextByte = False
            for index, element in enumerate(line):
                if skipNextByte == True:
                    skipNextByte = False
                    continue
                
                byteOne = struct.unpack('B', element)[0]
                validEvent = (byteOne & 0x80) >> 7;
                if validEvent == 0:
                    continue
                
                skipNextByte = True          
                if index+1 >= len(line):
                    break
                byteTwo = struct.unpack('B', line[index+1])[0]
                
                x = byteTwo & 0x7f;
                y = byteOne & 0x7f;
                
                if y < 100:
                    continue
                self.image.setPixel(x,y, QtGui.qRgb(254,0,0))
                self.image2.setPixel(x,y, QtGui.qRgb(254,0,0))

            self.painter.begin(self.image)
            self.painter2.begin(self.image2)
            pen = QPen()
            pen.setWidth(1)
            pen.setColor(QColor(qRgb(254,0,0)))
            self.painter.setPen(QColor(qRgb(0,255,0)))
            self.painter.setPen(pen)
            self.painter2.setPen(pen)

            array = self.QImageToCvMat(self.image)
            
            array2 = self.QImageToCvMat(self.image2)
            templates = ["left_hand_curve.png", "right_hand_curve.png", "straight_lines.png"]
            (startX, startY, endX, endY) = self.templateMatch(array2, templates[0])
#            homographyMatrix = np.matrix('-8.42014397e-01  -9.13836156e-01   1.10386987e+02;1.22808744e-02  -3.46093934e+00   3.46141924e+02;-5.21625321e-05  -1.55665641e-02   1.00000000e+00')
#            img = cv2.warpPerspective(array2, homographyMatrix, (128,128))
##            [x][y][2] is the red component
#            im = np.require(img, np.uint8, 'C')
            self.painter2.setPen(QColor(qRgb(0,0,255)))
            self.painter2.drawRect(startX, startY, endX-startX, endY-startY)
            if startX > 0:
                self.leftVotes += 1
                if self.leftVotes >= 3:    
                    t = Timer(0.75, self.turnLeft90Degrees)
                    t.start()
                else:
                    t = Timer(0.75, self.clearLeftVotes)
                    t.start()

            (startX, startY, endX, endY) = self.templateMatch(array2, templates[1])
            if startX > 0:
                self.rightVotes += 1
                if self.rightVotes >= 3:
                    t = Timer(0.75, self.turnRight90Degrees)
                    t.start()
                else:
                    t = Timer(0.75, self.clearRightVotes)
                    t.start()
            self.painter2.drawRect(startX, startY, endX-startX, endY-startY)
#
            
            totalX = 0
            totalY = 0
            countX = 0
            countY = 0
            for y in range(100,127):
                for x in range(0,127):
                    if array2[y][x][2] == 254:
                        totalX += (x-63)
                        totalY += y
                        countX += 1
                        countY += 1
            
            if countX > 0 and self.turning == False:
                COGx = totalX / countX
                if COGx > 0:
                    self.adjustLeft()
                if COGx < 0:
                    self.adjustRight()
                if COGx == 0:
                    self.moveForward()
                COGx += 63
                self.painter2.setPen(QColor(qRgb(0,0,255)))
                self.painter2.drawLine(COGx,63,63,63)
            self.painter.end()
            self.painter2.end()
            self.emit(SIGNAL('reset'), self.image2)
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

        
    def detectLines(self, img):
        '''  Applies contours and hough transform to detect lines '''
        homographyMatrix = np.matrix('-8.42014397e-01  -9.13836156e-01   1.10386987e+02;1.22808744e-02  -3.46093934e+00   3.46141924e+02;-5.21625321e-05  -1.55665641e-02   1.00000000e+00')
        im = cv2.warpPerspective(img, homographyMatrix, (128,128))
        rows,cols = im.shape[:2]
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,125,255,0)
        thresh = (255-thresh)
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contnumber=0
        image = np.zeros((128, 128, 3), np.uint8)
        image[:] = (255, 255, 255)
        for contour in contours:
            if(len(contour) >= 10):
                cv2.drawContours(image, contours, contnumber, (0,0,255), 1) #draw only contour contnumber
            contnumber+=1
        kernel = np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(image,-1,kernel)

        gray = cv2.cvtColor(dst ,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,100,apertureSize = 3)
        minLineLength = 50
        maxLineGap = 2
        lines = cv2.HoughLinesP(edges,1,np.pi/90,20,minLineLength,maxLineGap)
        return lines        
#    MV0 is left, MV1 is right

    def adjustRight(self):
        if self.turning == True:
            return

        self.ser.write('\n!MV0=35\n!!MV0=45\n!MV1=50\n!!MV1=50')
    def adjustLeft(self):
        if self.turning == True:
            return
        
        self.ser.write('\n!MV0=50\n!!MV0=50\n!MV1=35\n!!MV1=35')
    def moveForward(self):
        print "Move Forward"
        if self.turning == True:
            return
        self.rightMotorSpeed = 50
        self.leftMotorSpeed = 50
        self.ser.write('\n!MV0=50\n!!MV0=50\n!MV1=50\n!!MV1=50')
    def stop(self):
        self.rightMotorSpeed = 50
        self.leftMotorSpeed = 50
        self.ser.write('\n!M-\n!M-')
    def startBot(self):
        self.rightMotorSpeed = 50
        self.leftMotorSpeed = 50
        self.ser.write('\n!M+\n!M+')
        self.moveForward()
    def turnLeft90Degrees(self):
        if self.turning == True:
            return

        print "Left turn start"
        self.leftVotes = 0
        self.turning = True
        self.ser.write('\n!MV0=0\n!!MV0=0\n!MV1=50\n!!MV1=50')
        t = Timer(0.5, self.endTurning)
        t.start()
    def turnRight90Degrees(self):
        if self.turning == True:
            return
        print "Right turn start"
        self.rightVotes = 0
        self.turning = True        
        self.ser.write('\n!MV0=50\n!!MV0=50\n!MV1=0\n!!MV1=0')
        t = Timer(0.5, self.endTurning)
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
