import numpy as np
import cv2
import serial
import struct
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys
from PyQt4 import QtGui
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

class computationThread(QThread):
    def __init__(self):
        QThread.__init__(self)
    def __del__(self):
        self.wait()
    def run(self):
        ser = serial.Serial('COM3', 12000000, timeout=0.05)
        ser.write('E+\nE+')
        self.image = QtGui.QImage(128, 128, QtGui.QImage.Format_RGB32)
        self.image2 = QtGui.QImage(128, 128, QtGui.QImage.Format_RGB32)
        self.painter = QPainter()
        self.painter2 = QPainter()

        while ser.is_open:
            line = ser.read(4096)
            self.image.fill(QtGui.qRgb(255,255,255))
            for i,k in zip(line[0::2], line[1::2]):
                byteOne = struct.unpack('B', i)[0]
                byteTwo = struct.unpack('B', k)[0]
        
                validEvent = (byteOne & 0x80) >> 7;
                if validEvent == 0:
                    continue;
                y = byteOne & 0x7f;
                x = byteTwo & 0x7f;
#                if y < 100:
#                    continue
#                polarity = (byteTwo & 0x80) >> 7;
                self.image.setPixel(x,y, QtGui.qRgb(0,0,0))

            self.painter.begin(self.image)
            self.painter.setPen(QColor(qRgb(0,255,0)))


            array = self.QImageToCvMat(self.image)
            
            lines = self.detectLines(array)
            if np.any(lines) == None:
                self.painter.end()
                self.emit(SIGNAL('reset'), self.image)
                continue
            
#            left_lane = []
#            right_lane = []
#            for line in lines:
#                for x1,y1,x2,y2 in line:
#                    if y1 > 96 and y2 > 96:
#                        if x1 <= 64 or x2 <= 64:
#                            self.painter.setPen(QColor(qRgb(255,255,0)))
#                            self.painter2.setPen(QColor(qRgb(255,255,0)))
#                            point = []
#                            point.append(x1)
#                            point.append(y1)
#                            left_lane.append(point)
#                            point = []
#                            point.append(x2)
#                            point.append(y2)                            
#                            left_lane.append(point)
#                        else:
#                            pen = QPen()
#                            pen.setWidth(4)
#                            pen.setColor(QColor(qRgb(255,0,0)))
#                            self.painter.setPen(pen)
#                            self.painter2.setPen(pen)
#                            point = []
#                            point.append(x1)
#                            point.append(y1)
#                            right_lane.append(point)
#                            point = []
#                            point.append(x2)
#                            point.append(y2)                            
#                            right_lane.append(point)
#                    else:
#                        self.painter.setPen(QColor(qRgb(0,255,0)))
#                        self.painter2.setPen(QColor(qRgb(0,255,0)))
#                    self.painter.drawLine(x1,y1,x2,y2)
#                    self.painter2.drawLine(x1,y1,x2,y2)
#                    
#
#            left = np.array(left_lane)
#            if len(left) != 0:
#                [vx,vy,x,y] = cv2.fitLine(left,cv2.DIST_L2,0,0.01,0.01)
#                lefty = int((-x*vy/vx) + y)
#                righty = int(((128-x)*vy/vx)+y)
#                self.painter.setPen(QColor(qRgb(0,0,255)))
#                self.painter.drawLine(127,righty,0,lefty)
#                
#            right= np.array(right_lane)
#            
#            if len(right) != 0:
#                [vx,vy,x,y] = cv2.fitLine(right,cv2.DIST_L2,0,0.01,0.01)
#                lefty = int((-x*vy/vx) + y)
#                righty = int(((128-x)*vy/vx)+y)
#                self.painter.setPen(QColor(qRgb(0,0,255)))
#                self.painter.drawLine(127,righty,0,lefty)

            for line in lines:
                for x1,y1,x2,y2 in line:
                    self.painter.setPen(QColor(qRgb(255,0,0)))
                    self.painter.drawLine(x1,y1,x2,y2)

            self.painter.end()
            self.emit(SIGNAL('reset'), self.image)


    def QImageToCvMat(self, incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        incomingImage = incomingImage.convertToFormat(4)
        
        width = incomingImage.width()
        height = incomingImage.height()
        
        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        return arr
        
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
                cv2.drawContours(image, contours, contnumber, (0,0,255), 2) #draw only contour contnumber
            contnumber+=1
                
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,100,apertureSize = 3)
        minLineLength = 50
        maxLineGap = 2
        lines = cv2.HoughLinesP(edges,1,np.pi/90,10,minLineLength,maxLineGap)
        return lines        

        
        
class rectifiedThread(QThread):
    def __init__(self):
        QThread.__init__(self)
    def __del__(self):
        self.wait()
    def run(self):
        self.rectifiedImage = rectifiedImage()

class rectifiedImage(QtGui.QMainWindow):
    def __init__(self):
        super(rectifiedImage, self).__init__()
        self.setGeometry(50,50,128*6,128*6)
        self.setWindowTitle("Rectified Image")
        scale = 5
        self.graphics = QtGui.QGraphicsView(self)
        self.graphics.setGeometry(0,0,128*6,128*6)
        self.scene = QtGui.QGraphicsScene()
        self.image = QtGui.QImage(128, 128, QtGui.QImage.Format_RGB32)
        self.image.fill(QtGui.qRgb(255,255,255))
        self.pixmap = QtGui.QGraphicsPixmapItem()
        self.image.fill(QtGui.qRgb(127,127,127))
        tempPixmap = QtGui.QPixmap(1,1)
        tempPixmap.convertFromImage(self.image)
        tempPixmap = tempPixmap.scaled(128*scale,128*scale)
        self.pixmap.setPixmap(tempPixmap)
        self.scene.addItem(self.pixmap)    
        self.graphics.setScene(self.scene)
        self.show()
    def reset(self, image):
        tempPixmap = QtGui.QPixmap(1,1)
        tempPixmap.convertFromImage(image)
        tempPixmap = tempPixmap.scaled(128*5,128*5)
        self.pixmap.setPixmap(tempPixmap)


class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50,50,128*6,128*6)
        self.setWindowTitle("eDVS Line detector")
        self.scale = 5
#        self.rectifiedImage = rectifiedImage()
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
        self.get_thread.start()
        self.connect(self.get_thread, SIGNAL("reset"), self.reset)
    def reset(self, image):
        tempPixmap = QtGui.QPixmap(1,1)
        tempPixmap.convertFromImage(image)
        tempPixmap = tempPixmap.scaled(128*self.scale,128*self.scale)
        self.pixmap.setPixmap(tempPixmap)
#        self.rectify(image)
    def rectify(self, image):
        x = 0;
        y = 100;
        newImage = QtGui.QImage(128, 128, QtGui.QImage.Format_RGB32)
        newImage.fill(QtGui.qRgb(255,255,255))

        for y in range(100,127):
            for x in range(1,127):
                c = image.pixel(x,y)
                red = QColor(c).getRgbF()[0]
                if red > 0:
                    newX, newY = self.reverseMap(x, y)
#                    print newX, newY
                    newImage.setPixel(int(newX),int(newY), QtGui.qRgb(0,0,0))
                x += 1
            y += 1
            x = 0
            self.rectifiedImage.reset(newImage)
    def reverseMap(self, x, y):
        H = 10
        f = 55        
        d = math.fabs((H*f)/-1)+1
        newX = H * x * (f/-y) + d
        newY = H * y * (f/-y) + d
        return round(newX), round(newY)

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
