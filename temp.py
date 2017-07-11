import numpy as np
import cv2
import serial
import struct
from PyQt4.QtCore import QThread, SIGNAL
import sys
from PyQt4 import QtGui
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class computationThread(QThread):
    def __init__(self):
        QThread.__init__(self)
    def __del__(self):
        self.wait()
    def run(self):
        ser = serial.Serial('COM3', 12000000, timeout=1)
        ser.write('E+\nE+')
        self.image = QtGui.QImage(128, 128, QtGui.QImage.Format_RGB32)

        while ser.is_open:
            line = ser.read(512)
            self.image.fill(QtGui.qRgb(255,255,255))
            for i,k in zip(line[0::2], line[1::2]):
                byteOne = struct.unpack('B', i)[0]
                byteTwo = struct.unpack('B', k)[0]
        
                validEvent = (byteOne & 0x80) >> 7;
                if validEvent == 0:
                    continue;
                    
                y = byteOne & 0x7f;
                x = byteTwo & 0x7f;
                polarity = (byteTwo & 0x80) >> 7;
                if polarity == 1:    
                    self.image.setPixel(x,y, QtGui.qRgb(255,0,0))   
                else:
                    self.image.setPixel(x,y, QtGui.qRgb(0,255,0))  
            self.emit(SIGNAL('reset'), self.image)

class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50,50,128*6,128*6)
        self.setWindowTitle("eDVS Line detector")
        scale = 5
        self.graphics = QtGui.QGraphicsView(self)
        self.graphics.setGeometry(0,0,128*6,128*6)
        self.scene = QtGui.QGraphicsScene()
        self.image = QtGui.QImage(128, 128, QtGui.QImage.Format_RGB32)
        self.pixmap = QtGui.QGraphicsPixmapItem()
        self.image.fill(QtGui.qRgb(127,127,127))
        tempPixmap = QtGui.QPixmap(1,1)
        tempPixmap.convertFromImage(self.image)
        tempPixmap = tempPixmap.scaled(128*scale,128*scale)
        self.pixmap.setPixmap(tempPixmap)
        self.scene.addItem(self.pixmap)    
        self.graphics.setScene(self.scene)
        self.show()
        self.get_thread = computationThread()
        self.get_thread.start()
        self.connect(self.get_thread, SIGNAL("reset"), self.reset)
    def setPixel(self, x, y, polarity):
        if polarity == 1:    
            self.image.setPixel(x,y, QtGui.qRgb(255,0,0))   
        else:
            self.image.setPixel(x,y, QtGui.qRgb(0,255,0))               
    def reset(self, image):
        tempPixmap = QtGui.QPixmap(1,1)
        tempPixmap.convertFromImage(image)
        tempPixmap = tempPixmap.scaled(128*5,128*5)
        self.pixmap.setPixmap(tempPixmap)
        
def main():
    app = QtGui.QApplication(sys.argv)
    form = Window()
    form.show()
    app.exec_()

#if __name__ == '__main__':
    #main()



### WIP

def QImageToCvMat(incomingImage):
    '''  Converts a QImage into an opencv MAT format  '''

    incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format_RGB32)

    width = incomingImage.width()
    height = incomingImage.height()

    Matrix = [[0 for x in range(width)] for y in range(height)] 
    for index in Matrix:
        for i, val in value:
            Matrix[index][i] =  incomingImage.pixel(index, i).red()
    
    return Matrix


imgpath = 'C:/Users/sayegh/test.jpg'
img = cv2.imread(imgpath)
image = QtGui.QImage(128, 128, QtGui.QImage.Format_RGB32)
image.setPixel(50,0, QtGui.qRgb(0,0,0))   
image.setPixel(50,1, QtGui.qRgb(0,0,0))   
image.setPixel(50,2, QtGui.qRgb(0,0,0))   
image.setPixel(50,3, QtGui.qRgb(0,0,0))   
image.setPixel(50,4, QtGui.qRgb(0,0,0))   
image.setPixel(50,5, QtGui.qRgb(0,0,0))   
image.setPixel(50,6, QtGui.qRgb(0,0,0))   
image.setPixel(50,7, QtGui.qRgb(0,0,0))   
image.setPixel(50,8, QtGui.qRgb(0,0,0))   
image.setPixel(50,9, QtGui.qRgb(0,0,0))   
image.setPixel(50,10, QtGui.qRgb(0,0,0))   
image.setPixel(50,11, QtGui.qRgb(0,0,0))

doneImg = QImageToCvMat(image)
print doneImg
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 10
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('hough.jpg',img)
