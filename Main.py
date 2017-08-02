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
            self.image2.fill(QtGui.qRgb(255,255,255))
            for i,k in zip(line[0::2], line[1::2]):
                byteOne = struct.unpack('B', i)[0]
                byteTwo = struct.unpack('B', k)[0]
        
                validEvent = (byteOne & 0x80) >> 7;
                if validEvent == 0:
                    continue
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
            self.painter.setPen(pen)
            self.painter2.setPen(pen)

            

            array2 = self.QImageToCvMat(self.image2)
            homographyMatrix = np.matrix('-8.42014397e-01  -9.13836156e-01   1.10386987e+02;1.22808744e-02  -3.46093934e+00   3.46141924e+02;-5.21625321e-05  -1.55665641e-02   1.00000000e+00')
            img = cv2.warpPerspective(array2, homographyMatrix, (128,128))
#            [x][y][2] is the red component
            qImage = self.toQImage(img)
            self.painter3 = QPainter()
            self.painter3.begin(qImage)
            self.painter3.setPen(QColor(qRgb(0,0,255)))


            totalX = 0
            totalY = 0
            countX = 0
            countY = 0
            for y in range(0,127):
                for x in range(0,127):
                    if img[y][x][2] == 254:
                        totalX += (x-63)
                        totalY += y
                        countX += 1
                        countY += 1
            
            if countX > 0:
                COGx = totalX / countX
                COGx += 63
                self.painter3.drawLine(COGx,63,63,63)
#
            self.painter.end()
            self.painter2.end()
            self.painter3.end()
            self.emit(SIGNAL('reset'), qImage)


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
        
        

class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50,50,128*6,128*6)
        self.setWindowTitle("eDVS Line detector")
        self.scale = 5
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

def main():
    app = QtGui.QApplication(sys.argv)
    form = Window()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()



