#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing necessary libraries, mainly the OpenCV, and PyQt libraries
import cv2
import numpy as np
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSignal

import time


class ShowVideo(QtCore.QObject):
    # 好像是所谓的信号槽？ VideoSignal -> QImage
    DepthSignal = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)

    @QtCore.pyqtSlot()
    def startVideo(self):
        run_video = True

        # depth = core.PyMat()
        # depth_ndarray = (np.arange(256 * 256 ) % 256).reshape(256, 256)
        # depth_ndarray = np.ones((256, 256), dtype='int8').reshape(256, 256) * 10

        # depth_ndarray = np.dstack((depth_ndarray, depth_ndarray, depth_ndarray))

        # print(depth_ndarray)
        # @@@@@@@@@上面新加的
        # try:
        while run_video:
            # height, width, _ = color_swapped_image.shape
            depth_ndarray = (np.arange(256 * 256, dtype='int8') % 256).reshape(256, 256)
            depth_ndarray = np.dstack((depth_ndarray, depth_ndarray, depth_ndarray))
            # depth_ndarray = cv2.imread('./moon.png')
            height, width, _ = depth_ndarray.shape
            # height, width = depth_ndarray.shape
            print(depth_ndarray.shape)
            qt_depth = QtGui.QImage(depth_ndarray,
                                    width,
                                    height,
                                    depth_ndarray.strides[0],
                                    QtGui.QImage.Format_RGB888)
            # qt_depth.setColorTable(list(i for i in range(256)))

            # for i in range(256):
            #     qt_depth.setColor(i, QtGui.QColor(i, i, i).rgb())

            self.DepthSignal.emit(qt_depth)


class DepthViewer(QtWidgets.QWidget):
    # 继承Qwidget这个画布的基类
    def __init__(self, parent=None):
        super(DepthViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        pass

    # 注意VideoSignal的pyqtSinal 和slot应该是成对的
    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("DepthViewer Dropped frame!")
        # image = image.scaled(image.size() / 2)  # 把图像缩小一点
        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


if __name__ == '__main__':
    # 生成一个app
    app = QtWidgets.QApplication(sys.argv)

    # 开启一个线程
    thread = QtCore.QThread()
    thread.start()
    # 生成一个ShowVideo实例
    vid = ShowVideo()
    vid.moveToThread(thread)
    # 生成一个ImageViewer实例 应该就是那个展示视频的区域控件吧
    depth_viewer = DepthViewer()
    # VideoSignal是ShowVideo实例的一个属性 connect是连接啥的？ pyqtSignal和pyqtSlot应该是成对的
    vid.DepthSignal.connect(depth_viewer.setImage)
    # Button to start the videocapture:
    push_button = QtWidgets.QPushButton('Start')
    push_button.clicked.connect(vid.startVideo)

    # 生成一个垂直布局来放Image
    video_vlayout = QtWidgets.QVBoxLayout()
    video_vlayout.addWidget(depth_viewer)
    video_vlayout.addWidget(push_button)

    full_hwidget = QtWidgets.QWidget()
    full_hwidget.setLayout(video_vlayout)
    # 主窗口
    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(full_hwidget)
    main_window.setWindowTitle('场景理解识别平台')
    main_window.show()

    sys.exit(app.exec_())
