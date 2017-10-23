#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing necessary libraries, mainly the OpenCV, and PyQt libraries
import cv2
import numpy as np
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSignal

import time

from darkflow.net.build import TFNet

# 生成框的颜色
names = [
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]
RED = 255
GREEN = 255
BLUE = 255
colors = {}
i = 0
red = RED
for r in range(4):
    green = GREEN
    for g in range(4):
        blue = BLUE
        for b in range(5):
            colors[names[i]] = (red, green, blue)
            blue = blue / 2
            i += 1
        green = green / 2
    red = red / 2


class ShowVideo(QtCore.QObject):
    # initiating the built in camera
    # 初始化相机
    camera_port = 0
    camera = cv2.VideoCapture(camera_port)
    # 初始化网络
    options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.4}
    tfnet = TFNet(options)
    # 好像是所谓的信号槽？ VideoSignal -> QImage
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    InfoSignal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)

    def _drawBox(self, image, info, height, width):
        for item in info:
            mess = item['label']
            top = item['topleft']['y']
            left = item['topleft']['x']
            bottom = item['bottomright']['y']
            right = item['bottomright']['x']
            thick = int((height + width) // 300)
            color = colors[mess]

            topleft = (left, top)
            bottomright = (right, bottom)

            cv2.rectangle(image, topleft, bottomright, color, 5)
            cv2.putText(image, mess, (left, top - 12),
                        0, 1e-3 * height, color, thick // 3)

    def _formatJSON(self, json_list, fps):
        info_str = ''
        for json in json_list:
            label = json['label']
            confidence = json['confidence'] * 100

            info_str += 'Label %s, confidence: %.2f%%, depth: 1.2m\n' % (label, confidence)
        info_str = 'fps: %.2f\n' % fps + info_str
        return info_str

    @QtCore.pyqtSlot()
    def startVideo(self):

        run_video = True

        elaped = 0
        fps = 0
        start_time = time.time()
        while run_video:
            # 用opencv获得一帧
            ret, image = self.camera.read()
            # BGR => RGB
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width, _ = color_swapped_image.shape
            # 这里用了调换位置的image 但是原先写的代码没有调换 看看效果先
            info_json = self.tfnet.return_predict(color_swapped_image)
            # 在图片上画框修改像素值
            self._drawBox(color_swapped_image, info_json, height, width)
            # 把opencv获取的np.ndarray => QImage 这里把图片缩小了 方便看 默认的太大了
            qt_image = QtGui.QImage(color_swapped_image.data,
                                    width,
                                    height,
                                    color_swapped_image.strides[0],
                                    QtGui.QImage.Format_RGB888)
            # 将QImage发射到VideoSignal？还是说交给VideoSignal来emit？
            # 可以理解为 视频一帧帧循环并触发信号 把qt_image事件对象传出
            # 而槽则为后面connect的setImage
            # 换句话说 QImage实例作为事件对象 VideoSignal发出信号交给setImage来处理
            # 而我如果没估计错的话 update会调用paintEvent从而重新drawImage 完成图像更新
            if elaped < 5:
                elaped += 1
            else:
                end_time = time.time()
                interval_time = end_time - start_time
                fps = elaped / interval_time
                elaped = 0
                start_time = time.time()
            self.VideoSignal.emit(qt_image)  # 发图
            self.InfoSignal.emit(self._formatJSON(info_json, fps))  # 这里解析json并发送吧


class ImageViewer(QtWidgets.QWidget):

    # 继承Qwidget这个画布的基类
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        # 感觉就像个事件循环 重新又画一个一样
        # 果然 QImage绘图的方法在C++中就是重写虚函数paintEvent
        # 然后创建QPainter并调用drawImage来绘制图片
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        # 初始化UI 这里设置了窗口名字
        self.setWindowTitle('Test')

    # 注意VideoSignal的pyqtSinal 和slot应该是成对的
    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")
        image = image.scaled(image.size() / 2)  # 把图像缩小一点
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
    image_viewer = ImageViewer()
    # VideoSignal是ShowVideo实例的一个属性 connect是连接啥的？ pyqtSignal和pyqtSlot应该是成对的
    vid.VideoSignal.connect(image_viewer.setImage)

    # Button to start the videocapture:
    push_button = QtWidgets.QPushButton('Start')
    push_button.clicked.connect(vid.startVideo)

    # 生成一个垂直布局来放Image
    video_vlayout = QtWidgets.QVBoxLayout()
    video_vlayout.addWidget(image_viewer)
    video_vlayout.addWidget(push_button)
    video_vwidget = QtWidgets.QWidget()
    video_vwidget.setLayout(video_vlayout)
    # 生成一个垂直布局来放Info
    info_label = QtWidgets.QLabel('检测数据', parent=None)
    info_datashow = QtWidgets.QTextEdit('当前没有数据', parent=None)
    vid.InfoSignal.connect(info_datashow.setText)  # 连接信号槽

    info_datashow.setReadOnly(True)  # 设置为只读
    info_vlayout = QtWidgets.QVBoxLayout()
    info_vlayout.addWidget(info_label)
    info_vlayout.addWidget(info_datashow)
    info_vwidget = QtWidgets.QWidget()
    info_vwidget.setLayout(info_vlayout)
    info_vwidget.setFixedWidth(400)
    # 合成水平布局
    horizontal_layout = QtWidgets.QHBoxLayout()
    horizontal_layout.addWidget(video_vwidget)
    horizontal_layout.addWidget(info_vwidget)
    full_hwidget = QtWidgets.QWidget()
    full_hwidget.setLayout(horizontal_layout)
    # 主窗口
    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(full_hwidget)
    main_window.setWindowTitle('场景理解识别平台')
    main_window.show()

    sys.exit(app.exec_())
