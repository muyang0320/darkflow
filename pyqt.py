#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing necessary libraries, mainly the OpenCV, and PyQt libraries
import cv2
import numpy as np
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSignal

import os
import time
import platform

from darkflow.net.build import TFNet

# 引入zed相关包
import pyzed.camera as zcam
import pyzed.defines as sl
import pyzed.types as tp
import pyzed.core as core
import math

# super params 超参数
THRESHOLD = 0.3
MODEL_PATH = "cfg/yolo.cfg"
WEIGHTS_PATH = "bin/yolo.weights"
GPU = 0.8


# 生成框的颜色
def generate_colors():
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
    return colors


class ShowVideo(QtCore.QObject):
    # @@@@@@@@@@@@@下面是新加入
    # Create a PyZEDCamera object
    zed = zcam.PyZEDCamera()
    print('----------------创建ZED相机实例----------------')
    # Create a PyInitParameters object and set configuration parameters
    init_params = zcam.PyInitParameters()
    init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_PERFORMANCE  # 见pyzed/defines.pyx
    init_params.coordinate_units = sl.PyUNIT.PyUNIT_MILLIMETER  # 深度为毫米单位
    init_params.camera_fps = 30
    print('----------------相机参数初始化----------------')
    # Open the camera
    err = zed.open(init_params)
    if err != tp.PyERROR_CODE.PySUCCESS:
        exit(1)
    print('----------------打开ZED相机----------------')
    # Create and set PyRuntimeParameters after opening the camera
    runtime_parameters = zcam.PyRuntimeParameters()
    # runtime_parameters.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_STANDARD  # Use STANDARD sensing mode
    print('----------------设置运行时ZED相机参数----------------')
    # @@@@@@@@@@@@@上面是新加入

    # 初始化神经网络网络
    options = {"model": MODEL_PATH, "load": WEIGHTS_PATH, "threshold": THRESHOLD, "gpu": GPU}
    tfnet = TFNet(options)
    # 好像是所谓的信号槽？ VideoSignal -> QImage
    VideoSignal = QtCore.pyqtSignal(QtGui.QImage)
    DepthSignal = QtCore.pyqtSignal(QtGui.QImage)
    InfoSignal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)
        self.colors = generate_colors()

    def _drawBox(self, image, info, height, width):
        image = image.copy()  # 据说是python-opencv的bug 需要来个副本能好 目前看好像还行？
        for item in info:
            mess = item['label']
            top = item['topleft']['y']
            left = item['topleft']['x']
            bottom = item['bottomright']['y']
            right = item['bottomright']['x']
            thick = int((height + width) // 300)
            color = self.colors[mess]

            topleft = (left, top)
            bottomright = (right, bottom)
            cv2.rectangle(image, topleft, bottomright, color, 5)
            cv2.putText(image, mess, (left, top - 12),
                        0, 1e-3 * height, color, thick // 3)
        return image

    def _calcDepth(self, depth, info):
        # todo 把depth矩阵过滤一下 因为存在NaN
        depth[np.isnan(depth)] = np.Inf
        for item in info:
            top = item['topleft']['y']
            left = item['topleft']['x']
            bottom = item['bottomright']['y']
            right = item['bottomright']['x']
            sub_pic = depth[top:bottom + 1, left:right + 1]  # 注意切片要加1
            item['depth'] = np.min(sub_pic)
        depth = self._depthToGray(depth)
        return depth

    # @@@ api自带了相关方法的zed.retrieveImage(image_depth_zed, VIEW_DEPTH);
    # VIEW_DEPTH在sl.PyVIEW.VIEW_DEPTH里
    def _depthToGray(self, depth):
        depth = np.floor((depth / 20000.0) * 255).astype(dtype='uint8')  # int8会溢出出现负数 这也是之前出问题的原因
        depth = 255 - depth
        depth = np.dstack((depth, depth, depth))
        return depth

    def _formatJSON(self, json_list, fps):
        info_str = ''
        for json in json_list:
            label = json['label']
            depth = json['depth'] / 100
            confidence = json['confidence'] * 100

            info_str += 'Label %s, confidence: %.2f%%, depth: %.2fm\n' % (label, confidence, depth)
        info_str = 'fps: %.2f\n' % fps + info_str
        return info_str

    @QtCore.pyqtSlot()
    def startVideo(self):
        run_video = True

        elaped = 0  # 已经播放的帧数
        fps = 0  # 帧率
        start_time = time.time()  # 开始时间 用来算fps

        # @@@@@@@@@下面新加的
        image = core.PyMat()
        depth = core.PyMat() # 深度矩阵 用以计算距离
        gray_depth = core.PyMat() # 灰度图
        color_depth = core.PyMat() # 伪彩色图
        # @@@@@@@@@上面新加的
        # try:
        while run_video:
            # 用opencv获得一帧
            # 这里用zed获取image.get_data()就是np数据了
            # ret, image = self.camera.read()
            # BGR => RGB
            # color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.zed.grab(self.runtime_parameters) == tp.PyERROR_CODE.PySUCCESS:
                # Retrieve left image

                self.zed.retrieve_image(image, sl.PyVIEW.PyVIEW_LEFT) # 拿到左边的图
                # Retrieve depth map. Depth is aligned on the left image
                # self.zed.retrieve_measure(depth, sl.PyMEASURE.PyMEASURE_DEPTH)
                # 这样可以得到normalized depth image
                # 见https://www.stereolabs.com/documentation/integrations/opencv/getting-started.html的displaying depth
                # @@@@@@@@@@@@@@@@@注意 如果这里得到的是rgb 就可以直接跑 如果不是 就需要dstack一下
                # 因为这里的depth应该是MAT_TYPE_8U_C4转过来的 所以可能需要和image一样slice一下
                # 有可能需要ndarray.copy()一下不然会有bug
                self.zed.retrieve_image(gray_depth, sl.PyVIEW.PyVIEW_DEPTH) # 用api拿到灰度图
                # color_depth = cv2.applyColorMap(gray_depth, cv2.COLORMAP_JET) # 转换伪彩色图
                self.zed.retrieve_measure(depth, sl.PyMEASURE.PyMEASURE_DEPTH) # 拿到深度数据
                image_ndarray = image.get_data()[:, :, [2, 1, 0]]  # 拿到图片的ndarray数组并bgr->rgb
                depth_ndarray = depth.get_data()
                # height, width, _ = color_swapped_image.shape
                height, width, _ = image_ndarray.shape
                # 这里用了调换位置的image 但是原先写的代码没有调换 看看效果先
                # info_json = self.tfnet.return_predict(color_swapped_image)
                start_t = time.time()
                info_json = self.tfnet.return_predict(image_ndarray)  # 这一步操作巨慢啊怎么回事
                print('本次算框获取花费时间：', time.time() - start_t)
                # 在图片上画框修改像素值
                image_ndarray = self._drawBox(image_ndarray, info_json, height, width)
                depth_ndarray = self._calcDepth(depth_ndarray, info_json)
                # 把opencv获取的np.ndarray => QImage 这里把图片缩小了 方便看 默认的太大了
                image_ndarray = image_ndarray.copy()  # 可能copy又能解bug
                qt_image = QtGui.QImage(image_ndarray,
                                        width,
                                        height,
                                        image_ndarray.strides[0],
                                        QtGui.QImage.Format_RGB888)
                print('0-----------------')
                print(gray_depth.shape)
                qt_depth = QtGui.QImage(gray_depth,
                                        width,
                                        height,
                                        depth_ndarray.strides[0],
                                        QtGui.QImage.Format_RGB888)
                # 将QImage发射到VideoSignal？还是说交给VideoSignal来emit？
                # 可以理解为 视频一帧帧循环并触发信号 把qt_image事件对象传出
                # 而槽则为后面connect的setImage
                # 换句话说 QImage实例作为事件对象 VideoSignal发出信号交给setImage来处理
                # 而我如果没估计错的话 update会调用paintEvent从而重新drawImage 完成图像更新
                if elaped < 10:
                    elaped += 1
                else:
                    end_time = time.time()
                    interval_time = end_time - start_time
                    fps = elaped / interval_time
                    elaped = 0
                    start_time = time.time()

                self.VideoSignal.emit(qt_image)  # 发图
                self.DepthSignal.emit(qt_depth)
                self.InfoSignal.emit(self._formatJSON(info_json, fps))  # 这里解析json并发送吧
                # except Exception:
                #     print('----------------视频循环异常----------------')
                # finally:
                #     print('----------------视频循环中断----------------')
                #     return 0


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
        pass

    # 注意VideoSignal的pyqtSinal 和slot应该是成对的
    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("ImageViewer Dropped frame!")
        image = image.scaled(image.size() / 2)  # 把图像缩小一点
        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


class DepthViewer(QtWidgets.QWidget):
    # 继承Qwidget这个画布的基类
    def __init__(self, parent=None):
        super(DepthViewer, self).__init__(parent)
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
        pass

    # 注意VideoSignal的pyqtSinal 和slot应该是成对的
    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("DepthViewer Dropped frame!")
        image = image.scaled(image.size() / 2)  # 把图像缩小一点
        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


if __name__ == '__main__':
    # 如果是linux就执行下面的shell语句 因为在ubuntu那台机子上opencv环境有点问题
    # if platform.system() == 'Linux':
    #     os.system('export LD_LIBRARY_PATH=$HOME/anaconda2/lib:$LD_LIBRARY_PATH')
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
    depth_viewer = DepthViewer()
    # VideoSignal是ShowVideo实例的一个属性 connect是连接啥的？ pyqtSignal和pyqtSlot应该是成对的
    vid.VideoSignal.connect(image_viewer.setImage)
    vid.DepthSignal.connect(depth_viewer.setImage)
    # Button to start the videocapture:
    push_button = QtWidgets.QPushButton('Start')
    push_button.clicked.connect(vid.startVideo)

    # 生成一个垂直布局来放Image
    video_vlayout = QtWidgets.QVBoxLayout()
    video_vlayout.addWidget(image_viewer)
    video_vlayout.addWidget(depth_viewer)
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
