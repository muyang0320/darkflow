# -*- coding=utf-8 -*-
# 用来测试opencv怎么读取摄像头展示画面
import cv2
import numpy as np
from darkflow.net.build import TFNet

cap = cv2.VideoCapture(0)
# options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.4}
# tfnet = TFNet(options)

while (1):
    # get a frame
    ret, frame = cap.read()  # ret是个bool 应该是是否成功获取到了吧
    # print(frame.__class__.__name__) ndarray 也就是说opencv提供的帧数据是numpy的格式

    # 视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width, _ = frame.shape
    print('fps: %d' % fps)
    print('相机画面尺寸：(高%d, 宽%d)' % (height, width))

    # 制造一份拷贝 画上rectangle
    frame_tmp = frame.copy()
    # result = tfnet.return_predict(frame_tmp)  # 返回的是json
    # print(result)
    #
    # colors = {
    #     'person': (255, 255, 255),
    #     'cell phone': (0, 255, 0)
    # }
    # for item in result:
    #     mess = item['label']
    #     top = item['topleft']['y']
    #     left = item['topleft']['x']
    #     bottom = item['bottomright']['y']
    #     right = item['bottomright']['x']
    #     thick = int((height + width) // 300)
    #     color = colors[mess]
    #
    #     topleft = (left, top)
    #     bottomright = (right, bottom)
    #
    #     cv2.rectangle(frame_tmp, topleft, bottomright, color, 5)
    #
    #     cv2.putText(frame_tmp, mess, (left, top - 12),
    #                 0, 1e-3 * height, color, thick // 3)
    # show a frame
    cv2.imshow("capture", frame_tmp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
