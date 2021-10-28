import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
from tools.OnnxSatrnWoSoftmax import OnnxSatrnWoSoftmax


if __name__ == '__main__':
    onnx_path = '../onnx/satrn_wo_softmax.onnx'
    satrn = OnnxSatrnWoSoftmax(onnx_path)

    image_path = '../test_images/ssn00001_sub13.png'
    img = cv2.imread(image_path)

    satrn.infer(img)








