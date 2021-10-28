import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
from tools.OnnxSatrn import OnnxSatrn


if __name__ == '__main__':
    onnx_path = '../onnx/satrn.onnx'
    satrn = OnnxSatrn(onnx_path)
    satrn.is_verbose = True
    image_path = '../test_images/ssn00001_sub13.png'
    img = cv2.imread(image_path)
    pred, prob = satrn.infer(img)
    print(pred)
    print(prob)








