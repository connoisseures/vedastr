import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
from tools.OnnxSatrn import OnnxSatrn


if __name__ == '__main__':
    onnx_path = '../onnx/satrn.onnx'
    satrn = OnnxSatrn(onnx_path)
    satrn.is_verbose = True
    image_path = '../test_images/ssn00001_sub4.png'
    img = cv2.imread(image_path)
    pred, prob, each_max_prob = satrn.infer(img)
    print(pred)
    print(prob)
    print(each_max_prob)
    pred, hit_score = satrn.recognize_wi_hit_score(img)
    print(pred)
    print(hit_score)
    pred, hit_score = satrn.recognize_wi_hit_score(img, is_1st_char_skipped=True)
    print(pred)
    print(hit_score)









