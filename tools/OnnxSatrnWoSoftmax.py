import cv2, torch, onnxruntime, re
import numpy as np
import torch.nn.functional as F

class OnnxSatrnWoSoftmax:
    def __init__(self, model_path):
        self.log_header = '[{}]'.format(type(self).__name__)
        self.logs = ''
        self.is_verbose = False
        self.ort_session = onnxruntime.InferenceSession(model_path)

        # in/out of model
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        # mean, std for normalization
        self.mean = 0.5
        self.std = 0.5

        # resized image
        self.height = 32
        self.width = 100

        # decode
        self.sensitive = True
        self.character = '0123456789abcdefghijklmnopq' \
                    'rstuvwxyzABCDEFGHIJKLMNOPQRS' \
                    'TUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'  # need character
        list_token = ['[s]', '[GO]']
        self.character_list = list(self.character) + list_token

    def to_log(self, log):
        log = self.log_header + log
        if self.is_verbose:
            print(log)
        self.logs += log

    def infer(self, frame):
        img = self.preprocess(frame)
        # compute ONNX Runtime output prediction
        img = img.tolist()
        ort_inputs = {self.input_name: [img]}
        ort_outs = self.ort_session.run(None, ort_inputs)

        # to be removed
        preds = torch.as_tensor(ort_outs[0])

        pred, prob = self.postprocess(preds)
        print(pred)
        print(prob)
        return pred, prob

    def postprocess(self, preds):
        # to be removed, put the softmax into the onnx model
        # shape: torch.Size([1, 26, 95])
        probs = F.softmax(preds, dim=2)

        # shape: torch.Size([1, 26]), torch.Size([1, 26, 95])
        max_probs, indexes = probs.max(dim=2)

        preds_str = []
        preds_prob = []
        for i, pstr in enumerate(self.decode(indexes)):
            str_len = len(pstr)
            if str_len == 0:
                prob = 0
            else:
                # prob = product of each char's probability
                prob = max_probs[i, :str_len].cumprod(dim=0)[-1]
                # https://numpy.org/doc/stable/reference/generated/numpy.cumprod.html
            preds_prob.append(prob)
            if not self.sensitive:
                pstr = pstr.lower()

            if self.character:
                pstr = re.sub('[^{}]'.format(self.character), '', pstr)

            preds_str.append(pstr)

        return preds_str, preds_prob

    def decode(self, text_index):
        texts = []
        # batch_size = text_index.shape[0]
        batch_size = 1

        for index in range(batch_size):
            text = ''.join([self.character_list[i] for i in text_index[index, :]])
            text = text[:text.find('[s]')]
            texts.append(text)

        return texts


    def preprocess(self, frame):
        gray = self.to_gray(frame)
        resized = self.resize(gray, self.height, self.width)
        normalzied = self.normalize(resized, self.mean, self.std)
        #print(normalzied.shape)
        tensor_img = self.to_tensor(normalzied)
        # tensor_img = self.to_nchw(normalzied)
        return tensor_img

    def to_gray(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray

    def resize(self, img, height, width):
        return cv2.resize(img, (width, height))

    def normalize(self, img, mean, std, max_pixel_value=255.0):
        mean = np.array(mean, dtype=np.float32)
        mean *= max_pixel_value
        std = np.array(std, dtype=np.float32)
        std *= max_pixel_value
        denominator = np.reciprocal(std, dtype=np.float32)
        img = img.astype(np.float32)
        img -= mean
        img *= denominator
        return img

    def to_nchw(self, image):
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[None, :, :, None]
            image = image.transpose(0, 3, 1, 2)
        else:
            raise TypeError('image shoud be np.ndarray. Got {}'.format(type(image)))
        return image

    def to_tensor(self, image):
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[:, :, None]
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)
        else:
            raise TypeError('image shoud be np.ndarray. Got {}'.format(type(image)))
        return image


if __name__ == '__main__':
    onnx_path = '../onnx/satrn.onnx'
    satrn = OnnxSatrnWoSoftmax(onnx_path)

    image_path = '../test_images/ssn00001_sub13.png'
    img = cv2.imread(image_path)

    satrn.infer(img)








