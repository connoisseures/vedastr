import torch.nn as nn
import torch
from .bodies import build_body
from .heads import build_head
from .registry import MODELS
import torch.nn.functional as F


@MODELS.register_module
class GModel(nn.Module):
    def __init__(self, body, head, need_text=True):
        super(GModel, self).__init__()

        self.body = build_body(body)
        self.head = build_head(head)
        self.need_text = need_text
        self.export_wi_softmax = False

    def forward(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        x = self.body(inputs[0])
        if self.need_text:
            # add a dummy label for exporting onnx
            if len(inputs) < 2:
                dmy_label = torch.as_tensor([[95]])
                if torch.cuda.is_available():
                    dmy_label = dmy_label.coda()
                inputs = (inputs[0], dmy_label)

            # out shape: torch.Size([1, 26, 95])
            out = self.head(x, inputs[1])

            # add a softmax layer in the output
            # out shape: torch.Size([1, 26]), torch.Size([1, 26])
            if self.export_wi_softmax:
                probs = F.softmax(out, dim=2)
                max_probs, indexes = probs.max(dim=2)
                return indexes, max_probs
        else:
            out = self.head(x)

        return out
