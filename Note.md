Design Note
---

### environment 
- pytorch 1.10.0

### image preprocess before using SATRN model 

- ToGray -> Resize -> Normalize -> ToTensor
    - https://github.com/connoisseures/vedastr/blob/master/vedastr/transforms/builder.py#L10

### code review for SATRN

- character decode
    - https://github.com/connoisseures/vedastr/blob/master/vedastr/converter/attn_converter.py#L50

- token char for stop
    - https://github.com/connoisseures/vedastr/blob/master/vedastr/converter/attn_converter.py#L15

- add softmax layer in the output when exporting onnx model
    - https://github.com/connoisseures/vedastr/blob/master/vedastr/models/model.py#L34

### export onnx 
- input/output
    - input names: ['input.0']
    - output names: ['output.0']



