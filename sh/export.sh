# export model to onnx using GPU with gpu_id 0
CUDA_VISIBLE_DEVICES="0" python tools/export_torch2onnx.py configs/resnet_ctc.py "./pth/resnet_ctc.pth" "reset.onnx" --dummy_input_shape "3,32,100" --dynamic_shape

