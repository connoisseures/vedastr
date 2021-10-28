# export model to onnx using GPU with gpu_id 0
PTH="./pth/small_satrn.pth"
OUTFILE="satrn.onnx"
CUDA_VISIBLE_DEVICES="0" python tools/export_onnx.py configs/small_satrn.py $PTH $OUTFILE --dummy_input_shape "3,32,100" --dynamic_shape

