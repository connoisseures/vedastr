import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import torch
import numpy as np

from vedastr.runners import InferenceRunner
from vedastr.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('out', type=str, help='output model file name')
    parser.add_argument('--dummy_input_shape', type=str, default='3,32,100',
                        help='input shape (e.g. 3,32,100) in C,H,W format')
    parser.add_argument('--dynamic_shape', action='store_true',
                        help='whether to use dynamic shape')
    parser.add_argument('--opset_version', default=14, type=int,
                        help='onnx opset version')
    parser.add_argument('--do_constant_folding', action='store_true',
                        help='whether to apply constant-folding optimization')
    parser.add_argument('--verbose', action='store_true',
                        help='whether print convert info')
    parser.add_argument('--export_wo_softmax', action='store_true',
                        help='export onnx without softmax in the output layer')
    args = parser.parse_args()

    return args

def get_names(inp, prefix):
    if not isinstance(inp, (tuple, list)):
        inp = [inp]

    names = []
    for i, sub_inp in enumerate(inp):
        sub_prefix = '{}.{}'.format(prefix, i)
        if isinstance(sub_inp, (list, tuple)):
            names.extend(get_names(sub_inp, sub_prefix))
        else:
            names.append(sub_prefix)

    return names

def flatten(inp):
    if not isinstance(inp, (tuple, list)):
        return [inp]

    out = []
    for sub_inp in inp:
        out.extend(flatten(sub_inp))

    return out

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    print(list(tuple_of_tensors))
    return torch.stack(list(tuple_of_tensors), dim=0)

def to_log(log):
    if isinstance(log, str):
        print('[export] ' + log)
    else:
        print(log)

def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    deploy_cfg = cfg['deploy']
    common_cfg = cfg.get('common')
    device = torch.device("cpu")
    deploy_cfg['gpu_id'] = "0"
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        deploy_cfg['gpu_id'] = str(device)


    runner = InferenceRunner(deploy_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)

    C, H, W = [int(_.strip()) for _ in args.dummy_input_shape.split(',')]
    dummy_image = np.random.random_integers(0, 255, (H, W, C)).astype(np.uint8)

    aug = runner.transform(image=dummy_image, label='')

    # label is empty, label = ""
    image, label = aug['image'], aug['label']

    if torch.cuda.is_available():
        image = image.unsqueeze(0).cuda()
    else:
        image = image.unsqueeze(0)

    # dummy_input = (image, runner.converter.test_encode([''])[2])
    dummy_input = image
    to_log('dummy_input')
    to_log(dummy_input)

    if torch.cuda.is_available():
        model = runner.model.cuda().eval()
    else:
        model = runner.model.eval()

    # add a softmax layer in the output
    # out shape: torch.Size([1, 26]), torch.Size([1, 26])
    if not args.export_wo_softmax:
        model.export_wi_softmax = True
        to_log('export with softmax')
        to_log('args.export_wo_softmax = ')
        to_log(args.export_wo_softmax)

    #need_text = runner.need_text
    #if not need_text:
    #    dummy_input = dummy_input[0]

    if args.dynamic_shape:
        print(
            f'Convert to Onnx with dynamic input shape and opset version'
            f'{args.opset_version}'
        )
    else:
        print(
            f'Convert to Onnx with constant input shape'
            f' {args.dummy_input_shape} and opset version '
            f'{args.opset_version}'
        )

    with torch.no_grad():
        output = model(dummy_input)

    assert not isinstance(dummy_input, dict), 'input should not be dict.'
    assert not isinstance(output, dict), 'output should not be dict'

    input_names = get_names(dummy_input, 'input')
    output_names = get_names(output, 'output')

    dynamic_shape = args.dynamic_shape
    dynamic_axes = dict()
    for name, tensor in zip(input_names + output_names,
                            flatten(dummy_input) + flatten(output)):
        dynamic_axes[name] = list(range(tensor.dim())) if dynamic_shape else [0]

    onnx_model_name = args.out
    verbose = args.verbose
    opset_version = args.opset_version
    do_constant_folding = False

    to_log('--- export setting ---')
    to_log('output model name')
    to_log(onnx_model_name)
    to_log('input names')
    to_log(input_names)
    to_log('output names')
    to_log(output_names)
    to_log('pytorch version')
    to_log(torch.__version__)
    to_log('opset version')
    to_log(opset_version)
    to_log('do_constant_folding')
    to_log(do_constant_folding)
    to_log('verbose')
    to_log(verbose)
    to_log('dynamic_axes')
    to_log(dynamic_axes)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_name,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        verbose=verbose,
        dynamic_axes=dynamic_axes)

    runner.logger.info(
        f'Convert successfully, saved onnx file: {os.path.abspath(args.out)}'
    )


if __name__ == '__main__':
    main()
