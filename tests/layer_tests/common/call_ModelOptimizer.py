import logging as lg

from .constants import *
from .layer_utils import shell


def add_code_coverage(mo_root, coverage: bool):
    if coverage:
        cmd = " ".join(['-m', 'coverage', 'run', '-p', '--source={}'.format(mo_root),
                        '--omit=*_test.py', mo_root])
    else:
        cmd = mo_root
    return cmd


def generate_ir_from_caffe(path=caffe_models_path, name=None, precision='FP32', disable_fusing=False, extensions=None):
    caffe_file = '{}.caffemodel'.format(os.path.join(path, name))
    command = ' '.join([
        os.path.join('python3 {}'.format(add_code_coverage(mo_bin, mo_coverage)), 'mo_caffe.py'),
        '--input_model {}'.format(caffe_file),
        '--disable_fusing' if disable_fusing else '',
        '--data_type {}'.format(precision),
        '--output_dir {}'.format(ir_path),
        '--model_name {}'.format(name)
    ])
    if extensions:
        command = ' '.join([
            command,
            '--extensions {}'.format(extensions)
        ])
    lg.info(command)
    ret_code, out, err = shell(command)
    lg.info(out.decode().strip('\n'))
    if ret_code:
        raise RuntimeError('\n'.join(['ModelOptimizer error: ',
                                      err.decode().strip('\n'),
                                      out.decode().strip('\n')
                                      ]))


def generate_ir_from_mxnet(path=mxnet_models_path, name=None, input_shape=None, precision='FP32', disable_fusing=False,
                           input_names=['data'], extensions=None):
    input_model = '{}-0000.params'.format(os.path.join(path, name))
    command = ' '.join([
        os.path.join('python3 {}'.format(add_code_coverage(mo_bin, mo_coverage)), 'mo_mxnet.py'),
        '--input_model {}'.format(input_model),
        '--input_shape "{}"'.format(input_shape),
        '--data_type {}'.format(precision),
        '--output_dir {}'.format(ir_path),
        '--model_name {}'.format(name),
        '--disable_fusing' if disable_fusing else '',
        '--input {}'.format(",".join(input_names))
    ])
    if extensions:
        command = ' '.join([
            command,
            '--extensions {}'.format(extensions)
        ])
    lg.info(command)
    retcode, out, err = shell(command)
    lg.info(out.decode().strip('\n'))
    if retcode:
        raise RuntimeError('\n'.join(['ModelOptimizer error: ',
                                      err.decode().strip('\n'),
                                      out.decode().strip('\n')
                                      ]))
