from .logger import *


if 'MO_ROOT' in os.environ:
    mo_bin = os.environ['MO_ROOT']
    if not os.path.exists(mo_bin):
        raise EnvironmentError(
            "Environment variable MO_ROOT points to non existing path {}".format(mo_bin))
else:
    raise EnvironmentError("MO_ROOT variable is not set")

if os.environ.get('OUTPUT_DIR') is not None:
    out_path = os.environ['OUTPUT_DIR']
else:
    script_path = os.path.dirname(os.path.realpath(__file__))
    out_path = os.path.join(script_path, 'out')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

if 'DATA_PATH' in os.environ:
    img_path = os.path.join(os.environ['DATA_PATH'], '224x224', 'cat3.bmp')
else:
    tests_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    img_path = os.path.join(tests_root_dir, 'layer_tests', 'common', 'test_data', 'cat3.bmp')


caffe_models_path = os.path.join(out_path, 'caffe_models')
mxnet_models_path = os.path.join(out_path, 'mxnet_models')
ir_path = os.path.join(out_path, 'ir')

# supported_devices : CPU, GPU, MYRIAD, FPGA
test_device = os.environ.get('TEST_DEVICE', 'CPU;GPU').split(';')
test_precision = os.environ.get('TEST_PRECISION', 'FP32;FP16').split(';')

caffe_eps = 1e-5 if test_precision == 'FP16' else 5e-2
tf_eps = 1e-5 if test_precision == 'FP16' else 5e-2
mxnet_eps = 1e-5 if test_precision == 'FP16' else 5e-2

if 'TEST_EPS' in os.environ:
    caffe_eps, tf_eps, mxnet_eps = [float(os.environ.get('TEST_EPS'))] * 3

# Used in "add_code_coverage" in call_ModelOptimizer.py for code coverage
mo_coverage = False if os.environ.get("MO_COVERAGE") is None else True

"""
List of ignored attributes
"""
ignore_attributes = {'Split': ['num_split'],
                     'Flatten': ['end_axis'],
                     'RegionYolo': ['axis', 'end_axis'],
                     'Pooling': ['global_pool', 'convention','exclude_pad'],
                     'ReLU': ['engine']}
