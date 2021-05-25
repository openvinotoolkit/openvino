import subprocess
import sys
import xml.etree.ElementTree as ET

from common.legacy.utils.multiprocessing_utils import multiprocessing_run


def shell(cmd, env=None, cwd=None):
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        cmd = ['/bin/bash', '-c', "".join(cmd)]
    else:
        cmd = "".join(cmd)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    return p.returncode, stdout, stderr


def union(ref_params, params):
    def equal_cases(case1, case2):
        for i in range(len(case1)):
            if type(case1[i]) != type(case2[i]) or case1[i] != case2[i]:
                return False
        return True

    if not ref_params:
        return params

    final_params = ref_params
    for case in params:
        case_equal = False
        for ref_case in ref_params:
            if equal_cases(ref_case, case):
                case_equal = True
                break
        if not case_equal:
            final_params.append(case)
    return final_params


def is_power(x_shape, y_shape):
    """
    Function to check what MO should generate
    :param x_shape: shape of the first input
    :param y_shape: shape of the second input
    :return: True if MO will generate Power else False
    """
    import numpy as np
    x_shape_np = np.array(x_shape)
    y_shape_np = np.array(y_shape)
    # Check that one of the shapes mean constant ([1] or [1, 1] or etc.)
    return (x_shape_np == 1).all() or (y_shape_np == 1).all()


def is_scaleshift(x_shape, y_shape):
    """
    Function to check what MO should generate
    :param x_shape: shape of the first input (Placeholder) in (N,H,W,C) layout
    :param y_shape: shape of the second input (Const) in (N,H,W,C) layout
    :return: True if MO will generate ScaleShift else False
    """
    import numpy as np
    x_shape_np = np.array(x_shape)
    y_shape_np = np.array(y_shape)

    # Check for case "x_shape=[3], y_shape=[3]" when shapes don't contain C
    if (len(x_shape_np) == 1 and len(y_shape_np) == 1) and \
            (x_shape_np[-1] != 1 and y_shape_np[-1] != 1) and \
            x_shape_np[-1] == y_shape_np[-1]:
        return False

    # Check that y_shape (Const) contain 1 in all dimensions except C dimension
    cond = (y_shape_np == 1)[:-1].all()
    # Check that C in x_shape (Placeholder) are equal with C in y_shape (Const) and C != 1
    cond = cond and (x_shape_np[-1] != 1 and y_shape_np[-1] != 1) and x_shape_np[-1] == y_shape_np[-1]

    return cond


class BaseInfer:
    def __init__(self, name):
        self.name = name
        self.res = None

    def fw_infer(self, input_data):
        raise RuntimeError("This is base class, please implement infer function for the specific framework")

    def infer(self, input_data):
        self.res = multiprocessing_run(self.fw_infer, [input_data], self.name, timeout=60)
        return self.res


class IEInfer(BaseInfer):
    def __init__(self, model, weights, device):
        super().__init__('Inference Engine')
        self.device = device
        self.model = model
        self.weights = weights

    def fw_infer(self, input_data):
        from openvino.inference_engine import IECore, get_version as ie_get_version

        print("Inference Engine version: {}".format(ie_get_version()))
        print("Creating IE Core Engine...")
        ie = IECore()
        print("Reading network files")
        net = ie.read_network(self.model, self.weights)
        print("Loading network")
        exec_net = ie.load_network(net, self.device)
        print("Starting inference")
        result = exec_net.infer(input_data)

        if "exec_net" in locals():
            del exec_net
        if "ie" in locals():
            del ie

        return result
