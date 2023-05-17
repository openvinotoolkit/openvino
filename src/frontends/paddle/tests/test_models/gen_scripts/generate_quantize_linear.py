import paddle
import paddle.nn.functional as F
import paddle.nn as nn

from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
import numpy as np

from save_model import exportModel
import sys

def randtool(dtype, low, high, shape):
        """
        np random tools
        """
        if dtype == "int":
                return np.random.randint(low, high, shape)

        elif dtype == "float":
                return low + (high - low) * np.random.random(shape)

        elif dtype == "bool":
                return np.random.randint(low, high, shape).astype("bool")

'''
    Reference:
    https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/tests/quantize_ops.py
    https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/tests/test_auto_scan_quantize_linear.py
'''
@paddle.jit.not_to_static
def quantize_linear(x,
                    scale,
                    zero_point,
                    bit_length=8,
                    quant_axis=-1,
                    name=None):
    helper = LayerHelper('quantize_linear', **locals())

    if in_dygraph_mode():
        attrs = ('bit_length', bit_length, 'quant_axis', quant_axis)
        output = core.ops.quantize_linear(x, scale, zero_point, *attrs)
        return output

    else:
        output = helper.create_variable_for_type_inference(dtype=x.dtype)

        inputs = {'X': x, 'Scale': scale, "ZeroPoint": zero_point}
        outputs = {'Y': output}

        helper.append_op(
            type="quantize_linear",
            inputs=inputs,
            attrs={'bit_length': bit_length,
                   'quant_axis': quant_axis},
            outputs=outputs)
        output.stop_gradient = True
        return output

class QuantizeNet(paddle.nn.Layer):
    def __init__(self, scale_shape, zero_shape, quant_axis):
        super(QuantizeNet, self).__init__()
        self.scale = paddle.to_tensor(
                randtool("float", -8, 8, scale_shape), dtype='float32')
        self.zero_points = paddle.to_tensor(
                np.zeros(zero_shape), dtype='float32')
        self.quant_axis = quant_axis

    def forward(self, x):
        y = quantize_linear(
                x,
                self.scale,
                self.zero_points,
                bit_length=8,
                quant_axis=self.quant_axis)
        return y.astype('float32')

def test_quantize_linear(name: str, input_data, quant_axis, dynamic_shape=[]):
    input_tensor = paddle.to_tensor(input_data)
    input_spec = paddle.static.InputSpec.from_tensor(input_tensor, name="input_tensor")

    scale_shape = 1
    zero_shape = 1

    # instance
    model = QuantizeNet(scale_shape, zero_shape, quant_axis)
    model.eval()

    # save model
    exportModel(name, model, [input_tensor], target_dir=sys.argv[1], dyn_shapes=dynamic_shape)

def main():
    # The op quantize_linear is always equal to 1, which means per-layer quantization.
    quant_axis = -1

    input_shape = [1,3,64,224]
    input_data = randtool("float", -sys.float_info.max, sys.float_info.max, input_shape).astype("float32")

    # static
    test_quantize_linear("quantize_linear_static", input_data, quant_axis)

    # dynamic
    dynamic_shape = [-1,3,64,224]
    test_quantize_linear("quantize_linear_dynamic", input_data, quant_axis, [dynamic_shape])

if __name__ == "__main__":
    main()
