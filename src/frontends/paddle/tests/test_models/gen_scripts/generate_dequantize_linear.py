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
    https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/tests/test_auto_scan_dequantize_linear.py
'''
@paddle.jit.not_to_static
def dequantize_linear(x,
                      scale,
                      zero_point,
                      bit_length=8,
                      quant_axis=-1,
                      name=None):
    helper = LayerHelper('dequantize_linear', **locals())

    attrs = ('bit_length', bit_length, 'quant_axis', quant_axis)
    if in_dygraph_mode():
        return _legacy_C_ops.dequantize_linear(x, scale, zero_point, *attrs)
    else:
        output = helper.create_variable_for_type_inference(dtype=x.dtype)

        inputs = {'X': x, 'Scale': scale, "ZeroPoint": zero_point}
        outputs = {'Y': output}

        helper.append_op(
            type="dequantize_linear",
            inputs=inputs,
            attrs={'bit_length': bit_length,
                   'quant_axis': quant_axis},
            outputs=outputs)
        output.stop_gradient = True
        return output

class DequantizeNet(paddle.nn.Layer):
    def __init__(self, scale_shape, zero_shape, quant_axis):
        super(DequantizeNet, self).__init__()
        self.scale = paddle.to_tensor(
                randtool("float", -8, 8, scale_shape), dtype='float32')
        self.zero_points = paddle.to_tensor(
                np.zeros(zero_shape), dtype='float32')
        self.quant_axis = quant_axis

    def forward(self, x):
        y = dequantize_linear(
                x,
                self.scale,
                self.zero_points,
                bit_length=8,
                quant_axis=self.quant_axis)
        return y.astype('float32')

def test_dequantize_linear(name: str, input_data, quant_axis, dynamic_shape=[]):
    input_tensor = paddle.to_tensor(input_data)
    input_spec = paddle.static.InputSpec.from_tensor(input_tensor, name="input_tensor")

    if quant_axis == 0 or quant_axis == 1:
        scale_shape = input_data.shape[quant_axis]
        zero_shape = input_data.shape[quant_axis]
    elif quant_axis == -1:
        scale_shape = 1
        zero_shape = 1
    else:
        assert("quant_axis is not in the range [-1, 0, 1]!")

    # instance
    model = DequantizeNet(scale_shape, zero_shape, quant_axis)
    model.eval()

    # save model
    exportModel(name, model, [input_tensor], target_dir=sys.argv[1], dyn_shapes=dynamic_shape)

def main():
    input_shape = [4,3,64,224]
    quant_axis = -1
    input_data = randtool("float", -128, 127, input_shape).astype("float32")

    # for data static
    test_dequantize_linear("dequantize_linear_per_layer_static", input_data, quant_axis)

    # for data dynamic
    dynamic_shape = [-1,3,64,224]
    test_dequantize_linear("dequantize_linear_per_layer_dynamic", input_data, quant_axis, [dynamic_shape])

    # for weight with quant_axis 0
    quant_axis = 0
    test_dequantize_linear("dequantize_linear_per_channel0_static", input_data, quant_axis)

    # for weight with quant_axis 1
    quant_axis = 1
    test_dequantize_linear("dequantize_linear_per_channel1_static", input_data, quant_axis)

if __name__ == "__main__":
    main()
