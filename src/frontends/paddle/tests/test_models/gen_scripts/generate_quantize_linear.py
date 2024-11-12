import numpy as np
import paddle
from save_model import exportModel
import sys
from paddle.base.framework import in_dygraph_mode
if paddle.__version__ >= '2.6.0':
    from paddle.base.layer_helper import LayerHelper
else:
    from paddle.fluid.layer_helper import LayerHelper
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

@paddle.jit.not_to_static
def quantize_linear(x, scale, zero_point, bit_length=8, quant_axis=-1, name=None):
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
            attrs={'bit_length': bit_length, 'quant_axis': quant_axis},
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

def test_quantize_linear(name: str, input_data, quant_axis, bit_length=8, scale=None, zero_point=None, dynamic_shape=[]):
    input_tensor = paddle.to_tensor(input_data)
    input_spec = paddle.static.InputSpec.from_tensor(input_tensor, name="input_tensor")

    scale_shape = input_tensor.shape[quant_axis] if quant_axis != -1 else 1
    zero_shape = input_tensor.shape[quant_axis] if quant_axis != -1 else 1

    if scale is None:
        scale = randtool("float", 0.1, 0.3, scale_shape).astype("float32")
    if zero_point is None:
        zero_point = np.zeros(zero_shape).astype("float32")

    scale_tensor = paddle.to_tensor(scale)
    zero_point_tensor = paddle.to_tensor(zero_point)

    model = QuantizeNet(scale_shape, zero_shape, quant_axis)
    model.eval()

    exportModel(name, model, [input_tensor], target_dir=sys.argv[1], dyn_shapes=dynamic_shape)

def main():
    quant_axis = -1

    # Test: Static quantization
    input_data = randtool("float", -1.0, 1.0, [2, 3, 4, 5]).astype("float32")
    test_quantize_linear("quantize_linear_static", input_data, quant_axis)

    # Test: Per-channel quantization
    quant_axis = 1
    scale = [0.1, 0.2, 0.3]
    zero_point = [0, 0, 0]
    test_quantize_linear("quantize_linear_per_channel", input_data, quant_axis, scale=scale, zero_point=zero_point)

    # Test: Zero point handling
    zero_point = [10]
    test_quantize_linear("quantize_linear_zero_point", input_data, quant_axis, zero_point=zero_point)

    # Test: Dynamic shapes
    dynamic_shape = [-1, 3, -1, -1]
    test_quantize_linear("quantize_linear_dynamic", input_data, quant_axis, dynamic_shape=dynamic_shape)

if __name__ == "__main__":
    main()
