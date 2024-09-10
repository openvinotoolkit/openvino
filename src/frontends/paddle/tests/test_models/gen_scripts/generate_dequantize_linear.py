import unittest
import numpy as np
import paddle
from quantize_ops import dequantize_linear

class TestDequantizeLinearOp(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')
        self.rtol = 1e-5
        self.atol = 1e-8

    def test_dequantize_linear_static(self):
        x = paddle.to_tensor(np.random.randint(-128, 128, (2, 3, 4, 5)).astype(np.int8))
        scale = paddle.to_tensor(np.array([0.1]).astype(np.float32))
        zero_point = paddle.to_tensor(np.array([0]).astype(np.float32))
        
        y = dequantize_linear(x, scale, zero_point, bit_length=8, quant_axis=-1)
        
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, paddle.float32)

    def test_per_channel_dequantization(self):
        x = paddle.to_tensor(np.random.randint(-128, 128, (2, 3, 4, 5)).astype(np.int8))
        scale = paddle.to_tensor(np.array([0.1, 0.2, 0.3]).astype(np.float32))
        zero_point = paddle.to_tensor(np.array([0, 0, 0]).astype(np.float32))
        
        y = dequantize_linear(x, scale, zero_point, bit_length=8, quant_axis=1)
        
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, paddle.float32)

    def test_zero_point_handling(self):
        x = paddle.to_tensor(np.random.randint(-128, 128, (2, 3, 4, 5)).astype(np.int8))
        scale = paddle.to_tensor(np.array([0.1]).astype(np.float32))
        zero_point = paddle.to_tensor(np.array([10]).astype(np.float32))
        
        y = dequantize_linear(x, scale, zero_point, bit_length=8, quant_axis=-1)
        
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, paddle.float32)

    def test_dynamic_shapes(self):
        class DynamicDequantizeNet(paddle.nn.Layer):
            def __init__(self):
                super(DynamicDequantizeNet, self).__init__()
                self.scale = paddle.to_tensor(np.array([0.1]).astype(np.float32))
                self.zero_point = paddle.to_tensor(np.array([0]).astype(np.float32))

            def forward(self, x):
                return dequantize_linear(x, self.scale, self.zero_point, bit_length=8, quant_axis=-1)

        net = DynamicDequantizeNet()
        net = paddle.jit.to_static(
            net,
            input_spec=[paddle.static.InputSpec(shape=[None, 3, None, None], dtype='int8')]
        )
        
        x = paddle.to_tensor(np.random.randint(-128, 128, (2, 3, 4, 5)).astype(np.int8))
        y = net(x)
        
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, paddle.float32)

    def test_dequantize_extreme_values(self):
        x = paddle.to_tensor(np.array([-128, 0, 127]).astype(np.int8))
        scale = paddle.to_tensor(np.array([0.1]).astype(np.float32))
        zero_point = paddle.to_tensor(np.array([0]).astype(np.float32))
        
        y = dequantize_linear(x, scale, zero_point, bit_length=8, quant_axis=-1)
        
        expected = np.array([-12.8, 0, 12.7]).astype(np.float32)
        np.testing.assert_allclose(y.numpy(), expected, rtol=self.rtol, atol=self.atol)

if __name__ == '__main__':
    unittest.main()