import unittest
import numpy as np
import paddle
from quantize_ops import quantize_linear

class TestQuantizeLinearOp(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')
        self.rtol = 1e-5
        self.atol = 1e-8

    def test_quantize_linear_static(self):
        x = paddle.to_tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
        scale = paddle.to_tensor(np.array([0.1]).astype(np.float32))
        zero_point = paddle.to_tensor(np.array([0]).astype(np.float32))
        
        y = quantize_linear(x, scale, zero_point, bit_length=8, quant_axis=-1)
        
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(y >= -128) and np.all(y <= 127))

    def test_per_channel_quantization(self):
        x = paddle.to_tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
        scale = paddle.to_tensor(np.array([0.1, 0.2, 0.3]).astype(np.float32))
        zero_point = paddle.to_tensor(np.array([0, 0, 0]).astype(np.float32))
        
        y = quantize_linear(x, scale, zero_point, bit_length=8, quant_axis=1)
        
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(y >= -128) and np.all(y <= 127))

    def test_different_bit_lengths(self):
        x = paddle.to_tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
        scale = paddle.to_tensor(np.array([0.1]).astype(np.float32))
        zero_point = paddle.to_tensor(np.array([0]).astype(np.float32))
        
        y_4bit = quantize_linear(x, scale, zero_point, bit_length=4, quant_axis=-1)
        y_8bit = quantize_linear(x, scale, zero_point, bit_length=8, quant_axis=-1)
        
        self.assertTrue(np.all(y_4bit >= -8) and np.all(y_4bit <= 7))
        self.assertTrue(np.all(y_8bit >= -128) and np.all(y_8bit <= 127))

    def test_zero_point_handling(self):
        x = paddle.to_tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
        scale = paddle.to_tensor(np.array([0.1]).astype(np.float32))
        zero_point = paddle.to_tensor(np.array([10]).astype(np.float32))
        
        y = quantize_linear(x, scale, zero_point, bit_length=8, quant_axis=-1)
        
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(y >= -118) and np.all(y <= 137))  

    def test_dynamic_shapes(self):
        class DynamicQuantizeNet(paddle.nn.Layer):
            def __init__(self):
                super(DynamicQuantizeNet, self).__init__()
                self.scale = paddle.to_tensor(np.array([0.1]).astype(np.float32))
                self.zero_point = paddle.to_tensor(np.array([0]).astype(np.float32))

            def forward(self, x):
                return quantize_linear(x, self.scale, self.zero_point, bit_length=8, quant_axis=-1)

        net = DynamicQuantizeNet()
        net = paddle.jit.to_static(
            net,
            input_spec=[paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32')]
        )
        
        x = paddle.to_tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
        y = net(x)
        
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(y >= -128) and np.all(y <= 127))

if __name__ == '__main__':
    unittest.main()