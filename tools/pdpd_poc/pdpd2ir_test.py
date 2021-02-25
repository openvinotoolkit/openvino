from unittest import TestCase

from pdpd2ir import convert_model
import numpy as np
import ngraph as ng

test_path = '/tmp/pdpd2ir_test/model'


class TestConversion(TestCase):
    @staticmethod
    def infer_ie(func, inp_dict: dict):
        from openvino.inference_engine import IECore

        ie = IECore()
        ie_network = ng.function_to_cnn(func)
        executable_network = ie.load_network(ie_network, 'CPU')
        return executable_network.infer(inp_dict)

    @staticmethod
    def validate(var: list, inp_dict: dict):
        import paddle
        from paddle import fluid

        paddle.enable_static()
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        res_pdpd = exe.run(fluid.default_main_program(), fetch_list=var, feed=inp_dict)

        fluid.io.save_inference_model(test_path, list(inp_dict.keys()), var, exe)

        func = convert_model(test_path)

        res_ie = TestConversion.infer_ie(func, inp_dict)

        return np.all(np.isclose(res_pdpd[0], list(res_ie.values())[0], rtol=1e-4, atol=1e-5))

    def test_convert_pure_conv_model(self):
        import paddle
        from paddle import fluid
        paddle.enable_static()

        inp_blob = np.random.randn(1, 3, 224, 224).astype(np.float32)

        x = fluid.data(name='x', shape=[1, 3, 224, 224], dtype='float32')
        my_conv = fluid.layers.conv2d(input=x, num_filters=64, filter_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                      dilation=(1, 1), groups=1, bias_attr=False)

        result = self.validate([my_conv], {'x': inp_blob})

        self.assertTrue(result)

    def test_convert_conv_model(self):
        import paddle
        from paddle import fluid
        paddle.enable_static()

        inp_blob = np.random.randn(1, 3, 224, 224).astype(np.float32)

        x = fluid.data(name='x', shape=[1, 3, 224, 224], dtype='float32')
        my_conv = fluid.layers.conv2d(input=x, num_filters=64, filter_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                      dilation=(1, 1), groups=1, bias_attr=False)
        bn = fluid.layers.batch_norm(my_conv, act='relu', is_test=True)

        result = self.validate([bn], {'x': inp_blob})

        self.assertTrue(result)

    def test_convert_resnet50(self):
        import cv2

        img = cv2.imread("cat3.bmp")
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, [2, 0, 1]) / 255
        img = np.expand_dims(img, 0)
        img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std

        import paddlehub as hub

        module = hub.Module(name="resnet_v2_50_imagenet")

        model_path = module.directory + '/model'

        import paddle
        from paddle import fluid
        paddle.enable_static()

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        [program, feed, fetchs] = fluid.io.load_inference_model(model_path, exe)

        result = exe.run(program, fetch_list=fetchs,
                         feed={feed[0]: img.astype(np.float32)},
                         feed_var_name='@HUB_resnet_v2_50_imagenet@feed',
                         fetch_var_name='@HUB_resnet_v2_50_imagenet@fetch')

        ng_function = convert_model(model_path)

        ie_network = ng.function_to_cnn(ng_function)
        ie_network.reshape({'@HUB_resnet_v2_50_imagenet@image': [1, 3, 224, 224]})
        ie_network.serialize('PDPD_Resnet50_Function.xml', 'PDPD_Resnet50_Function.bin')

        from openvino.inference_engine import IECore

        ie = IECore()
        executable_network = ie.load_network(ie_network, 'CPU')

        output = executable_network.infer(
            {'@HUB_resnet_v2_50_imagenet@image': img.astype(np.float32)})

        self.assertTrue(np.all(np.isclose(result[0], list(output.values())[0], rtol=1e-4, atol=1e-5)))
