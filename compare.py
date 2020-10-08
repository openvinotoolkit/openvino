import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

np.random.seed(324)
torch.manual_seed(32)

# TODO: there is an issue with torch.cat when test multiple models.
models = {
    # models.alexnet: (227, 227),
    # models.resnet18: (227, 227),
    models.segmentation.deeplabv3_resnet50: (240, 320),
}

for model_func, size in models.items():
    print('run', model_func)
    inp_size = [1, 3, size[0], size[1]]

    # Create model
    model = model_func(pretrained=True)
    inp = Variable(torch.randn(inp_size))
    model.eval()
    ref = model(inp)

    # Convert to OpenVINO IR
    import mo_pytorch
    mo_pytorch.convert(model, input_shape=inp_size, model_name='model')

    # Run model with OpenVINO and compare outputs
    from openvino.inference_engine import IECore

    ie = IECore()
    net = ie.read_network('model.xml', 'model.bin')
    exec_net = ie.load_network(net, 'CPU')
    out = exec_net.infer({'input': inp.detach().numpy()})

    if isinstance(ref, torch.Tensor):
        ref = {'': ref}
    for out0, ref0 in zip(out.values(), ref.values()):
        diff = np.max(np.abs(out0 - ref0.detach().numpy()))
        assert(diff < 1e-5)
        print('diff:', diff)


# # Create model
# # model = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True, min_size=200, max_size=300)
# # help(models.segmentation.deeplabv3)
# model = models.segmentation.deeplabv3_resnet50(pretrained=True)
# # model = models.alexnet(pretrained=True)

# # Run model to get reference output
# inp = Variable(torch.randn([1, 3, 240, 320]))
# model.eval()
# with torch.no_grad():
#     ref = model(inp)
#     # torch.onnx.export(model, inp, 'deeplabv3_onnx.onnx')

# # Convert to OpenVINO IR
# import mo_pytorch
# # mo_pytorch.convert(model, input_shape=[1, 3, 500, 500], model_name='deeplabv3', log_level='DEBUG')
# mo_pytorch.convert(model, input_shape=[1, 3, 240, 320], model_name='deeplabv3')

# # Run model with OpenVINO and compare outputs
# from openvino.inference_engine import IECore

# ie = IECore()
# net = ie.read_network('deeplabv3.xml', 'deeplabv3.bin')
# exec_net = ie.load_network(net, 'CPU')
# out = exec_net.infer({'input': inp.detach().numpy()})
# # out = next(iter(out.values()))
# # print('diff:', np.max(np.abs(out - ref.detach().numpy())))  # 1.3113022e-06

# print(out.keys())
# for out0, ref0 in zip(out.values(), ref.values()):
#     print('diff:', np.max(np.abs(out0 - ref0.detach().numpy())))
# # print('diff:', np.max(np.abs(out['PyTorchInterpolate_562590'] - ref['out'].detach().numpy())))
# # print('diff:', np.max(np.abs(out['PyTorchInterpolate_576586'] - ref['aux'].detach().numpy())))
