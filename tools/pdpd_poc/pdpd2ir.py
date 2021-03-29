import numpy as np
import ngraph as ng

import framework_pb2


def download_pdpd_resnet50():
    import paddlehub as hub

    module = hub.Module(name="resnet_v2_50_imagenet")
    return module.directory


def make_ng_node(inputs: dict, nodes: dict, op, block):
    creators = {
        'conv2d': conv2d_creator,
        'batch_norm': batch_norm_creator,
        'relu': relu_creator,
        'pool2d': pool2d_creator,
        'elementwise_add': elementwise_add_creator,
        'mul': mul_creator,
        'scale': scale_creator
    }
    assert op.type in creators, 'No creator for {} layer'.format(op.type)
    inputs_preproc = {}
    for input, input_names in inputs.items():
        inputs_preproc[input] = [nodes[input_name] for input_name in input_names]
    return creators[op.type](inputs_preproc, op, block)


def get_attr(op, name: str, field: str, default=None, dst_type=None):
    attrs = [a for a in op.attrs if a.name == name]
    if len(attrs) == 0:
        # there is no requested attribute in the protobuf message
        return default
    elif len(attrs) > 1:
        raise Exception(
            'Found multiple entries for attribute name {} when at most one is expected. Protobuf message with '
            'the issue: {}.', name, op)
    else:
        res = getattr(attrs[0], field)
        if dst_type is not None:
            return dst_type(res)
        else:
            return res


def conv2d_creator(inputs: dict, op, block):
    assert len(inputs['Input']) == 1
    data = inputs['Input'][0]
    assert len(inputs['Filter']) == 1
    filter = inputs['Filter'][0]
    assert len(inputs['Bias']) == 0
    assert len(inputs['ResidualData']) == 0
    # TODO: resolve padding according to spec
    return ng.convolution(data,
                          filter,
                          strides=get_attr(op, 'strides', 'ints'),
                          pads_begin=get_attr(op, 'paddings', 'ints'),
                          pads_end=get_attr(op, 'paddings', 'ints'),
                          dilations=get_attr(op, 'dilations', 'ints'))


def batch_norm_creator(inputs: dict, op, block):
    assert len(inputs['X']) == 1
    assert len(inputs['Scale']) == 1
    assert len(inputs['Bias']) == 1
    assert len(inputs['Mean']) == 1
    assert len(inputs['Variance']) == 1
    data = inputs['X'][0]
    gamma = inputs['Scale'][0]
    beta = inputs['Bias'][0]
    mean = inputs['Mean'][0]
    variance = inputs['Variance'][0]
    return ng.batch_norm_inference(data, gamma, beta, mean, variance, epsilon=get_attr(op, 'epsilon', 'f'))


def relu_creator(inputs: dict, op, block):
    data = inputs['X'][0]
    return ng.relu(data)


def pool2d_creator(inputs: dict, op, block):
    data = inputs['X'][0]
    # TODO: resolve padding according to spec
    pooling_type = get_attr(op, 'pooling_type', 's')
    global_pooling = get_attr(op, 'global_pooling', 'b')
    if pooling_type == 'max' and not global_pooling:
        return ng.max_pool(data,
                           strides=get_attr(op, 'strides', 'ints'),
                           pads_begin=get_attr(op, 'paddings', 'ints'),
                           pads_end=get_attr(op, 'paddings', 'ints'),
                           kernel_shape=get_attr(op, 'ksize', 'ints'))
    elif pooling_type == 'avg' and global_pooling:
        # TODO: resolve axes according to rank
        return ng.reduce_mean(data, np.array([2, 3]), keep_dims=True)
    else:
        raise Exception('Unsupported pooling type')


def elementwise_add_creator(inputs: dict, op, block):
    x = inputs['X'][0]
    y = inputs['Y'][0]
    # TODO: resolve broadcast
    return ng.add(x, y)


def mul_creator(inputs: dict, op, block):
    x = inputs['X'][0]
    y = inputs['Y'][0]
    assert x.output(0).get_partial_shape().rank.is_static
    x_rank = x.output(0).get_partial_shape().rank.get_length()
    assert y.output(0).get_partial_shape().rank.is_static and y.output(0).get_partial_shape().rank.get_length() == 2
    if x_rank > 2:
        shape = ng.shape_of(x)
        x_num_col_dims = get_attr(op, 'x_num_col_dims', 'i')
        split = ng.variadic_split(shape,
                                  axis=0,
                                  split_lengths=np.array([x_num_col_dims, x_rank - x_num_col_dims]))
        first_dim = ng.reduce_prod(split.output(0), reduction_axes=0)
        first_dim = ng.reshape(first_dim, np.array([1]), special_zero=False)
        second_dim = ng.reduce_prod(split.output(1), reduction_axes=0)
        second_dim = ng.reshape(second_dim, np.array([1]), special_zero=False)
        out_shape = ng.concat([first_dim, second_dim], axis=0)
        x = ng.reshape(x, out_shape, special_zero=False)
    return ng.matmul(x, y, transpose_a=False, transpose_b=False)


def scale_creator(inputs: dict, op, block):
    data = inputs['X'][0]
    return ng.multiply(data, np.array(get_attr(op, 'scale', 'f'), dtype=np.float32))


DTYPE_PADDLE_NUMPY_MAP = {
    np.float32: framework_pb2.VarType.FP32,
    np.float64: framework_pb2.VarType.FP64,
    np.int16: framework_pb2.VarType.INT16,
    np.int32: framework_pb2.VarType.INT32,
    np.int64: framework_pb2.VarType.INT64,
    np.bool: framework_pb2.VarType.BOOL,
    framework_pb2.VarType.FP32: np.float32,
    framework_pb2.VarType.FP64: np.float64,
    framework_pb2.VarType.INT16: np.int16,
    framework_pb2.VarType.INT32: np.int32,
    framework_pb2.VarType.INT64: np.int64,
    framework_pb2.VarType.BOOL: np.bool
}


def read_tensor(var, model_dir):
    assert var.type.type == framework_pb2.VarType.LOD_TENSOR
    tensor = var.type.lod_tensor.tensor
    with open(model_dir + '/' + var.name, mode='rb') as file:
        fileContent = file.read()
    assert tensor.data_type == framework_pb2.VarType.FP32
    t_len = np.prod(tensor.dims) * 4
    # TODO: figure out what is written in header of a file
    return np.frombuffer(fileContent[len(fileContent) - t_len:], dtype=np.float32).reshape(tensor.dims)


def convert_model(model_dir):
    fw_model = framework_pb2.ProgramDesc()

    with open(model_dir + '/__model__', 'rb') as f:
        fw_model.ParseFromString(f.read())

    nodes_dict = {}
    parameter_nodes = []
    result_nodes = []

    global_block = fw_model.blocks[0]
    for var in global_block.vars:
        if var.name.endswith('feed') or var.name.endswith('fetch'):
            # feed and fetch is the names of inputs and outputs of the model
            continue
        if not var.persistable:
            continue
        tensor = read_tensor(var, model_dir)
        nodes_dict[var.name] = ng.constant(tensor, tensor.dtype, var.name)

    for block in fw_model.blocks:
        vars_dict = dict(zip([var.name for var in block.vars],
                             [var.type for var in block.vars]))
        for i, op in enumerate(block.ops):
            outputs_dict = dict(zip([output.parameter for output in op.outputs],
                                    [output.arguments for output in op.outputs]))
            inputs_dict = dict(zip([inp.parameter for inp in op.inputs],
                                   [inp.arguments for inp in op.inputs]))
            if op.type == 'feed':
                layer_name = outputs_dict['Out'][0]
                var = vars_dict[layer_name]
                assert var.type == framework_pb2.VarType.LOD_TENSOR
                tensor_desc = var.lod_tensor.tensor
                param = ng.parameter(tensor_desc.dims, DTYPE_PADDLE_NUMPY_MAP[tensor_desc.data_type], name=layer_name)
                nodes_dict[layer_name] = param
                parameter_nodes.append(param)
            elif op.type == 'fetch':
                input_node = inputs_dict['X'][0]
                assert input_node in nodes_dict
                result_nodes.append(ng.result(nodes_dict[input_node]))
            else:
                node = make_ng_node(inputs_dict, nodes_dict, op, block)
                for outp_var_list in outputs_dict.values():
                    assert len(outp_var_list) <= 1
                    if len(outp_var_list) == 1:
                        nodes_dict[outp_var_list[0]] = node

    return ng.Function(result_nodes, parameter_nodes, "PDPD_Resnet50_Function")


if __name__ == "__main__":
    import cv2

    img = cv2.imread("cat3.bmp")
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, [2, 0, 1]) / 255
    img = np.expand_dims(img, 0)
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    model_path = download_pdpd_resnet50() + '/model'

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
    # executable_network = ie.load_network(ie_network, 'CPU')

    net = ie.read_network('/media/data/OpenVINO/openvino/build/PDPD_Resnet50_Function.xml',
                          '/media/data/OpenVINO/openvino/build/PDPD_Resnet50_Function.bin')
    executable_network = ie.load_network(net, 'CPU')

    # output = executable_network.infer(
    #     {'@HUB_resnet_v2_50_imagenet@image': img.astype(np.float32)})
    output = executable_network.infer(
        {'Parameter_267': img.astype(np.float32)})

    print(np.abs(result[0] - list(output.values())[0]).max())
    print(np.abs(result[1] - list(output.values())[1]).max())

    print('')
