import operator


def accum_param_to_proto(layer):
    attributes = ['top_width', 'top_height', 'size_divisible_by', 'have_reference']
    return '\n'.join([
        '  accum_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        '  }'
    ])


def accum_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      *['  bottom: "{}"'.format(i) for i in layer.inputs.keys()],
                      accum_param_to_proto(layer),
                      '}'
                      ])


def activation_params_to_proto(layer):
    return '\n'.join([
        '  {}_param {}'.format(layer.attrs['type'], '{'),
        '  }'
    ])


def activation_to_proto(layer):
    layer_type = 'Sigmoid' if layer.attrs['type'] == 'sigmoid' else 'TanH'
    return '\n'.join([
        common_to_proto(layer, layer_type=layer_type),
        bottom_to_proto(layer),
        activation_params_to_proto(layer),
        '}'
    ])


def arg_max_param_to_proto(layer):
    attributes = ['out_max_val', 'axis', 'top_k']
    return '\n'.join([
        '  argmax_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        '  }'
    ])


def arg_max_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      bottom_to_proto(layer),
                      arg_max_param_to_proto(layer),
                      '}'
                      ])


def data_augmentation_param_to_proto(layer):
    attributes = layer.attrs.keys()
    return '\n'.join([
        '  augmentation_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        '  }'
    ])


def data_augmentation_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      bottom_to_proto(layer),
                      data_augmentation_param_to_proto(layer),
                      '}'
                      ])


def batch_norm_param_to_proto(layer):
    return '\n'.join([
        '  batch_norm_param {',
        '    use_global_stats: {}'.format(int(True)),
        '    eps: {}'.format(layer.attrs['epsilon']),
        '  }'
    ])


def batch_norm_to_proto(layer):
    return '\n'.join([common_to_proto(layer, layer_type='BatchNorm'),
                      bottom_to_proto(layer),
                      '  param { lr_mult: 0 decay_mult: 0 }',
                      '  param { lr_mult: 0 decay_mult: 0 }',
                      '  param { lr_mult: 0 decay_mult: 0 }',
                      batch_norm_param_to_proto(layer),
                      '}'
                      ])


def bias_filler_to_proto():
    return '\n'.join([
        '  bias_filler {',
        '    type: "constant"',
        '    value: 0.2',
        '  }'
    ])


def bottom_to_proto(layer):
    return '  bottom: "{}"'.format(list(layer.inputs.keys())[0])


def common_to_proto(layer, layer_type=None):
    layer_type = layer_type if layer_type else layer.type
    return '\n'.join([
        'layer {',
        '  name: "{}"'.format(layer.name),
        '  type: "{}"'.format(layer_type),
        '  top: "{}"'.format(layer.name)
    ])


def concat_param_to_proto(layer):
    return '\n'.join([
        '  concat_param {',
        '    axis: {}'.format(layer.attrs['axis']),
        '  }'
    ])


def concat_to_proto(layer):
    bottoms = []
    for k in layer.inputs.keys():
        _id = int(''.join(c for c in k if c.isdigit()))
        for _ in layer.inputs[k]:
            bottoms.append('  bottom: "{}{}"'.format(k[:len(k)-len(str(_id))], _id))
            _id += 1
    return '\n'.join([common_to_proto(layer),
                      *bottoms,
                      concat_param_to_proto(layer),
                      '}'
                      ])


def correlation_param_to_proto(layer):
    # TODO: check that isn't required or return
    correlation_type = {'caffe.CorrelationType.MULTIPLY': 'MULTIPLY'}
    attributes = ['pad', 'kernel_size', 'max_displacement', 'stride_1', 'stride_2']
    return '\n'.join([
        '  correlation_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        # '    correlation_type: {}'.format(correlation_type[layer.attrs['correlation_type']]),
        '  }'
    ])


def correlation_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      *['  bottom: "{}"'.format(i) for i in layer.inputs.keys()],
                      correlation_param_to_proto(layer),
                      '}'
                      ])


def conv_param_to_proto(layer):
    return '\n'.join([
        '  convolution_param {',
        '    num_output: {}'.format(layer.attrs['output']),
        '    kernel_h: {}'.format(layer.attrs['kernel'][0]),
        '    kernel_w: {}'.format(layer.attrs['kernel'][1]),
        '    stride_h: {}'.format(layer.attrs['strides'][0]),
        '    stride_w: {}'.format(layer.attrs['strides'][1]),
        '    pad_h: {}'.format(layer.attrs['pads_begin'][0]),
        '    pad_w: {}'.format(layer.attrs['pads_begin'][1]),
        weight_filler_to_proto(),
        bias_filler_to_proto(),
        '}'
    ])


def conv_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      bottom_to_proto(layer),
                      '  param { lr_mult: 1 decay_mult: 1 }',
                      '  param { lr_mult: 2 decay_mult: 0 }',
                      conv_param_to_proto(layer),
                      '}'
                      ])


def crop_param_to_proto(layer):
    input_dims_len = len(layer.get_inputs_shape(layer.get_inputs_names()[0]))
    axis_len = len(layer.attrs['axis'].split(','))
    return '\n'.join([
        '  crop_param {',
        '    axis: {}'.format(input_dims_len - axis_len),
        '    offset: {}'.format(layer.attrs['offset'].split(',')[0]),
        '  }'
    ])


def crop_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      *['  bottom: "{}"'.format(i) for i in layer.inputs.keys()],
                      crop_param_to_proto(layer),
                      '}'
                      ])


def ctc_greedy_decoder_param_to_proto(layer):
    attributes = ['ctc_merge_repeated']
    return '\n'.join([
        '  ctc_decoder_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        '  }'
    ])


def ctc_greedy_decoder_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      *['  bottom: "{}"'.format(i) for i in layer.inputs.keys()],
                      ctc_greedy_decoder_param_to_proto(layer),
                      '}'
                      ])


def ctc_beam_search_decoder_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      *['  bottom: "{}"'.format(i) for i in layer.inputs.keys()],
                      '  ctc_decoder_param {',
                      '  }'
                      '}'
                      ])


def detection_output_param_to_proto(layer):
    code_type = {'caffe.PriorBoxParameter.CORNER': 1}
    attributes = ['num_classes', 'share_location', 'background_label_id', 'keep_top_k',
                  'confidence_threshold', 'variance_encoded_in_target', 'input_width', 
                  'input_height', 'normalized']
    return '\n'.join([
        '  detection_output_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        '    code_type: {}'.format(code_type[layer.attrs['code_type']]),
        '    nms_param {',
        '      nms_threshold: {}'.format(layer.attrs['nms_threshold']),
        '      top_k: {}'.format(layer.attrs['top_k']),
        '      eta: {}'.format(layer.attrs['eta']),
        '    }',
        '  }'
    ])


def detection_output_to_proto(layer):
    sorted_inputs = sorted(layer.inputs.items(), key=operator.itemgetter(0))
    return '\n'.join([common_to_proto(layer),
                      *['  bottom: "{}"'.format(i[0]) for i in sorted_inputs],
                      detection_output_param_to_proto(layer),
                      '}'
                      ])


def dropout_param_to_proto(layer):
    return '\n'.join([
        '  dropout_param {',
        '    dropout_ratio: {}'.format(layer.attrs['dropout_ratio']),
        '  }'
    ])


def dropout_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      bottom_to_proto(layer),
                      dropout_param_to_proto(layer),
                      '}'
                      ])


def eltwise_param_to_proto(layer):
    return '\n'.join([
        '  eltwise_param {',
        '    operation: {}'.format(
            'PROD' if layer.attrs['operation'].lower() == 'mul' else layer.attrs['operation'].upper()),
        '  }'
    ])


def eltwise_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      *['  bottom: "{}"'.format(i) for i in layer.inputs.keys()],
                      eltwise_param_to_proto(layer),
                      '}'
                      ])


def elu_param_to_proto(layer):
    attributes = ['alpha']
    return '\n'.join([
        '  elu_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        '  }'
    ])


def elu_to_proto(layer):
    layer_type = "ELU"
    return '\n'.join([common_to_proto(layer, layer_type=layer_type),
                      bottom_to_proto(layer),
                      elu_param_to_proto(layer),
                      '}'
                      ])


def flatten_param_to_proto(layer):
    attributes = ['axis', 'end_axis']
    return '\n'.join([
        '  flatten_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        '  }'
    ])


def flatten_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      bottom_to_proto(layer),
                      flatten_param_to_proto(layer),
                      '}'
                      ])


def fullyconnected_param_to_proto(layer):
    return '\n'.join([
        '  inner_product_param {',
        '    num_output: {}'.format(layer.attrs['out_size']),
        '  }'
    ])


def fullyconnected_to_proto(layer):
    return '\n'.join([common_to_proto(layer, layer_type='InnerProduct'),
                      bottom_to_proto(layer),
                      '  param { lr_mult: 1 decay_mult: 1 }',
                      '  param { lr_mult: 2 decay_mult: 0 }',
                      fullyconnected_param_to_proto(layer),
                      '}'
                      ])


def grn_param_to_proto(layer):
    return '\n'.join([
        '  grn_param {',
        '    bias: {}'.format(layer.attrs['bias']),
        '  }'
    ])


def grn_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      bottom_to_proto(layer),
                      grn_param_to_proto(layer),
                      '}'
                      ])


def input_to_proto(layer):
    return '\n'.join([
        common_to_proto(layer),
        '  input_param {',
        '    shape: {',
        *['      dim: {}'.format(dim) for dim in layer.get_out_shapes()[0]],
        '    }',
        '  }',
        '}'
    ])


def interp_param_to_proto(layer):
    pad_attrs = ['pad_beg', 'pad_end']
    factor_attrs = ['shrink_factor', 'zoom_factor']
    size_attrs = ['width', 'height']
    return '\n'.join([
        '  interp_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in factor_attrs if layer.attrs[attr] != 1],
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in size_attrs if layer.attrs[attr] != 0],
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in pad_attrs if layer.attrs[attr]],
        '  }'
    ])


def interp_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      bottom_to_proto(layer),
                      interp_param_to_proto(layer),
                      '}'
                      ])


def mvn_param_to_proto(layer):
    attributes = ['normalize_variance', 'across_channels', 'eps']
    return '\n'.join([
        '  mvn_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        '  }'
    ])


def mvn_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      bottom_to_proto(layer),
                      mvn_param_to_proto(layer),
                      '}'
                      ])


def lrn_param_to_proto(layer):
    attributes = ['local_size', 'alpha', 'beta']
    return '\n'.join([
        '  lrn_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        '    norm_region: {}'.format(0 if layer.attrs['region'] == 'across' else 1),
        '  }'
    ])


def lrn_to_proto(layer):
    return '\n'.join([common_to_proto(layer, layer_type='LRN'),
                      bottom_to_proto(layer),
                      lrn_param_to_proto(layer),
                      '}'
                      ])


def normalize_param_to_proto(layer):
    return '\n'.join([
        '  norm_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in ['channel_shared', 'across_spatial', 'eps']],
        '  }'
    ])


def normalize_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      bottom_to_proto(layer),
                      normalize_param_to_proto(layer),
                      '}'
                      ])


def permute_param_to_proto(layer):
    return '\n'.join([
        '  permute_param {',
        *['    order: {}'.format(order) for order in layer.attrs['order']],
        '}'
    ])


def permute_to_proto(layer):
    return '\n'.join([common_to_proto(layer, layer_type='Permute'),
                      bottom_to_proto(layer),
                      permute_param_to_proto(layer),
                      '}'
                      ])


def pool_param_to_proto(layer):
    pool = 'MAX' if layer.attrs['pool_method'] == 'max' else 'AVE'
    return '\n'.join([
        '  pooling_param {',
        '    pool: {}'.format(pool),
        '    kernel_h: {}'.format(layer.attrs['kernel'][0]),
        '    kernel_w: {}'.format(layer.attrs['kernel'][1]),
        '    stride_h: {}'.format(layer.attrs['strides'][0]),
        '    stride_w: {}'.format(layer.attrs['strides'][1]),
        '    pad_h: {}'.format(layer.attrs['pads_begin'][0]),
        '    pad_w: {}'.format(layer.attrs['pads_begin'][1]),
        '  }'
    ])


def pool_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      bottom_to_proto(layer),
                      pool_param_to_proto(layer),
                      '}'
                      ])


def power_params_to_proto(layer):
    attributes = ['power', 'scale', 'shift']
    return '\n'.join([
        '  power_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        '  }'
    ])


def power_to_proto(layer):
    return '\n'.join([
        common_to_proto(layer),
        bottom_to_proto(layer),
        power_params_to_proto(layer),
        '}'
    ])


def power_file_to_proto(layer):
    return '\n'.join([
        common_to_proto(layer),
        bottom_to_proto(layer),
        '  power_file_param {',
        '  }',
        '}'
    ])


def prelu_params_to_proto(layer):
    variance_norm = {'caffe.FillerParameter.FAN_IN': 0,
                     'caffe.FillerParameter.FAN_OUT': 1,
                     'caffe.FillerParameter.AVERAGE': 2
                     }
    filler_attributes = ['min', 'max', 'mean', 'sparse', 'std']
    return '\n'.join([
        '  prelu_param {',
        '    filler {',
        *['        {}: {}'.format(attr, layer.attrs[attr]) for attr in filler_attributes],
        '        variance_norm: {}'.format(variance_norm[layer.attrs['variance_norm']]),
        '        type: "constant"',
        '    }',
        '    channel_shared: {}'.format(layer.attrs['channel_shared']),
        '  }'
    ])


def prelu_to_proto(layer):
    return '\n'.join([
        common_to_proto(layer),
        bottom_to_proto(layer),
        prelu_params_to_proto(layer),
        '}'
    ])


def prior_box_param_to_proto(layer):
    attributes = ['step', 'min_size', 'max_size', 'offset', 'flip', 'clip']
    return '\n'.join([
        '  prior_box_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        *['    aspect_ratio: {}'.format(ar) for ar in layer.attrs['aspect_ratio'].split(',')],
        *['    variance: {}'.format(v) for v in layer.attrs['variance'].split(',')],
        '  }'
    ])


def prior_box_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      *['  bottom: "{}"'.format(i) for i in layer.inputs.keys()],
                      prior_box_param_to_proto(layer),
                      '}'
                      ])


def prior_box_clustered_param_to_proto(layer):
    attributes = ['width', 'height', 'clip', 'flip', 'img_w', 'img_h', 'step_w', 'step_h', 'offset']
    return '\n'.join([
        '  prior_box_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        *['    variance: {}'.format(v) for v in layer.attrs['variance'].split(',')],
        '  }'
    ])


def prior_box_clustered_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      *['  bottom: "{}"'.format(i) for i in layer.inputs.keys()],
                      prior_box_clustered_param_to_proto(layer),
                      '}'
                      ])


def proposal_params_to_proto(layer):
    attrs = ['base_size', 'feat_stride', 'pre_nms_topn', 'post_nms_topn', 'nms_thresh', 'min_size']
    return '\n'.join([
        '  proposal_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attrs],
        '  }'
    ])


def proposal_to_proto(layer):
    return '\n'.join([
        common_to_proto(layer),
        bottom_to_proto(layer),
        proposal_params_to_proto(layer),
        '}'
    ])


def ps_roi_pool_param_to_proto(layer):
    attrs = ['output_dim', 'group_size', 'spatial_scale']
    return '\n'.join([
        '  psroi_pooling_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attrs],
        '  }'
    ])


def ps_roi_pool_to_proto(layer):
    d_sorted_inputs = sorted(layer.inputs.items(), key=operator.itemgetter(0), reverse=True)
    return '\n'.join([common_to_proto(layer),
                      *['  bottom: "{}"'.format(i[0]) for i in d_sorted_inputs],
                      ps_roi_pool_param_to_proto(layer),
                      '}'
                      ])


def region_yolo_params_to_proto(layer):
    attrs = ['classes', 'coords', 'num']
    return '\n'.join([
        '  region_yolo_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attrs],
        '  }',
        '  flatten_param {',
        '    axis: {}'.format(layer.attrs['axis']),
        '    end_axis: {}'.format(layer.attrs['end_axis']),
        '  }'
    ])


def region_yolo_to_proto(layer):
    return '\n'.join([
        common_to_proto(layer),
        bottom_to_proto(layer),
        region_yolo_params_to_proto(layer),
        '}'
    ])


def relu_params_to_proto(layer):
    engine = {'caffe.ReLUParameter.DEFAULT': 0,
              'caffe.ReLUParameter.CAFFE': 1,
              'caffe.ReLUParameter.CUDNN': 2
              }

    return '\n'.join([
        '  relu_param {',
        '    negative_slope: {}'.format(layer.attrs['negative_slope']),
        '    engine: {}'.format(engine[layer.attrs.get('engine', 'caffe.ReLUParameter.DEFAULT')]),
        '  }'
    ])


def relu_to_proto(layer):
    return '\n'.join([
        common_to_proto(layer),
        bottom_to_proto(layer),
        relu_params_to_proto(layer),
        '}'
    ])


def resample_params_to_proto(layer):
    attrs = ['height', 'width', 'antialias']

    resample_type = {'caffe.ResampleParameter.NEAREST': 'NEAREST',
                     'caffe.ResampleParameter.LINEAR': 'LINEAR',
                     'caffe.ResampleParameter.CUBIC': 'CUBIC'
                     }

    return '\n'.join([
        '  resample_param {',
        '    type: {}'.format(resample_type.get(layer.attrs.get('type', 'caffe.ResampleParameter.NEAREST'))),
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attrs],
        '  }'
    ])


def resample_to_proto(layer):
    return '\n'.join([
        common_to_proto(layer),
        bottom_to_proto(layer),
        resample_params_to_proto(layer),
        '}'
    ])


def reshape_param_to_proto(layer):
    return '\n'.join([
        '  reshape_param {',
        '    shape{',
        *['      dim: {}'.format(shape) for shape in layer.attrs['dim'].split(',')],
        '    }',
        '  }'
    ])


def reshape_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      bottom_to_proto(layer),
                      reshape_param_to_proto(layer),
                      '}'
                      ])


def roi_pool_param_to_proto(layer):
    attrs = ['pooled_h', 'pooled_w', 'spatial_scale']
    return '\n'.join([
        '  roi_pooling_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attrs],
        '  }'
    ])


def roi_pool_to_proto(layer):
    d_sorted_inputs = sorted(layer.inputs.items(), key=operator.itemgetter(0), reverse=True)
    return '\n'.join([common_to_proto(layer),
                      *['  bottom: "{}"'.format(i[0]) for i in d_sorted_inputs],
                      roi_pool_param_to_proto(layer),
                      '}'
                      ])


def scale_param_to_proto():
    return '\n'.join([
        '  scale_param {',
        '    bias_term: {}'.format(int(True)),
        '  }'
    ])


def scale_to_proto(layer):
    return '\n'.join([common_to_proto(layer, layer_type='Scale'),
                      bottom_to_proto(layer),
                      scale_param_to_proto(),
                      '}'
                      ])


def sigmoid_to_proto(layer):
    return '\n'.join([
        common_to_proto(layer),
        bottom_to_proto(layer),
        '}'
    ])


def simpler_nms_param_to_proto(layer):
    attributes = ['pre_nms_topn', 'post_nms_topn', 'cls_threshold', 'iou_threshold', 'feat_stride', 'min_bbox_size']
    return '\n'.join([
        '  simpler_nms_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attributes],
        *['    scale: {}'.format(scale) for scale in layer.attrs['scale']],
        '  }'
    ])


def simpler_nms_to_proto(layer):
    d_sorted_inputs = sorted(layer.inputs.items(), key=operator.itemgetter(0), reverse=True)
    return '\n'.join([common_to_proto(layer),
                      *['  bottom: "{}"'.format(i[0]) for i in d_sorted_inputs],
                      simpler_nms_param_to_proto(layer),
                      '}'
                      ])


def slice_param_to_proto(layer):
    return '\n'.join([
        '  slice_param {',
        '    axis: {}'.format(layer.attrs['axis']),
        *['    slice_point: {}'.format(p) for p in layer.attrs['slice_point']],
        '  }'
    ])


def slice_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      *['  top: "{}{}"'.format(layer.type, layer.get_out_port_ids()[0] + i) for i in range(
                          0, len(layer.attrs['slice_point']))],
                      bottom_to_proto(layer),
                      slice_param_to_proto(layer),
                      '}'
                      ])


def soft_max_param_to_proto(layer):
    return '\n'.join([
        '  softmax_param {',
        '    axis: {}'.format(layer.attrs['axis']),
        '  }'
    ])


def soft_max_to_proto(layer):
    return '\n'.join([common_to_proto(layer, layer_type='Softmax'),
                      bottom_to_proto(layer),
                      soft_max_param_to_proto(layer),
                      '}'
                      ])


def st_param_to_proto(layer):
    attrs = ['to_compute_dU']
    size_attrs = ['output_W', 'output_H']
    return '\n'.join([
        '  st_param {',
        '    transform_type: "{}"'.format(layer.attrs['transform_type']),
        '    sampler_type: "{}"'.format(layer.attrs['sampler_type']),
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in attrs],
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in size_attrs if attr in layer.attrs],
        '  }'
    ])


def st_to_proto(layer):
    sorted_inputs = sorted(layer.inputs.items(), key=operator.itemgetter(0))
    return '\n'.join([common_to_proto(layer),
                      *['  bottom: "{}"'.format(i[0]) for i in sorted_inputs],
                      st_param_to_proto(layer),
                      '}'
                      ])


def split_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      bottom_to_proto(layer),
                      '}'
                      ])


def tanh_to_proto(layer):
    return '\n'.join([
        common_to_proto(layer),
        bottom_to_proto(layer),
        '}'
    ])


def tile_param_to_proto(layer):
    return '\n'.join([
        '  tile_param {',
        *['    {}: {}'.format(attr, layer.attrs[attr]) for attr in ['axis', 'tiles']],
        '  }'
    ])


def tile_to_proto(layer):
    return '\n'.join([common_to_proto(layer),
                      bottom_to_proto(layer),
                      tile_param_to_proto(layer),
                      '}'
                      ])


def weight_filler_to_proto():
    return '\n'.join([
        '  weight_filler {',
        '    type: "xavier"',
        '    std: 0.2',
        '  }'
    ])
