################################################################################
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################


def everyone_is(list, value='None'):
    if [value == 'None']:
        value = list[0]
    return [e for e in list if e != value] == []


primitives_with_algs = ('binary', 'convolution', 'deconvolution', 'eltwise',
                        'lrn', 'pooling', 'reduction', 'resampling', 'rnn')


def alg_remove_primitive(alg):
    for p in primitives_with_algs:
        if alg.find(p) != -1:
            alg = alg[(alg.find(p) + len(p) + 1):]
    return alg


def convert_driver(prop_kind):
    driver = {
        'batch_normalization': 'bnorm',
        'binary': 'binary',
        'concat': 'concat',
        'convolution': 'conv',
        'deconvolution': 'deconv',
        'eltwise': 'eltwise',
        'inner_product': 'ip',
        'layer_normalization': 'lnorm',
        'lrn': 'lrn',
        'matmul': 'matmul',
        'pooling': 'pool',
        'pooling_v2': 'pool',
        'prelu': 'prelu',
        'reduction': 'reduction',
        'reorder': 'reorder',
        'resampling': 'resampling',
        'rnn': 'rnn',
        'shuffle': 'shuffle',
        'softmax': 'softmax',
        'sum': 'sum',
    }.get(prop_kind)
    return driver


def convert_engine(engine):
    return f"--engine={engine}"


def convert_dir(entry):
    # get base direction
    dir = {
        'forward_training': 'FWD_D',
        'forward_inference': 'FWD_I',
        'backward_data': 'BWD_D',
        'backward_weights': 'BWD_W',
        'backward': 'BWD_DW'
    }.get(entry['prop_kind'])

    if not dir:
        return ''

    found_bias = [
        e for e in entry['mds']
        if 'bia' == e['arg'] and e['data_type'] != 'undef'
    ]
    dir = 'FWD_B' if 'FWD' in dir and found_bias else dir
    dir = 'BWD_WB' if dir == 'BWD_W' and found_bias else dir
    if entry['prim_kind'] == 'rnn':
        return f"--prop={dir}"
    else:
        return f"--dir={dir}"


def convert_alg(entry):
    alg = entry['alg_kind']
    pk = entry['prim_kind']
    if alg:
        if pk == 'convolution':
            str = ''
            alg = alg_remove_primitive(alg)
            algs = {'winograd': 'WINO', 'direct': 'direct'}
            alg = algs.get(alg)
            if alg != None:
                str = f"--alg={alg}"
            return str
        if pk == 'eltwise':
            alg, alpha, beta = alg.split(' ')
            alpha = alpha.split(':')[1]
            beta = beta.split(':')[1]
            alg += f" --alpha={alpha} --beta={beta}"
            return f"--alg={alg}"
        elif pk == 'concat':
            axis = alg[len('axis:'):]
            return f"--axis={axis}"
        elif pk in ['batch_normalization', 'layer_normalization']:
            flags = alg[len('flags:'):]
            return f"--flags={flags}"
        elif pk == 'lrn':
            str = ''
            alg = alg_remove_primitive(alg)
            algs = {'across_channels': 'ACROSS', 'within_channel': 'WITHIN'}
            alg = algs.get(alg)
            if alg != None:
                str = f"--alg={alg}"
            return str
        elif pk == 'reduction':
            alg, p, eps = alg.split(' ')
            p = p.split(':')[1]
            eps = eps.split(':')[1]
            alg += f" --p={p} --eps={eps}"
            return f"--alg={alg}"
        elif pk == 'rnn':
            str = ''
            alg, dir, act = alg.split(' ')
            algs = {
                'vanilla_rnn': 'VANILLA_RNN',
                'vanilla_lstm': 'VANILLA_LSTM'
            }
            alg = algs.get(alg)
            if alg != None:
                str = f"--alg={alg}"
            return str
        elif pk == 'shuffle':
            axis, group = alg.split(' ')
            axis = axis[len('axis:'):]
            group = group[len('group:'):]
            return f"--axis={axis} --group={group}"
        elif pk == 'softmax':
            alg, axis = alg.split(' ')
            axis = axis[len('axis:'):]
            return f"--alg={alg} --axis={axis}"
        elif pk == 'pooling':
            return f"--alg={alg}"
        else:
            alg = alg_remove_primitive(alg)
            return f"--alg={alg}"
    else:
        return ''


def convert_bias_mask(mds):
    bia_mds = [md for md in mds if md['arg'] == 'bia']
    if len(bia_mds) != 0:
        bia_md = bia_mds[0]
        flags = bia_md['flags']['value'].split('_')
        if len(flags) > 1:
            mask = flags[1][4:]
            return f"--bia_mask={mask}"
    return ''


def convert_dts(mds, prim_kind):
    def convert_dts_common(mds):
        dts = [md['data_type'] for md in mds if md['data_type'] != 'undef']
        dt = dts[0]
        return f'--dt={dt}'

    def convert_dts_cfg(mds):
        cfg = "--cfg="
        mds_strip = mds
        # bias is not part of cfg
        mds_strip = [md for md in mds_strip if 'bia' not in md['arg']]
        # ws is not part of cfg
        mds_strip = [md for md in mds_strip if 'ws' not in md['arg']]
        common_dt = everyone_is([md['data_type'] for md in mds_strip])
        if common_dt and mds_strip[0]['data_type'] in ['f32', 'f16']:
            cfg += mds_strip[0]['data_type']
        else:
            for md in mds_strip:
                if md['arg'] in ['src', 'wei', 'dst']:
                    cfg += md['data_type']
        return cfg

    def convert_dts_cfg_with_bias(mds):
        cfg = convert_dts_cfg(mds)
        mds_bias = [md for md in mds if 'bia' in md['arg']]
        if len(mds_bias) != 0:
            md_bias = mds_bias[0]
            bias_dt = md_bias['data_type']
            cfg += ' ' + f"--bia_dt={bias_dt}"
        return cfg

    def convert_dts_cfg_pool(mds):
        cfg = mds[0]['data_type']
        return f"--cfg={cfg}"

    def convert_dts_all(mds):
        dts = ''
        md_args = ''
        for md in mds:
            md_arg = md['arg'][0]
            if md_args.find(md_arg) == -1:
                md_dt = md['data_type']
                dts += f' --{md_arg}dt={md_dt}'
                md_args += md_arg
        return dts

    def convert_dts_prelu(mds):
        data_md = [md for md in mds if 'data' in md['arg']][0]
        weights_md = [md for md in mds if 'wei' in md['arg']][0]

        data_dt = data_md['data_type']
        weights_dt = weights_md['data_type']

        return f' --sdt={data_dt}:{weights_dt}'

    def convert_dts_multiple_src(mds):
        src_dts = ''
        dts = ''
        first_src = True
        for md in mds:
            md_dt = md['data_type']
            md_arg = md['arg']
            if md_arg == 'src':
                if not first_src:
                    src_dts += f':{md_dt}'
                else:
                    src_dts += f' --{md_arg[0]}dt={md_dt}'
                    first_src = False
            else:
                if md_dt != 'undef':
                    dts += f' --{md_arg[0]}dt={md_dt}'
        return src_dts + dts

    convert_dts = {
        'batch_normalization': convert_dts_common,
        'binary': convert_dts_multiple_src,
        'concat': convert_dts_all,
        'convolution': convert_dts_cfg,
        'deconvolution': convert_dts_cfg,
        'eltwise': convert_dts_common,
        'inner_product': convert_dts_cfg,
        'layer_normalization': convert_dts_common,
        'lrn': convert_dts_common,
        'matmul': convert_dts_cfg_with_bias,
        'pooling': convert_dts_cfg_pool,
        'prelu': convert_dts_prelu,
        'reduction': convert_dts_all,
        'reorder': convert_dts_all,
        'resampling': convert_dts_all,
        'rnn': convert_dts_cfg,
        'shuffle': convert_dts_common,
        'softmax': convert_dts_common,
        'sum': convert_dts_multiple_src
    }

    convert = convert_dts.get(prim_kind)
    if convert != None:
        return convert(mds)
    # FIXME: Error handling. Throw an error if get() is used, but None returned
    return ''


def convert_tags(mds, prim_kind):
    def convert_tags_common(mds):
        tags = [md['tag'] for md in mds if md['tag'] != '']
        tag = tags[0]
        return f'--tag={tag}' if tag else ''

    def convert_tags_all(mds):
        tags = ''
        for md in mds:
            md_arg = md['arg'][0]
            # skip bias
            if md_arg == 'b':
                continue

            # pass wtag any for cases with compensation
            if md_arg == 'w' and md['flags']['value'] != 'f0':
                tags += f' --{md_arg}tag=any'
            else:
                md_tag = md['tag']
                tags += f' --{md_arg}tag={md_tag}'
        return tags

    def convert_tags_multiple_src(mds):
        src_tags = ''
        tags = ''
        first_src = False
        for md in mds:
            md_tag = md['tag']
            md_arg = md['arg']
            if md_arg == 'src':
                if first_src:
                    src_tags += f':{md_tag}'
                else:
                    src_tags += f' --{md_arg[0]}tag={md_tag}'
                    first_src = True
            else:
                if md_tag != '':
                    tags += f' --{md_arg[0]}tag={md_tag}'
        return src_tags + tags

    def convert_tags_prelu(mds):
        # FIXME: fix benchdnn input template
        data_md = [md for md in mds if 'data' in md['arg']][0]
        weights_md = [md for md in mds if 'wei' in md['arg']][0]

        data_tag = data_md['tag']
        weights_tag = weights_md['tag']

        return f' --stag={data_tag}:{weights_tag}'

    def convert_tags_rnn(mds):
        return ''

    def convert_tags_lnorm(mds):
        tag = convert_tags_common(mds)
        stat_md = ''
        for md in mds:
            if md['arg'] == 'stats':
                stat_tag = md['tag']

        return f'{tag} --stat_tag={stat_tag}'

    cvt_tags = {
        'batch_normalization': convert_tags_common,
        'binary': convert_tags_multiple_src,
        'concat': convert_tags_multiple_src,
        'convolution': convert_tags_all,
        'deconvolution': convert_tags_all,
        'eltwise': convert_tags_common,
        'inner_product': convert_tags_all,
        'layer_normalization': convert_tags_lnorm,
        'lrn': convert_tags_common,
        'matmul': convert_tags_all,
        'pooling': convert_tags_common,
        'prelu': convert_tags_prelu,
        'reduction': convert_tags_all,
        'reorder': convert_tags_all,
        'resampling': convert_tags_common,
        'rnn': convert_tags_rnn,
        'shuffle': convert_tags_common,
        'softmax': convert_tags_common,
        'sum': convert_tags_multiple_src
    }

    convert = cvt_tags.get(prim_kind)
    if convert:
        return convert(mds)
    return ''


def convert_flags(mds, prim_kind):
    def convert_flags_reorder(mds):
        def convert_flag(prefix, md):
            flag = ''
            flag_fields = md.get('flags')
            if flag_fields != None:
                cvt = {'s8_comp_mask': 's8s8_comp', 'zp_comp_mask': 'zp_comp'}
                for f in cvt.keys():
                    value = flag_fields.get(f)
                    if value != None:
                        benchdnn_flag = cvt[f] + ':' + value
                        if flag == '':
                            flag = benchdnn_flag
                        else:
                            flag += '+' + benchdnn_flag
            if flag != '':
                return f"--{prefix}flag={flag}"
            else:
                return ''

        flags = ''
        # FIXME: fix benchdnn input template
        input_md = [md for md in mds if 'src' in md['arg']][0]
        output_md = [md for md in mds if 'dst' in md['arg']][0]

        iflag = convert_flag('i', input_md)
        oflag = convert_flag('o', output_md)

        if iflag != '':
            flags += iflag
        if oflag != '':
            flags += ' ' + oflag
        return flags

    cvt_flags = {
        'reorder': convert_flags_reorder,
    }

    convert = cvt_flags.get(prim_kind)
    if convert:
        return convert(mds)
    return ''


def extract_attr(attrs, type):
    start_idx = attrs.find(type)
    if start_idx == -1:
        return ''

    start_idx += len(type) + 1
    end_symbol = ';'
    if type == 'post_ops':
        start_idx += 1
        end_symbol = '\''
    end_idx = attrs.find(end_symbol, start_idx)
    if type == 'post_ops':
        start_idx -= 1
        end_idx += 1
    return attrs[start_idx:end_idx]


def convert_scale_policy(value):
    # 4 is used by batched matmul
    masks = {0: 'common', 2: 'per_oc', 4: 'per_oc'}
    mask = masks.get(int(value))
    if mask:
        return mask
    # this is a workaround for tensors with mask more than 4
    return 'per_tensor'


def convert_zp_policy(value):
    # 4 is used by batched matmul
    masks = {0: 'common', 2: 'per_dim_1', 4: 'per_dim_2'}
    mask = masks.get(int(value))
    if mask:
        return mask
    # this is a workaround for tensors with mask more than 4
    return 'per_tensor'


def convert_post_ops(post_ops):
    def convert_binary_post_op(post_op):
        policy = convert_scale_policy(post_op['mask'])
        po = post_op['alg'] + ':' + post_op['dt'] + ':' + policy
        if post_op['tag'] != None:
            po += ':' + post_op['tag']
        return po

    def convert_dw_post_op(post_op):
        policy = convert_scale_policy(post_op['scales']['mask'])
        po = post_op['alg'] + ':' + post_op['dst_dt'] + ':' + policy
        if post_op['scales']['value'] != None:
            po += ':' + post_op['scales']['value']
        return po

    def convert_eltwise_post_op(post_op):
        benchdnn_p_op = post_op['alg']
        alpha = post_op['alpha']
        beta = post_op['beta']
        scale = post_op['scale']
        if alpha != '1.0':
            benchdnn_p_op += ':' + alpha
            if beta != '0.0':
                benchdnn_p_op += ':' + beta
                if alpha != '1.0':
                    benchdnn_p_op += ':' + scale
        return benchdnn_p_op

    def convert_sum_post_op(post_op):
        benchdnn_p_op = post_op['alg']
        if post_op['scale'] != 1.0:
            benchdnn_p_op += ':' + post_op['scale']
            if post_op['zp'] != 0:
                benchdnn_p_op += ':' + post_op['zp']
                if post_op['dt'] != '':
                    benchdnn_p_op += ':' + post_op['dt']
        return benchdnn_p_op

    def convert_prelu_post_op(post_op):
        benchdnn_p_op = post_op['alg']
        if post_op['mask'] != 0:
            benchdnn_p_op += ':' + post_op['mask']
        return benchdnn_p_op

    convert = {
        'binary': convert_binary_post_op,
        'dw': convert_dw_post_op,
        'eltwise': convert_eltwise_post_op,
        'sum': convert_sum_post_op,
        'prelu': convert_prelu_post_op,
    }

    benchdnn_postops = ''
    for e in post_ops:
        for k in convert.keys():
            if k in e['alg']:
                cvt = convert.get(k)
                if benchdnn_postops != '':
                    benchdnn_postops += '+'
                benchdnn_postops += cvt(e)
                break
    return benchdnn_postops


def convert_oscale(oscale):
    benchdnn_oscale = convert_scale_policy(oscale['mask'])
    value = oscale['value']
    if value != None:
        benchdnn_oscale += ':' + value
    return benchdnn_oscale


def convert_scales(scales):
    res = []
    for arg in scales.keys():
        s = scales[arg]
        policy = convert_scale_policy(s['mask'])
        benchdnn_scale = arg + ':' + policy
        value = s['value']
        # benchdnn requires user to pass a value
        if value == None:
            value = '0.5'
        # benchdnn doesn't allow user to pass * without an actual value
        if value == '*':
            value = '0.5'
        benchdnn_scale += ':' + value
        res.append(benchdnn_scale)
    return '+'.join(res)


def convert_zero_points(zero_points):
    res = []
    for arg in zero_points.keys():
        zp = zero_points[arg]
        policy = convert_zp_policy(zp['mask'])
        benchdnn_zp = arg + ':' + policy
        value = zp['value']
        # benchdnn requires user to pass a value
        if value == None:
            value = '1'
        # benchdnn doesn't allow user to pass * without an actual value
        if value == '*':
            value = '1'
        benchdnn_zp += ':' + value
        res.append(benchdnn_zp)
    return '+'.join(res)


def convert_scratchpad_mode(scratchpad_mode):
    return scratchpad_mode


def convert_attrs(exts):
    converters = {
        'attr-post-ops': convert_post_ops,
        'attr-oscale': convert_oscale,
        'attr-scales': convert_scales,
        'attr-zero-points': convert_zero_points,
        'attr-scratchpad': convert_scratchpad_mode
    }

    benchdnn_attrs = ''
    for e in converters.keys():
        attr = exts.get(e)
        if attr != None:
            if benchdnn_attrs != '':
                benchdnn_attrs += ' '
            benchdnn_attrs += f"--{e}=" + converters[e](attr)
    return benchdnn_attrs


def convert_shapes(shapes, prim_kind):
    if prim_kind == 'binary':
        shapes = shapes.split(' ')[0]
    return f"{shapes}"


class InputGenerator:
    """
    Generates an input for benchdnn from internal representation.
    """
    def __init__(self, writer):
        self.__writer = writer

    def generate(self, input, split_by_driver=False):
        data = {}

        def generate_case(entry, add_driver=True):
            case = ''
            if add_driver:
                case += '--' + convert_driver(entry['prim_kind'])
            # reset everything, because benchdnn is a state machine and options
            # affect all following test cases
            case += ' --reset'
            # allow extended set of tags
            case += ' --allow-enum-tags-only=0'

            case += ' ' + convert_engine(entry['engine'])
            # XXX: direction depends on mds (FWD_B is forward + defined bias md)
            case += ' ' + convert_dir(entry)
            case += ' ' + convert_alg(entry)
            if entry['prim_kind'] == 'matmul':
                case += ' ' + convert_bias_mask(entry['mds'])
            # XXX: data types configuration is not unified across drivers
            case += ' ' + convert_dts(entry['mds'], entry['prim_kind'])
            case += ' ' + convert_tags(entry['mds'], entry['prim_kind'])
            case += ' ' + convert_flags(entry['mds'], entry['prim_kind'])
            case += ' ' + convert_attrs(entry['exts'])
            case += ' ' + convert_shapes(entry['shapes'], entry['prim_kind'])
            return case

        if split_by_driver:
            for key, value in input.items():
                case = generate_case(value, False) + '\n'
                driver_cases = data.get(convert_driver(value['prim_kind']))
                if driver_cases:
                    data[convert_driver(value['prim_kind'])] += case
                else:
                    data[convert_driver(value['prim_kind'])] = case
        else:
            for key, value in input.items():
                case = generate_case(value, True) + '\n'
                if data.get('all'):
                    data['all'] += case
                else:
                    data['all'] = case
        return data
