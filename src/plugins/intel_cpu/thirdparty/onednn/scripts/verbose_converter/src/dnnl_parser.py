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


class LogParser:
    """
    Parses a log file with oneDNN verbose and converts it into internal
    representation.
    """
    def __init__(self, writer, input=''):
        # each data entry is a dictionary that consists of:
        # engine(str),
        # primitive(str),
        # implementation(str),
        # prop_kind(str),
        # alg_kind(str),
        # mds({ arg(str) : { data_type(str), format_kind(str), tag(str), flags(str) }})
        # shapes(str)
        # extensions(str)
        # time(float)
        self.__raw_data = []
        self.__data = {}
        self.__writer = writer
        self.__input = input

    def process(self):
        """
        Adds data from the last log file.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        def convert_primitive(log_entry, template):
            """
            Converts oneDNN verbose primitive entry into the internal
            representation.
            """
            def convert_mds(log_mds):
                mds = []
                for md in log_mds.split(' '):
                    # arg_dt:padding:format_kind:tag:flags
                    fields = md.split(':')
                    arg_dt = fields[0]
                    padding = fields[1]
                    format_kind = fields[2]
                    tag = fields[3]
                    flags = {}
                    flags['value'] = fields[4]
                    if len(fields) > 5:
                        flag_fields = fields[5:]
                        for f in flag_fields:
                            if f[:3] == 's8m':
                                flags['s8_comp_mask'] = f[3:]
                            if f[:3] == 'zpm':
                                flags['zp_comp_mask'] = f[3:]

                    data_type = arg_dt.split('_')[-1]
                    arg = arg_dt[:-len(data_type) - 1]
                    mds.append({
                        'arg': arg,
                        'data_type': data_type,
                        'padding': padding,
                        'format_kind': format_kind,
                        'tag': tag,
                        'flags': flags
                    })
                return mds

            def convert_alg(alg):
                found_alg = alg.find('alg')
                if found_alg != -1:
                    alg = alg[len('alg') + 1:]
                return alg

            def convert_prim_kind(prim_kind):
                if prim_kind == 'pooling_v2':
                    prim_kind = 'pooling'
                return prim_kind

            def convert_exts(exts):
                def extract_attr(attrs, type):
                    start_idx = attrs.find(type)
                    if start_idx == -1:
                        return ''

                    start_idx += len(type) + 1
                    end_symbol = ' '
                    end_idx = attrs.find(end_symbol, start_idx)
                    return attrs[start_idx:end_idx]

                def convert_structure_to_ir_seq(ir, value):
                    params = value.split(':')
                    fields = list(ir.keys())
                    ir.update(
                        (fields[i], params[i])
                        for i in range(0, min(len(params), len(fields))))
                    return ir

                def convert_post_ops(value):
                    def convert_binary_post_op(value):
                        p_op = {
                            'alg': '',
                            'dt': 'f32',
                            'mask': '0',
                            'tag': None
                        }
                        p_op = convert_structure_to_ir_seq(p_op, value)
                        p_op['prim_kind'] = 'binary'
                        return p_op

                    def convert_dw_post_op(value):
                        p_op = {
                            'alg': '',
                            'dst_dt': 'f32',
                            'wei_dt': 'f32',
                            'scales': {
                                'mask': '0',
                                'value': None
                            }
                        }
                        params = value.split(':')
                        len_params = len(params)
                        p_op['alg'] = params[0]
                        if len_params > 1:
                            p_op['dst_dt'] = params[1]
                        if len_params > 2:
                            p_op['wei_dt'] = 's8'
                            p_op['scales']['mask'] = params[2]
                        if len_params > 3:
                            p_op['scales']['value'] = params[3]
                        return p_op

                    def convert_eltwise_post_op(value):
                        p_op = {
                            'alg': '',
                            'alpha': '1.0',
                            'beta': '0.0',
                            'scale': '1.0'
                        }
                        return convert_structure_to_ir_seq(p_op, value)

                    def convert_sum_post_op(value):
                        p_op = {'alg': '', 'scale': '1.0', 'zp': '0', 'dt': ''}
                        return convert_structure_to_ir_seq(p_op, value)

                    def convert_prelu_post_op(value):
                        p_op = {'alg': '', 'mask': '0'}
                        return convert_structure_to_ir_seq(p_op, value)

                    convert = {
                        'binary': convert_binary_post_op,
                        'dw': convert_dw_post_op,
                        'eltwise': convert_eltwise_post_op,
                        'sum': convert_sum_post_op,
                        'prelu': convert_prelu_post_op,
                    }

                    entries = value.split('+')
                    postops = []
                    for e in entries:
                        for k in convert.keys():
                            if k in e:
                                cvt = convert.get(k)
                                postops.append(cvt(e))
                                break
                    return postops

                def convert_oscale(value):
                    oscale = {'mask': '0', 'value': None}
                    return convert_structure_to_ir_seq(oscale, value)

                def convert_scales(value):
                    res = {}
                    scales = value.split('+')
                    for s in scales:
                        scale = {'mask': '0', 'value': None}
                        arg = s[:s.find(':')]
                        s_wo_arg = s[s.find(':')+1:]
                        res[arg] = convert_structure_to_ir_seq(scale, s_wo_arg)
                    return res

                def convert_zero_points(value):
                    res = {}
                    zp_value = value.split('+')
                    for zp in zp_value:
                        arg = zp[:zp.find(':')]
                        zp_value_wo_arg = zp[zp.find(':')+1:]
                        zp_dict = {'mask': '0', 'value': None}
                        res[arg] = convert_structure_to_ir_seq(zp_dict, zp_value_wo_arg)
                    return res

                def convert_scratchpad_mode(value):
                    return value

                converters = {
                    'attr-post-ops': convert_post_ops,
                    'attr-oscale': convert_oscale,
                    'attr-scales': convert_scales,
                    'attr-zero-points': convert_zero_points,
                    'attr-scratchpad': convert_scratchpad_mode
                }
                attrs = {}
                for e in converters.keys():
                    attr = extract_attr(exts, e)
                    if attr != '':
                        attrs[e] = converters[e](attr)
                return attrs

            def convert_pass(v):
                return v

            convert = {
                'prim_kind': convert_prim_kind,
                'mds': convert_mds,
                'alg_kind': convert_alg,
                'exts': convert_exts
            }

            dnnl_to_ir = {
                'engine': 'engine',
                'prim_kind': 'primitive',
                'impl': 'implementation',
                'prop_kind': 'prop_kind',
                'mds': 'memory_descriptors',
                'exts': 'attributes',
                'alg_kind': 'auxiliary',
                'shapes': 'problem_desc',
                'time': 'exec_time',
                'timestamp': 'timestamp'
            }

            ir_req = [ 'engine', 'prim_kind', 'impl', 'prop_kind', 'mds',
                    'exts', 'alg_kind', 'shapes']

            entry = {}

            t = template.split(',')
            for key, value in dnnl_to_ir.items():
                notification_level = "WARN" if key in ir_req else "INFO"
                try:
                    idx = t.index(value)
                    if idx != -1:
                        cvt = convert.get(key)
                        if cvt is None:
                            cvt = convert_pass
                        field = log_entry[idx]
                        try:
                            entry[key] = cvt(field)
                        except:
                            self.__writer.print(
                                f"Parser: parsing entry error: {field}: {value}",
                                notification_level)
                    else:
                        self.__writer.print(f"Parser: Unknown entry: {value}",
                                            notification_level)
                except:
                    self.__writer.print(f"Parser: skipping empty entry: {key}",
                                        notification_level)
            return entry

        verbose_template = "dnnl_verbose,operation,engine,primitive," + \
            "implementation,prop_kind,memory_descriptors,attributes," + \
            "auxiliary,problem_desc"

        i = len(self.__data)
        for line in self.__input:
            self.__raw_data.append(line.rstrip())
            l_raw = line.split(",")
            marker = l_raw[0]
            if marker == "dnnl_verbose":
                event = l_raw[1]
                if event == "info":
                    opt = l_raw[2]
                    if opt == "prim_template":
                        verbose_template = "dnnl_verbose," + line.split(':')[1]
                if event == "exec":
                    l_converted = convert_primitive(l_raw, verbose_template)
                    if l_converted:
                        self.__data[i] = l_converted
                        i = i + 1

    def get_data(self):
        """
        Returns information about DNN calls.

        Parameters
        ----------
        None

        Returns
        -------
        data
        """

        return self.__data

    def dump(self, converted=False):
        """
        Prints data parsed from input to stdout.

        Parameters
        ----------
        converted (default: False) -- If True dump() prints data in internal
        represenataion, otherwise prints data in the original form.

        Returns
        -------
        None
        """

        if converted:
            [
                self.__writer.print(f"{key}, {value}", 'STDIO')
                for key, value in self.__data.items()
            ]
        else:
            [self.__writer.print(d, 'STDIO') for d in self.__raw_data]
