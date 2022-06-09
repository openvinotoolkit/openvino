#!/usr/bin/python3

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.runtime as ov
from openvino.runtime.passes import Manager
from openvino.runtime.utils.types import get_dtype
import numpy as np
import sys, os
from graphviz import Digraph, Source
import ctypes, time, re
import argparse
import dot_to_html

def get_value_strings(n, nlimit = 8):
    values = n.get_vector()
    limit = min(len(values), nlimit)
    return [str(v) for v in values[:limit]]

# print Model in readable text
def generate_str(model, show_rt_info = False):
    out2name = {}
    nameid = 1
    simpleconst_node2vstr = {}
    ilist = [i.get_node().get_name() for i in model.inputs]
    result = []

    def get_rt_info(n):
        return {k:str(v) for k,v in n.get_rt_info().items()}

    result.append("model({}):".format(",".join(ilist)))

    for k, v in model.get_rt_info().items():
        result.append("  {}={}".format(k,v))
    for n in model.get_ordered_ops():
        # collect output and also allocate output names
        rt_info = get_rt_info(n)
        if "reginfo" in rt_info or "effectiveAddress" in rt_info:
            if "reginfo" in rt_info:
                varname = "vmm{}".format(rt_info["reginfo"])
            else:
                varname = "t{}".format(nameid)
                nameid += 1
            str_output = varname
            args = []
            if "effectiveAddress" in rt_info:
                args.append("[r{}]".format(rt_info["effectiveAddress"]))
            for i in n.inputs():
                r2 = get_rt_info(i.get_source_output().get_node())
                if "reginfo" in r2:
                    args.append("vmm{}".format(r2["reginfo"]))
                else:
                    args.append(out2name[i.get_source_output()])

            for k, out in enumerate(n.outputs()):
                out2name[out] = varname if num_out == 1 else "{}[{}]".format(varname, k)
        else:
            out_types = []
            varname = "t{}".format(nameid)
            nameid += 1
            num_out = len(n.outputs())
            for k, out in enumerate(n.outputs()):
                out_types.append("Tensor<{}x{}>".format(
                                    "x".join([str(s) for s in out.get_shape()]),
                                    out.get_element_type().get_type_name()))
                out2name[out] = varname if num_out == 1 else "{}[{}]".format(varname, k)

            #out_types
            str_out_types = out_types[0] if len(out_types)==1 else "tuple({})".format(",".join(out_types))
            str_output = "{} {}".format(str_out_types, varname)
        
            # collect source output names of corresponding inputs
            args = []
            for i in n.inputs():
                o = i.get_source_output()
                if o in simpleconst_node2vstr:
                    args.append(simpleconst_node2vstr[o])
                else:
                    args.append(out2name[o])

        # generate psuedo code
        type_name = n.get_type_name()
        friendly_name = n.get_friendly_name()
        rt_info = n.get_rt_info()
        if type_name == "ExecutionNode" and "layerType" in rt_info:
            type_name = str(rt_info["layerType"])
        attrs = ["{}={}".format(k, v) for k,v in n.get_attributes().items()]
        rtinfo = ["{}={}".format(k, v) for k,v in rt_info.items()]

        comment = friendly_name
        comment = "" if len(comment)==0 else "   # {}".format(comment)
        if type_name.startswith("Constant"):
            if len(n.get_vector()) <= 8:
                simpleconst_node2vstr[n.outputs()[0]] = "[{}]".format(get_value_strings(n, 8))
            else:
                result.append("    {} = {}([{}]) {}".format(
                                    str_output,
                                    type_name,
                                    get_value_strings(n, 16),
                                    comment))
        else:
            result.append("    {} = {}({}{}) {}".format(
                        str_output,
                        type_name,
                        ",".join(args),
                        "" if len(attrs) == 0 else ("," if len(args)>0 else "") + (",".join(attrs)),
                        comment ))
        if (show_rt_info and rtinfo):
            result.append("\t\t\t#rt_info:\n\t\t\t#\t{}\n".format("\n\t\t\t#\t".join(rtinfo)))

    olist = [out2name[i] for i in model.outputs]
    result.append("    return {}".format(",".join(olist)))
    return "\n".join(result)

def print_model(model, show_rt_info = False):
    print(generate_str(model, show_rt_info))

def generate_graph(model, fontsize=12, graph_name="", detailed_label=False):
    # create all nodes before edges
    g = Digraph(graph_name, graph_attr={"outputorder":"edgesfirst"})
    node2name = {}
    name2node = {}
    data_map = {}
    data_color = {}
    precision2ctype = {
        "I8":ctypes.c_int8,
        "U8":ctypes.c_uint8,
        "I32": ctypes.c_int32,
        "FP32":ctypes.c_float,
        "BF16":ctypes.c_int16,
    }

    def gen_rand_color():
        if not hasattr(gen_rand_color, "color_hue"):
            gen_rand_color.color_hue = 0
        gen_rand_color.color_hue = (gen_rand_color.color_hue + 5/8) % 1
        return "{:.3f} 1 0.7".format(gen_rand_color.color_hue)

    def strings2label(strings, nlimit = 20, line_limit = 1):
        r = ""
        line = 0
        prev_cnt = 0
        for s in strings:
            if len(r) + len(s) - prev_cnt > nlimit:
                r += "\\n"
                prev_cnt = len(r)
                line += 1
                if line >= line_limit:
                    r += "..."
                    break
            r += s + ","
        return r.rstrip(",")

    op2color = {"Parameter":"gold", "Result":"deeppink", "Constant":"gray85", "Const":"gray85"}

    inode2index = {input.node:k for k,input in enumerate(model.inputs)}

    def name_normalize(n):
        name = n.get_friendly_name()
        # add input id if it's input node of the model
        if n in inode2index:
            name += "_#{}".format(inode2index[n])
        name = name.replace("<","(").replace(">",")").replace(":","_")
        if len(graph_name):
            return '{}'.format(graph_name, name)
        return '{}'.format(name)

    # statistics on execTime
    execTimeMcs_total = 0
    execTimeMcs_by_type = {}
    execTimeMcs_by_node = {}
    for n in model.get_ordered_ops():
        friendly_name = name_normalize(n)
        rt_info = n.get_rt_info()
        type_name = n.get_type_name()
        if type_name == "ExecutionNode" and "layerType" in rt_info:
            type_name = str(rt_info["layerType"])
        if "primitiveType" in rt_info:
            type_name += "({})".format(rt_info["primitiveType"])
        execTimeMcs = 0
        if ("execTimeMcs" in rt_info):
            execTimeMcs = rt_info["execTimeMcs"]
            try:
                execTimeMcs = int(execTimeMcs)
            except:
                execTimeMcs = 0
        execTimeMcs_by_node[n] = execTimeMcs
        execTimeMcs_total += execTimeMcs
        if type_name in execTimeMcs_by_type:
            execTimeMcs_by_type[type_name][0] += execTimeMcs
            execTimeMcs_by_type[type_name][1] += 1
        else:
            execTimeMcs_by_type[type_name] = [execTimeMcs, 0]

    num_limit = 10
    sort_execTimeMcs_by_type = []
    acc_percentage = 0
    for (type_name, (t, cnt)) in sorted(execTimeMcs_by_type.items(), key=lambda x: x[1][0], reverse=True):
        percentage = 0 if execTimeMcs_total <= 0 else t*100/execTimeMcs_total
        acc_percentage += percentage
        sort_execTimeMcs_by_type.append("{:>6}%  {:>6}%  {} x {}".format(
                                f"{acc_percentage:.1f}", f"+{percentage:.1f}", cnt, type_name))
        if acc_percentage >= 90 and len(sort_execTimeMcs_by_type) >= num_limit:
            break
    kwargs = {"shape":'box',
            "style":'filled',
            "fillcolor":"gold",
            "fontsize":str(fontsize + 2),
            "margin":"0,0","width":"0","height":"0",
            "tooltip":"\n".join(sort_execTimeMcs_by_type)}
    g.node(name="ProfileSummary_ByType",
            label="ProfileSummary\\nByType",
            **kwargs)

    if execTimeMcs_total > 0:
        sort_execTimeMcs_by_name = []
        acc_percentage = 0
        for (n, t) in sorted(execTimeMcs_by_node.items(), key=lambda x: x[1], reverse=True):
            friendly_name = name_normalize(n)
            type_name = n.get_type_name()
            rt_info = n.get_rt_info()
            if type_name == "ExecutionNode" and "layerType" in rt_info:
                type_name = str(rt_info["layerType"])
            percentage = 0 if execTimeMcs_total <= 0 else t*100/execTimeMcs_total
            acc_percentage += percentage
            sort_execTimeMcs_by_name.append("{:>6}%  {:>6}%  {}({})".format(
                                f"{acc_percentage:.1f}", f"+{percentage:.1f}", friendly_name, type_name))
            if acc_percentage >= 90 and len(sort_execTimeMcs_by_name) >= num_limit:
                break

        kwargs = {"shape":'box',
            "style":'filled',
            "fillcolor":"gold",
            "fontsize":str(fontsize + 2),
            "margin":"0,0","width":"0","height":"0",
            "tooltip":"\n".join(sort_execTimeMcs_by_name)}
        g.node(name="ProfileSummary_ByName",
                label="ProfileSummary\\nByName",
                **kwargs)

    for nindex, n in enumerate(model.get_ordered_ops()):
        friendly_name = name_normalize(n)
        rt_info = n.get_rt_info()
        type_name = n.get_type_name()
        if friendly_name in name2node:
            print("WARNNING:  {} (type {}) already exist as {}, skipped!".format(
                friendly_name, type_name,
                name2node[friendly_name].get_type_name()))
            continue
        
        # ExecutionNode is fake wrapper of runtime node
        # and this type name gives less information than friendly_name
        if type_name == "ExecutionNode" and "layerType" in rt_info:
            type_name = str(rt_info["layerType"])

        attrs = ["{}={}".format(k, v) for k,v in n.get_attributes().items()]
        def rtinfo2string(k, v):
            if k == "originalLayersNames":
                v = "\n    " + str(v).replace(",","\n    ")
            return "{}={}".format(k, v)
        rtinfo = [rtinfo2string(k, v) for k,v in rt_info.items()]

        # originalLayersNames gives mapping between runtime nodes and orginal nodes
        fsize = fontsize
        color = op2color[type_name] if type_name in op2color else "cyan"
        if type_name == "Constant":
            vstr = get_value_strings(n)
            label = strings2label(vstr)
            fsize = fontsize - 2
        elif "fusedTypes" in rt_info:
            if (rt_info['fusedTypes'].startswith(type_name)):
                label = "{" + rt_info['fusedTypes'].replace(",","|") +"}"
            else:
                label = "{" + type_name + "|" + rt_info['fusedTypes'].replace(",","|") +"}"
        else:
            label = "{}".format(type_name)
            if color == "cyan":
                color = "darkolivegreen1" # node color is different for ngraph

        allinfo = "{} / #{}".format(friendly_name, nindex)

        if (execTimeMcs_by_node[n] > 0):
            allinfo += "\n----execTime of the node {:.2f}%".format(execTimeMcs_by_node[n]*100/execTimeMcs_total)
        if (attrs):
            allinfo += "\n----attributes----\n{}".format("\n".join(attrs))
        if (rt_info):
            allinfo += "\n----rt_info----\n{}".format("\n".join(rtinfo))

        #if type_name.startswith("Constant"):
        #    allinfo += "\n----values----\n{}".format(",".join(n.get_value_strings()[:32]))

        
        if type_name == "Subgraph":
            submodel = rt_info["body"]
            allinfo += "\n----model-----\n{}".format(generate_str(submodel))
            data_map[friendly_name] = submodel

        if detailed_label:
            label = allinfo.replace("\n", "\\n")
            allinfo = label.replace("\\n", "\n")
        kwargs = {"shape":'Mrecord',
              "style":'filled,rounded',
               "fillcolor":color,
               "fontsize":str(fsize),
               "margin":"0,0","width":"0","height":"0",
               "tooltip":allinfo}
        g.node(name=friendly_name,
               label=label,
               **kwargs)
        assert(friendly_name not in name2node) # make sure the name is uinque
        name2node[friendly_name] = n
        node2name[n] = friendly_name
    
    # generate color table for in-place mem
    for n in model.get_ordered_ops():
        for i in n.inputs():
            mem_rt_info = i.get_source_output().get_rt_info()
            if "Data" in mem_rt_info:
                Data = mem_rt_info["Data"]
                if not Data in data_color:
                    # single non-inplace color is black
                    data_color[Data] = "black"
                elif data_color[Data] == "black":
                    # replace in-place color with non-black
                    data_color[Data] = gen_rand_color()

    max_act_sz = 0
    for n in model.get_ordered_ops():
        for i in n.inputs():
            ps = i.get_partial_shape()
            if ps.is_dynamic:
                max_act_sz = -1
                break
            act_sz = np.prod(np.array(i.get_shape()))
            if (max_act_sz < act_sz):
                max_act_sz = act_sz
        if max_act_sz < 0:
            break

    for n in model.get_ordered_ops():
        for i in n.inputs():
            src_out = i.get_source_output()
            tail_name = name_normalize(src_out.get_node())
            head_name = name_normalize(n)

            if (len(src_out.get_target_inputs()) > 4 or len(src_out.get_node().outputs()) > 4):
                found_ki = False
                for ki, si in enumerate(src_out.get_target_inputs()):
                    if si.get_node() is n:
                        found_ki = True
                        break
                assert(found_ki)
                tail_name += ".out{}.{}".format(src_out.get_index(), ki)

            ps = i.get_partial_shape()
            if ps.is_dynamic:
                str_shape = ",".join([str(s) for s in ps])
            else:
                act_sz = np.prod(np.array(i.get_shape()))
                str_shape = ",".join([str(s) for s in i.get_shape()])
            str_ele_type = i.get_element_type().get_type_name()
            src_rt_info = i.get_source_output().get_node().get_rt_info()
            mem_rt_info = i.get_source_output().get_rt_info()

            label = '[{}]'.format(str_shape)
            layout_fmt = None
            if "Format" in mem_rt_info:
                layout_fmt = mem_rt_info["Format"]
            elif "outputLayouts" in src_rt_info:
                layout_fmt = src_rt_info["outputLayouts"]

            if layout_fmt not in ("a","ab","abc","abcd","abcde","abcdef",None):
                label += "\n" + layout_fmt

            precision = None
            if "Precision" in mem_rt_info:
                precision = mem_rt_info["Precision"]
            elif "outputPrecisions" in src_rt_info:
                precision = src_rt_info["outputPrecisions"]
            else:
                precision = str_ele_type
            
            if precision not in ("FP32","float","float32",None):
                label += "\n" + precision

            style = "solid"
            color = "black"
            if "Data" in mem_rt_info:
                Data = mem_rt_info["Data"]
                #label += "\n0x{:X}".format(Data)
                try:
                    # build a numpy array and return
                    p=ctypes.c_void_p(Data)
                    c_type = precision2ctype[mem_rt_info["Precision"]]
                    pf = ctypes.cast(p, ctypes.POINTER(c_type))
                    cnt = mem_rt_info["MaxMemSize"]//ctypes.sizeof(c_type)
                    dynamic_mem = cnt <= 0
                    if dynamic_mem:
                        style = "dashed"
                    else:
                        base_array = np.ctypeslib.as_array(pf, shape=(cnt,))
                        part_array = base_array[mem_rt_info["OffsetPadding"]:]

                    BlockDims = mem_rt_info["BlockDims"]
                    OffsetPaddingToData = mem_rt_info["OffsetPaddingToData"]
                    Strides = mem_rt_info["Strides"]

                    total_shape = np.array(BlockDims) + np.array(OffsetPaddingToData)
                    total_cnt = np.prod(total_shape)
                    if not dynamic_mem:
                        new_array = part_array[:total_cnt].reshape(total_shape)
                        nd_strides = np.array(new_array.strides)//ctypes.sizeof(c_type)
                        if (nd_strides != np.array(Strides)).any():
                            # TODO new_array = part_array.reshape(np.array(Strides))
                            label += "\n(strided)"

                    color = data_color[Data]
                    if not Data in data_map:
                        data_map[Data] = []
                    if not dynamic_mem:
                        data_map[Data].append(new_array)
                except Exception as e:
                    print("edge '{}->{}' with Data but failed to parse:\n{}".format(
                        tail_name, head_name, e
                    ))
                    raise e

            labeltooltip = []
            for k,v in mem_rt_info.items():
                if k == "Data" or k == "Ptr":
                    value = "0x{:X}".format(mem_rt_info[k])
                else:
                    value = str(v)
                labeltooltip.append("{}={}".format(k, value))
            if max_act_sz > 0:
                penwidth = act_sz*4.5/max_act_sz + 0.5
            else:
                penwidth = 1
            g.edge(
                tail_name,
                head_name,
                label=label,
                edgetooltip="{}:{}->{}:{}".format(tail_name, i.get_source_output().get_index(), head_name, i.get_index()),
                labeltooltip="\n".join(labeltooltip),
                headURL="head",
                headtooltip="headtooltip",
                tailtooltip="tailtooltip",
                color=color,
                style=style,
                penwidth = "{:.3f}".format(penwidth),
                fontsize=str(fontsize*8//10))
    return g, data_map

def visualize_model(model, fontsize=12, filename=None, detailed_label=False):
    g, data_map = generate_graph(model, fontsize, detailed_label=detailed_label)
    graph_src = Source(g.source, format="svg")
    if filename:
        svg = graph_src.pipe().decode('utf-8')
        if filename.endswith(".html"):
            output_src = dot_to_html.dot_to_html(svg)
        else:
            output_src = svg
        with open(filename,'w') as output_file:
            output_file.write(output_src)
        return
    return graph_src, data_map

def serialize_model(self, model_path):
    weight_path = model_path[:model_path.find(".xml")] + ".bin"
    pass_manager = Manager()
    pass_manager.register_pass("Serialize", model_path, weight_path)
    pass_manager.run_passes(self)
    return model_path, weight_path

ov.Model.serialize = serialize_model
ov.Model.print = print_model
ov.Model.visualize = visualize_model

def fill_tensors_with_random(input, shape):
    dtype = get_dtype(input.get_element_type())
    rand_min, rand_max = (0, 1) if dtype == np.bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    # np.random.uniform excludes high: add 1 to have it generated
    if np.dtype(dtype).kind in ['i', 'u', 'b']:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    a = rs.uniform(rand_min, rand_max, list(shape)).astype(dtype)
    return ov.Tensor(a)

def test_infer_queue(compiled_model, input_shapes, num_request, num_infer, time_limit=60):
    infer_queue = ov.AsyncInferQueue(compiled_model, num_request)

    latency_list = []
    prof_list = []
    def callback(request, userdata):
        latency_list.append(request.latency)
        prof_list.append(request.profiling_info)

    infer_queue.set_callback(callback)

    all_input = {}
    for port, input in enumerate(compiled_model.inputs):
        if port < len(input_shapes):
            static_shape = input_shapes[port]
        else:
            static_shape = input.get_shape()
        all_input[port] = fill_tensors_with_random(input, static_shape)

    for i in range(num_request):
        infer_queue.start_async(all_input, userdata=i)

    t0 = time.time()
    for i in range(num_infer):
        wtime = time.time() - t0
        if time_limit and (wtime > time_limit):
            break
        infer_queue.start_async(None, userdata=i)
    infer_queue.wait_all()
    fps = i/wtime
    return latency_list, prof_list, fps, wtime

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str,
                        help="Model file path")
    parser.add_argument("-p","--perf", action="store_true",
                        help="Enable profiling")
    parser.add_argument("--raw", action="store_true",
                        help="Dump raw model")
    parser.add_argument("--bf16", action="store_true",
                        help="Enable inference with bf16")
    parser.add_argument("--nthreads", type=int, default=4,
                        help="Set INFERENCE_NUM_THREADS for profiling")
    parser.add_argument("-r","--reshape", type=str, default="()",
                        help="Reshape raw model before visualization/profiling")
    parser.add_argument("-i","--input_shapes", type=str, default="()",
                        help="Python tuple/list of static shapes used for profiling, e.g. [[6,23,240],[5,10,2]]")
    args = parser.parse_args()

    core = ov.Core()
    model_path = args.model
    model_fname = os.path.splitext(os.path.split(model_path)[1])[0]
    model = core.read_model(model_path)

    device = "CPU"

    dev_prop = {"PERF_COUNT": "YES",
                "AFFINITY": "CORE",
                "INFERENCE_PRECISION_HINT": "bf16" if args.bf16 else "f32",
                "NUM_STREAMS" : 1,
                "INFERENCE_NUM_THREADS" : args.nthreads}

    reshape_target = {i:s for i,s in enumerate(eval(args.reshape))}
    if len(reshape_target) > 0:
        model.reshape(reshape_target)
        dest_file = "{}_reshaped.xml".format(model_fname)
        print("reshape {} using {}".format(model_fname, reshape_target))
        model.serialize(dest_file)
        print("{} is saved!".format(dest_file))

    if args.raw:
        dest_file = "visual_{}_raw.html".format(model_fname)
        print("saving {} ...".format(dest_file))
        model.visualize(filename=dest_file)
        print("{} is saved!".format(dest_file))

    print("Compiling model for {} device with properties: ".format(device))
    for k,v in dev_prop.items():
        print("\t{:>32} : {}".format(k, v))
    compiled_model = core.compile_model(model, device, dev_prop)

    input_shapes = eval(args.input_shapes)
    print("Inputs of the model:")
    for port, _input in enumerate(compiled_model.inputs):
        print("\t[{}] {}".format(port, _input))
        if (port < len(input_shapes)):
            print("\t\t static input shape:{}".format(input_shapes[port]))
    print("Outputs of the model:")
    for port, _output in enumerate(compiled_model.outputs):
        print("\t[{}] {}".format(port, _output))

    if args.perf:
        latency_list, prof_list, fps, wtime = test_infer_queue(compiled_model, input_shapes, 2, 20000, time_limit=10)
        print(f"test_infer_queue FPS:{fps:.1f}")

    dest_file = "visual_{}_{}.html".format(model_fname, device)
    print("Saving runtime model to: {}".format(dest_file))
    compiled_model.get_runtime_model().visualize(filename=dest_file)
    print("Model is successfully saved")
