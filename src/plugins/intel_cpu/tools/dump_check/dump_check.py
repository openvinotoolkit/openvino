#!/usr/bin/python3

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import Core, Model, Tensor, PartialShape, Type
from openvino.runtime import opset8 as opset
from openvino.runtime.op import Constant, Parameter, tensor_iterator
from openvino.runtime.passes import Manager, Serialize
from openvino.runtime.utils.types import get_dtype
import openvino as ov
import numpy as np
import sys
import os, errno
import struct
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"

def mkdirp(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def fill_tensors_with_random(input):
    dtype = get_dtype(input.get_element_type())
    rand_min, rand_max = (0, 1) if dtype == bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    # np.random.uniform excludes high: add 1 to have it generated
    if np.dtype(dtype).kind in ['i', 'u', 'b']:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    shape = input.get_shape()
    a = rs.uniform(rand_min, rand_max, list(shape)).astype(dtype)
    return Tensor(a)

def fill_tensors_from_image(input, input_file):
    dtype = get_dtype(input.get_element_type())
    shape = input.get_shape()

    data = np.load(input_file, allow_pickle=True)
    for itm in data.files:
        print(itm)
        print(data[itm])

    return Tensor(data[data.files[0]].astype(dtype).reshape(shape))

class IEB:
    precision_table = {
        5:(np.float32, 4),
        3:(np.int16, 2),
        14:(np.uint8, 1),
        8:(np.int8, 1),
        10:(np.int32, 4),
        15:(np.uint32, 4),
        11:(np.int64, 8),
        17:(np.uint64, 8)
    }

    @classmethod
    def dump(cls, ieb_file, nparray):
        # b'IEB0', 256, 10, 4, 1, 32, 1104, 1104, 0, 0, 0, 255, 0, 0, 0, 72, 156008448, 0, 0
        fmt = "@4sHBB7IB3BLLLL"

        magic, ver = b'IEB0', 256
        
        precision = -1
        for k,v in IEB.precision_table.items():
            if (v[0] == nparray.dtype):
                precision = k
        
        assert(precision >= 0)

        ndims = len(nparray.shape)
        dims = [0 for _ in range(7)]
        for i, s in enumerate(nparray.shape):
            dims[i] = s
        scaling_axis = 255
        reserved = [0,0,0]
        data_offset = struct.calcsize(fmt)
        data_size = np.prod(nparray.shape) * nparray.itemsize
        scaling_data_offset = 0
        scaling_data_size = 0
        header = struct.pack(fmt, magic, ver, precision, ndims,
                           dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6],
                           scaling_axis, reserved[0], reserved[1], reserved[2],
                           data_offset, data_size, scaling_data_offset, scaling_data_size)
        
        with open(ieb_file,"wb") as f:
            f.write(header)
            f.write(nparray.tobytes())
        return

    def __init__(self, ieb_file) -> None:
        with open(ieb_file,"rb") as f:
            data = f.read() # bytes
            header = struct.unpack_from("@4sHBB7IB3BLLLL", data, offset=0)
            # print(header, len(header))
            (self.magic, self.ver, self.precision, self.ndims,
            self.dims0, self.dims1, self.dims2, self.dims3, self.dims4, self.dims5, self.dims6,
            self.scaling_axis,
            self.reserved0, self.reserved1, self.reserved2,
            self.data_offset, self.data_size, self.scaling_data_offset, self.scaling_data_size) = header

            (dtype, type_size, ) = IEB.precision_table[self.precision]
            count = self.data_size//type_size
            
            # recover the data as numpy array
            self.dims = np.array([self.dims0, self.dims1, self.dims2, self.dims3, self.dims4, self.dims5, self.dims6])
            self.dims = self.dims[0:self.ndims]
            self.value = np.frombuffer(data, dtype = dtype, count=count, offset=self.data_offset)
            dims = self.dims
            # bf16 blob is parsed with numpy with int16. Append 0 in lower/higer 16 bit 0 on little/big endian then view with float32 type.
            if (dtype == np.int16):
                zero_array=np.zeros(self.value.shape, dtype=dtype)
                if (sys.byteorder == "little"):
                    self.value=np.dstack((zero_array, self.value)).flatten()
                else:
                    self.value=np.dstack((self.value, zero_array)).flatten()
                self.value=self.value.view(dtype=np.float32)
            self.value = np.reshape(self.value, dims)

            # self.values = struct.unpack_from(f"@{count}{stype}", data, offset=self.data_offset)
            # print(self.values.shape, self.values.dtype)
        pass

class DumpIndex:
    def __init__(self, args) -> None:
        (self.ExecIndex, self.Name, self.OriginalLayers, self.tag, self.itag, self.ieb_file) = args


def dump_tensors(core, model, dump_dir = "./cpu_dump", dump_ports="OUT", device_target="CPU", infer_bf16=False, filter_type=""):
    os.environ["OV_CPU_BLOB_DUMP_DIR"] = dump_dir
    os.environ["OV_CPU_BLOB_DUMP_FORMAT"] = "BIN"
    os.environ["OV_CPU_BLOB_DUMP_NODE_PORTS"] = dump_ports
    if filter_type != "":
        os.environ["OV_CPU_BLOB_DUMP_NODE_TYPE"]  = filter_type
    mkdirp(dump_dir)

    device_config = {"PERF_COUNT": "NO",
                "AFFINITY": "CORE",
                "PERFORMANCE_HINT_NUM_REQUESTS":0,
                "PERFORMANCE_HINT":"LATENCY",
                "INFERENCE_PRECISION_HINT": "f32",
                "NUM_STREAMS":1,
                "INFERENCE_NUM_THREADS":1}
    if infer_bf16 == True:
        device_config["INFERENCE_PRECISION_HINT"] = "bf16"
    print("compiling model with {}".format(device_config))
    exec_net = core.compile_model(model, device_target, device_config)
    req = exec_net.create_infer_request()

    print("fill input with random data:")
    inputs={}
    for i in exec_net.inputs:
        inputs[i] = fill_tensors_with_random(i)
        print(f"  {i}")

    print("infer with dump..")
    
    result = req.infer(inputs)

    # dump result as ieb, so even no dump_ports, you can still know
    # final correctness
    print("Dump result as ieb...")
    result_exec_id = 999900
    for out, value in result.items():
        names = [name.replace(":","_").replace("/","_") for name in out.names]
        names.sort()
        ieb_name = os.path.join(dump_dir, "#{}_{}.ieb".format(result_exec_id, "~".join(names)))
        print("  {}..".format(ieb_name))
        IEB.dump(ieb_name, value)
        result_exec_id += 1

    runtime_func = exec_net.get_runtime_model()
    base_name = dump_dir.split('/')
    base_name = base_name[-1].split('\\')
    xml_path = f"{base_name[-1]}.xml"
    bin_path = f"{base_name[-1]}.bin"
    pass_manager = Manager()
    pass_manager.register_pass(Serialize(path_to_xml=xml_path, path_to_bin=bin_path))
    pass_manager.run_passes(runtime_func)
    
    print(f"{device_target} Runtime model (exec_graph) is serialized to {xml_path}.")


def visualize_diff_abs(diff_abs):
    vis_abs = diff_abs
    cur_shape = diff_abs.shape
    if len(vis_abs.shape) > 3:
        vis_abs = vis_abs.reshape(-1,cur_shape[-2],cur_shape[-1])
    
    fig, ax = plt.subplots()

    # first channel with diff
    for cur_channel in range(0, vis_abs.shape[0]):
        diff_img = vis_abs[cur_channel,:,:]
        if np.amax(diff_img) > 1e-8:
            break

    im = ax.imshow(vis_abs[cur_channel,:,:])

    def update_channel(val):
        nonlocal cur_channel
        val = int(val)
        cur_channel = val
        diff_img = vis_abs[val,:,:]
        max_diff = np.amax(diff_img)
        ax.set_title(" channel:{}  shape:{}  Max diff: {:.8f}".format(
                        val, diff_img.shape, np.amax(diff_img)))
        # normalize intensity
        im.set_data(diff_img * 255 / max_diff)
        fig.canvas.draw_idle()

    update_channel(cur_channel)

    ax_ch_slider = plt.axes([0.1, 0.25, 0.0225, 0.63])
    ch_slider = Slider(
        ax=ax_ch_slider,
        label="Channels",
        valmin=0,
        valmax=vis_abs.shape[0],
        valinit=0,
        valstep=1,
        orientation="vertical"
    )

    ch_slider.on_changed(update_channel)

    def on_press(event):
        # print('press', event.key, 'cur_channel', cur_channel)
        sys.stdout.flush()
        if event.key == 'escape':
            print("escape key detected, exit.")
            sys.exit(1)
        if event.key == 'up':
            for c in range(cur_channel+1, vis_abs.shape[0]):
                diff_img = vis_abs[c,:,:]
                if np.amax(diff_img) > 1e-8:
                    ch_slider.set_val(c)
                    break
        if event.key == 'down':
            for c in range(cur_channel-1, -1, -1):
                diff_img = vis_abs[c,:,:]
                if np.amax(diff_img) > 1e-8:
                    ch_slider.set_val(c)
                    break
    fig.canvas.mpl_connect('key_press_event', on_press)

    plt.show()

def compare_dumps(model, atol, rtol, visualize, dump_dir1, dump_dir2):

    output_tensors = []
    for out in model.outputs:
        for oname in out.get_names():
            output_tensors.append(oname.split(":")[0])

    def is_output(name):
        for tag in output_tensors:
            if tag in name:
                return True
        return False

    def get_sorted_ied_list(dir):
        iebs = []
        for file_name in os.listdir(dir):
            if file_name.endswith(".ieb"):
                k = file_name.find("_")
                id = int(file_name[1:k])
                name = file_name[k:]
                iebs.append((id, name, file_name))
        return sorted(iebs, key=lambda item:item[0])

    ieb_list1 = get_sorted_ied_list(dump_dir1)
    ieb_list2 = get_sorted_ied_list(dump_dir2)

    def get_match_ieb_file2(f1):
        for f2 in ieb_list2:
            if f1[1] == f2[1]:
                return f2
        return None

    MAX_atol = {}
    for f1 in ieb_list1:
        f2 = get_match_ieb_file2(f1)
        if not f2:
            print("{}[  SKIPPED   ]: not found {} in {} {}".format(Colors.YELLOW, f1[-1], dump_dir2, Colors.END))
            continue
        
        ieb_file1 = f1[-1]
        ieb_file2 = f2[-1]
        # compare 
        ieb1 = IEB(os.path.join(dump_dir1, ieb_file1))
        ieb2 = IEB(os.path.join(dump_dir2, ieb_file2))

        if "Input_Constant" in ieb_file1 and "Input_Constant" in ieb_file2:
            print("Skipped Input_Constant {ieb_file1} vs {ieb_file2}")
            continue

        if not np.allclose(ieb1.value, ieb2.value, atol=atol, rtol=rtol):
            diff_abs = np.abs(ieb1.value.astype('float32') - ieb2.value.astype('float32'))
            thresh = atol + rtol * np.abs(ieb2.value)
            idx = np.where(diff_abs >= thresh)
            atol_max = np.amax(diff_abs[idx])

            if ieb1.value.dtype in MAX_atol:
                if MAX_atol[ieb1.value.dtype] < atol_max:
                    MAX_atol[ieb1.value.dtype] = atol_max
            else:
                MAX_atol[ieb1.value.dtype] = 0

            prefixERR = Colors.RED
            if is_output(f1[-1]):
                prefixERR += Colors.UNDERLINE
            print("{}[  FAILED ]: {} {} {}".format(prefixERR, f1[-1], f2[-1], Colors.END))
            info  = ""
            if (np.prod(diff_abs.shape) < 8):
                info = "{} vs {}".format(ieb1.value.reshape(-1), ieb2.value.reshape(-1))
            
            max_abs = np.amax(diff_abs[idx])
            max_idx = np.where(diff_abs[idx] >= max_abs)
            max_org = np.abs(ieb2.value)[idx][max_idx]
            print("  {} {}  ({:.2e} ~ {:.2e}/{:.2e}={:.2e})  @ mean:{:.2e} std:{:.2e} detail: {}".format(
                    diff_abs.shape, diff_abs.dtype,
                    np.amin(diff_abs[idx]), max_abs,
                    max_org[0], max_abs / (max_org[0] + 0.000001),
                    np.mean(diff_abs[idx]), np.std(diff_abs[idx]), info))

            if (visualize):
                visualize_diff_abs(diff_abs)
        else:
            print("{}[  OK     ]: {} {} {}".format(Colors.GREEN, f1[-1], f2[-1], Colors.END))
            pass

    print("============================================")
    if (len(MAX_atol) == 0):
        print("Pass")
    else:
        for prec in MAX_atol:
            print("Max atol {} : {}".format(prec, MAX_atol[prec]))

def compare_dump_file(ieb_file1, ieb_file2, visualize):
    ieb1 = IEB(ieb_file1)
    ieb2 = IEB(ieb_file2)

    if ieb1.value.shape != ieb2.value.shape :
        print(" Shape mismatch {} != {} , will compare in flatten.".format(ieb1.value.shape, ieb2.value.shape))
        diff_abs = np.abs(ieb1.value.reshape(-1) - ieb2.value.reshape(-1))
    else:
        diff_abs = np.abs(ieb1.value - ieb2.value)

    if not np.all(diff_abs.shape):
        print(" Shape{} has dim 0".format(ieb1.shape))
        return

    max_abs = np.amax(diff_abs)
    max_idx = np.where(diff_abs >= max_abs)
    max_org = np.abs(ieb2.value)[max_idx]
    print("  {} {}  ({:.2e} ~ {:.2e}/{:.2e}={:.2e})  @ mean:{:.2e} std:{:.2e} ".format(
            diff_abs.shape, diff_abs.dtype,
            np.amin(diff_abs), max_abs,
            max_org[0], max_abs / (max_org[0] + 0.00001),
            np.mean(diff_abs), np.std(diff_abs)))

    if (visualize):
        visualize_diff_abs(diff_abs)

def main():
    parser = argparse.ArgumentParser("cpu_cross_check")
    parser.add_argument("-m", type=str, default="", required=True, help="Model file path")
    parser.add_argument("-atol", type=float, default=1e-8, help="absolute error")
    parser.add_argument("-rtol", type=float, default=1e-4, help="relative error")
    parser.add_argument("-v", action="store_true", help="visualize error")
    parser.add_argument("-p", "--ports", type=str, default="OUT", help="dump ports: OUT | ALL")
    parser.add_argument("dumps", type=str, default="", nargs="+", help="dump folders or files")
    parser.add_argument("-bf16", help="Enables infer with BF16 precision", action='store_true')
    parser.add_argument("-f", "--filter_op", type=str, default="", help="op type filter: Convolution | ConvolutionBackpropData")
    args = parser.parse_args()

    print(f"Read model {args.m}...")
    core = Core()
    model = core.read_model(args.m)

    if len(args.dumps) == 1:
        dump_tensors(core, model, args.dumps[0],  args.ports, "CPU", args.bf16, args.filter_op)
    else:
        assert(len(args.dumps) == 2)
        if (os.path.isdir(args.dumps[0])):
            compare_dumps(model, args.atol, args.rtol, args.v, args.dumps[0], args.dumps[1])
        else:
            compare_dump_file(args.dumps[0], args.dumps[1], args.v)

if __name__ == "__main__":
    main()
