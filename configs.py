from typing import List, Tuple
from time import time

dir = '/home/pkrzemin/tasks/pycpp_align/'

benchmark_app_py  = dir + 'openvino/tools/benchmark_tool/benchmark_app.py '
benchmark_app_cpp = dir + 'openvino/bin/intel64/Debug/benchmark_app '
model             = dir + 'model/model.xml'
model_xanno       = dir + 'model/model_xanno.xml'
img               = dir + 'model/test.png'
bin_obj           = dir + 'stuff.bin'
json_obj          = dir + 'stuff.json'
report_folder     = dir + 'reports/'
cache_folder      = dir + "caches/"

img_shape         = '[1,?,?,3]'
data_shape        = '[1,800,800,3]'

# ======================================
helps = [1,None]
images = [img, "",None]
models = [model_xanno,model,"",None]
devices = ["CPU","GPU","",None]
hints = ["throughput","cumulative_throughput","latency","none","",None]
apis = ["async","sync","",None]
niters = ["10","1","0","",None]
nireqs = ["10","1","0","",None]
batches = ["1","10","0","",None]
times = ["10","1","0","",None]
progresses = ["True","False","",None]
shapes = [img_shape,data_shape,"",None]
data_shapes = [None,img_shape,data_shape, ""]
layouts = ["[NHWC]","[NCHW]","[NHW]","inputblablabla[NHWC]","",None]
nstreams = ["10","1","0","",None]
latency_percentiles = ["99","101","10","1","0","",None]
nthreads = ["10","10000","1","0","",None]
pins = ["YES","NO","NUMA","HYBRID_AWARE","wut","",None]
exec_graph_paths = [bin_obj, "none","",None]
inference_onlys = ["True","False","wut","",None]
report_types = ["no_counters","average_counters","detailed_counters","wut","",None]
report_folders = [report_folder,"none","",None]
dump_configs = [json_obj, "none", "", None]
infer_precisions = ["CPU:f32","none","",None]
ips = ["f32","none","",None]
ops = ["f32","none","",None]
cdirs = [cache_folder, "none", "", None]

"""
-l PATH_TO_EXTENSION, --path_to_extension PATH_TO_EXTENSION  - not used
-c PATH_TO_CLDNN_CONFIG, --path_to_cldnn_config PATH_TO_CLDNN_CONFIG
-stream_output [STREAM_OUTPUT]
-pc [PERF_COUNTS], --perf_counts [PERF_COUNTS]
-pcseq [PCSEQ], --pcseq [PCSEQ]
-load_config LOAD_CONFIG
-lfile [LOAD_FROM_FILE], --load_from_file [LOAD_FROM_FILE]
-iscale INPUT_SCALE, --input_scale INPUT_SCALE
-imean INPUT_MEAN, --input_mean INPUT_MEAN 
"""

def addArg(flag, arg):
    if arg is None:
        return ""
    else:
        return f"{flag} {arg} "

def getAllConfigs(benchmark:str) -> Tuple[List[str], int]:
    benches = [""]*100
    i = 0
    #for help_ in helps:
    for image in images:
        for model in models:
            for device in devices:
                #for hint in hints:
                    for api in apis:
                        for niter in niters:
                            for nireq in nireqs:
                                #for batch in batches:
                                    #for time in times:
                                        for progress in progresses:
                                            for shape in shapes:
                                                for data_shape in data_shapes:
                                                    for layout in layouts:
                                                        for nstream in nstreams:
                                                            for latency_percentile in latency_percentiles:
                                                                for nthread in nthreads:
                                                                    for pin in pins:
                                                                        for exec_graph_path in exec_graph_paths:
                                                                            for inference_only in inference_onlys:
                                                                                #for report_type in report_types:
                                                                                    #for report_folder in report_folders:
                                                                                        #for dump_config in dump_configs:
                                                                                            for infer_precision in infer_precisions:
                                                                                                for ip in ips:
                                                                                                    for op in ops:
                                                                                                        #for cdir in cdirs:
                                                                                                            bench = "".join([
                                                                                                                    benchmark,
                                                                                                                    #addArg("-h",help_),
                                                                                                                    addArg("-i",image),
                                                                                                                    addArg("-m",model), 
                                                                                                                    addArg("-d",device), 
                                                                                                                    #addArg("-hint",hint), 
                                                                                                                    addArg("-api",api), 
                                                                                                                    addArg("-niter",niter), 
                                                                                                                    addArg("-nireq",nireq), 
                                                                                                                    addArg("-shape",shape), 
                                                                                                                    addArg("-layout",layout), 
                                                                                                                    # addArg("-b",batch), 
                                                                                                                    # addArg("-t",time), 
                                                                                                                    # addArg("-progress",progress), 
                                                                                                                    # addArg("-data_shape",data_shape), 
                                                                                                                    # addArg("-nstreams",nstream), 
                                                                                                                    # addArg("-latency_percentile",latency_percentile), 
                                                                                                                    # addArg("-nthreads",nthread), 
                                                                                                                    # addArg("-pin",pin), 
                                                                                                                    # addArg("-exec_graph_path",exec_graph_path), 
                                                                                                                    # addArg("-inference_only",inference_only), 
                                                                                                                    # #addArg("-report_type",report_type), 
                                                                                                                    # #addArg("-report_folder",report_folder), 
                                                                                                                    # #addArg("-dump_config",dump_config), 
                                                                                                                    # addArg("-infer_precision",infer_precision), 
                                                                                                                    addArg("-ip",ip), 
                                                                                                                    addArg("-op",op),
                                                                                                                    # #addArg("-a",cdir)
                                                                                                                    ])
                                                                                                            benches[i] = bench
                                                                                                            i+=1
                                                                                                            if i % 1000000 == 0: print(i)
                                                                                                            return (benches, i)
    return (benches, i)

