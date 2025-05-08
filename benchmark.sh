/home/REPO/openvino/openvino/bin/intel64/Release/benchmark_app -m /home/REPO/MODELS/wav2vec/i8/model.xml -shape "[1,16000]" -d GPU -hint latency -api sync -nireq 1 -niter 10 -pcsort simple_sort

# OV_GPU_LOG_TO_FILE=log.txt OV_VERBOSE=1
# -i "/home/REPO/MODELS/gridsample/input"
# -shape "[1,640,480,3]"