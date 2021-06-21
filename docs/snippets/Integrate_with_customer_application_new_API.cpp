#include <ie_core.hpp>

int main() {
const std::string output_name = "output_name";
const std::string input_name = "input_name";
//! [part0]
InferenceEngine::Core core;
InferenceEngine::CNNNetwork network;
InferenceEngine::ExecutableNetwork executable_network;
//! [part0]

//! [part1]
network = core.ReadNetwork("Model.xml");
//! [part1]

//! [part2]
network = core.ReadNetwork("model.onnx");
//! [part2]

//! [part3]
/** Take information about all topology inputs **/
InferenceEngine::InputsDataMap input_info = network.getInputsInfo();
/** Take information about all topology outputs **/
InferenceEngine::OutputsDataMap output_info = network.getOutputsInfo();
//! [part3]

//! [part4]
/** Iterate over all input info**/
for (auto &item : input_info) {
    auto input_data = item.second;
    input_data->setPrecision(InferenceEngine::Precision::U8);
    input_data->setLayout(InferenceEngine::Layout::NCHW);
    input_data->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
    input_data->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);
}
/** Iterate over all output info**/
for (auto &item : output_info) {
    auto output_data = item.second;
    output_data->setPrecision(InferenceEngine::Precision::FP32);
    output_data->setLayout(InferenceEngine::Layout::NC);
}
//! [part4]

//! [part5]
executable_network = core.LoadNetwork(network, "CPU");
//! [part5]

//! [part6]
/** Optional config. E.g. this enables profiling of performance counters. **/
std::map<std::string, std::string> config = {{ InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES }};
executable_network = core.LoadNetwork(network, "CPU", config);
//! [part6]

//! [part7]
auto infer_request = executable_network.CreateInferRequest();
//! [part7]

auto infer_request1 = executable_network.CreateInferRequest();
auto infer_request2 = executable_network.CreateInferRequest();

//! [part8]
/** Iterate over all input blobs **/
for (auto & item : input_info) {
    auto input_name = item.first;
    /** Get input blob **/
    auto input = infer_request.GetBlob(input_name);
    /** Fill input tensor with planes. First b channel, then g and r channels **/
//     ...
}
//! [part8]

//! [part9]
auto output = infer_request1.GetBlob(output_name);
infer_request2.SetBlob(input_name, output);
//! [part9]

//! [part10]
/** inputBlob points to input of a previous network and
    cropROI contains coordinates of output bounding box **/
InferenceEngine::Blob::Ptr inputBlob;
InferenceEngine::ROI cropRoi;
//...

/** roiBlob uses shared memory of inputBlob and describes cropROI
    according to its coordinates **/
auto roiBlob = InferenceEngine::make_shared_blob(inputBlob, cropRoi);
infer_request2.SetBlob(input_name, roiBlob);
//! [part10]

//! [part11]
/** Iterate over all input blobs **/
for (auto & item : input_info) {
    auto input_data = item.second;
    /** Create input blob **/
    InferenceEngine::TBlob<unsigned char>::Ptr input;
    // assuming input precision was asked to be U8 in prev step
    input = InferenceEngine::make_shared_blob<unsigned char>(
        InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, input_data->getTensorDesc().getDims(),
        input_data->getTensorDesc().getLayout()));
    input->allocate();
    infer_request.SetBlob(item.first, input);

    /** Fill input tensor with planes. First b channel, then g and r channels **/
//     ...
}
//! [part11]

//! [part12]
infer_request.StartAsync();
infer_request.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
//! [part12]

auto sync_infer_request = executable_network.CreateInferRequest();

//! [part13]
sync_infer_request.Infer();
//! [part13]

//! [part14]
    for (auto &item : output_info) {
        auto output_name = item.first;
        auto output = infer_request.GetBlob(output_name);
        {
            auto const memLocker = output->cbuffer(); // use const memory locker
            // output_buffer is valid as long as the lifetime of memLocker
            const float *output_buffer = memLocker.as<const float *>();
            /** output_buffer[] - accessing output blob data **/
        }
    }
//! [part14]

return 0;
}
