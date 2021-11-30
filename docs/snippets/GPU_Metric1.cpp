#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part1]
InferenceEngine::Core core;
CNNNetwork cnnNetwork = core.ReadNetwork("network.xml");
uint32_t n_streams = 2;
int64_t available_device_mem_size = 3221225472;

std::map<std::string, Parameter> options = {{"MODEL_PTR", cnnNetwork.getFunction()}}; // Required. Set the address of the target network. If this is not set, the MAX_BATCH_SIZE returns 1.
options.insert(std::make_pair("GPU_THROUGHPUT_STREAMS", n_streams)); // Optional. Set only when you want to estimate max batch size for a specific throughtput streams. Default is 1 or throughtput streams set by SetConfig.
options.insert(std::make_pair("AVAILABLE_DEVICE_MEM_SIZE", available_device_mem_size)); // Optional. Set only when you want to limit the available device mem size.

auto max_batch_size = core.GetMetric("GPU", GPU_METRIC_KEY(MAX_BATCH_SIZE), options).as<uint32_t>();
//! [part1]
}
