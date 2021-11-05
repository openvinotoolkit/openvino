#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
InferenceEngine::Core core;
CNNNetwork cnnNetwork = core.ReadNetwork(FLAGS_m);
uint32_t n_streams = 2;
int64_t available_device_mem_size = 3221225472;

std::map<std::string, Parameter> options = {{"CNN_NETWORK", &cnnNetwork}}; // Required. Set the address of the target network.
options.insert(std::make_pair("GPU_THROUGHPUT_STREAMS", n_streams)); // Optional. Set only when you want to estimate max batch size for a specific throughtput streams. Default is 1 or throughtput streams set by SetConfig.
options.insert(std::make_pair("AVAILABLE_DEVICE_MEM_SIZE", available_device_mem_size)); // Optional. Set only when you want to limit the available device mem size,

auto max_batch_size = core.GetMetric("GPU", METRIC_KEY(MAX_BATCH_SIZE), options).as<uint32_t>();
//! [part0]
}
