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
//! [part2]
std::map<std::string, Parameter> opt = {{"MODEL_PTR", cnnNetwork.getFunction()}}; // Required. Same usage as for the MAX_BATCH_SIZE above. If not set, the OPTIONAL_BATCH_SIZE returns 1.
// This is not entirely GPU-specific metric (so METRIC_KEY is used rather than GPU_METRIC_KEY below),
// but the GPU is the only device that supports that at the moment.
// For the GPU, the metric already accommodates limitation for the on-device memory that the MAX_BATCH_SIZE poses.
// so OPTIMAL_BATCH_SIZE is always less than MAX_BATCH_SIZE. Unlike the latter it is also aligned to the power of 2.
auto optimal_batch_size = core.GetMetric("GPU", METRIC_KEY(OPTIMAL_BATCH_SIZE), options).as<unsigned int>();
//! [part2]
}
