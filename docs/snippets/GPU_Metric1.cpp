#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

int main() {
//! [part1]
ov::Core core;
std::shared_ptr<ov::Model> network = core.read_model("sample.xml");
uint32_t n_streams = 2;
int64_t available_device_mem_size = 3221225472;
std::map<std::string, ov::Any> options = {{ov::hint::model_ptr.name(), network}}; // Required. Set the address of the target network. If this is not set, the MAX_BATCH_SIZE returns 1.
options.insert(std::make_pair(ov::streams::num.name(), n_streams)); // Optional. Set only when you want to estimate max batch size for a specific throughtput streams. Default is 1 or throughtput streams set by SetConfig.
options.insert(std::make_pair(ov::intel_gpu::hint::available_device_mem.name(), available_device_mem_size)); // Optional. Set only when you want to limit the available device mem size.

uint32_t max_batch_size = core.get_property("GPU", ov::max_batch_size.name(), options);
//! [part1]
//! [part2]
// This is not entirely GPU-specific metric (so METRIC_KEY is used rather than GPU_METRIC_KEY below),
// but the GPU is the only device that supports that at the moment.
// For the GPU, the metric already accommodates limitation for the on-device memory that the MAX_BATCH_SIZE poses.
// so OPTIMAL_BATCH_SIZE is always less than MAX_BATCH_SIZE. Unlike the latter it is also aligned to the power of 2.
uint32_t optimal_batch_size = core.get_property("GPU", ov::optimal_batch_size.name(), options);
//! [part2]
}
