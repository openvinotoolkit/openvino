#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

int main() {
//! [part0]
ov::Core core;
auto model = core.read_model("sample.xml");
auto compiledModel = core.compile_model(model, "GPU");
std::map<std::string, uint64_t> statistics_map = core.get_property("GPU", ov::intel_gpu::memory_statistics);
//! [part0]
return 0;
}
