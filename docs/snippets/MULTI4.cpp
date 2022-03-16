#include <openvino/openvino.hpp>

int main() {
ov::AnyMap hddl_config  = {{ov::enable_profiling(true)}};
ov::AnyMap gpu_config = {{ov::enable_profiling(true)}};
//! [part4]
// configure the HDDL device first
ov::Core core;
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");
core.set_property({ov::device::properties("HDDL", hddl_config),
    ov::device::properties("GPU", gpu_config)});

// load the network to the multi-device, while specifying the configuration (devices along with priorities):
ov::CompiledModel compileModel = core.compile_model(model, "MULTI", ov::device::priorities("HDDL,GPU"));
// new property allows to query the optimal number of requests:
uint32_t nireq = compileModel.get_property(ov::optimal_number_of_infer_requests);
//! [part4]
return 0;
}
