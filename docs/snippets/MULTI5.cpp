#include <openvino/openvino.hpp>

int main() {
std::string device_name = "MULTI:HDDL,GPU";
const ov::AnyMap full_config = {};
//! [part5]
ov::Core core;
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");
// 'device_name' can be "MULTI:HDDL,GPU" to configure the multi-device to use HDDL and GPU
ov::CompiledModel compileModel = core.compile_model(model, device_name, full_config);
// new property allows to query the optimal number of requests:
uint32_t nireq = compileModel.get_property(ov::optimal_number_of_infer_requests);
//! [part5]
return 0;
}
