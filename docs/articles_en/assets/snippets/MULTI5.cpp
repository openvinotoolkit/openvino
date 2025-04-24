#include <openvino/openvino.hpp>

int main() {
//! [part5]
ov::Core core;

// // Read a model and compile it on MULTI
ov::CompiledModel compileModel = core.compile_model("sample.xml", "MULTI:GPU,CPU");

// query the optimal number of requests
uint32_t nireq = compileModel.get_property(ov::optimal_number_of_infer_requests);
//! [part5]
return 0;
}
