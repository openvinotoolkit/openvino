#include <openvino/openvino.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model("sample.xml");
    //NEW IE-CENTRIC API, the "MULTI" plugin is (globally) pre-configured with the explicit option:
    core.set_property("MULTI", ov::device::priorities("HDDL,GPU"));
    ov::CompiledModel compileModel0 = core.compile_model(model, "MULTI");

    //NEW IE-CENTRIC API, configuration of the "MULTI" is part of the network configuration (and hence specific to the network):
    ov::CompiledModel compileModel1 = core.compile_model(model, "MULTI", ov::device::priorities("HDDL,GPU"));
    //NEW IE-CENTRIC API, same as previous, but configuration of the "MULTI" is part of the name (so config is empty), also network-specific:
    ov::CompiledModel compileModel2 = core.compile_model(model, "MULTI:HDDL,GPU");
//! [part0]
return 0;
}
