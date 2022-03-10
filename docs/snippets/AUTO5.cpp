#include <openvino/openvino.hpp>

int main() {
//! [part5]
	ov::Core core;

	// Read a network in IR, PaddlePaddle, or ONNX format:
	std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

	// Configure the VPUX and the Myriad devices separately and load the network to Auto-Device plugin:
	// set VPU config
	core.set_property("VPUX", {});
	// set MYRIAD config
	core.set_property("MYRIAD", {});
	ov::CompiledModel compiled_model = core.compile_model(model, "AUTO");

//! [part5]
    return 0;
}
