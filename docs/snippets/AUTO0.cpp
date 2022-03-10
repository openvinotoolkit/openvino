#include <openvino/openvino.hpp>

int main() {
//! [part0]
	/*********************
	 * With Inference Engine 2.0 API
	 *********************/
	ov::Core core;

	// Read a network in IR, PaddlePaddle, or ONNX format:
	std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

	// Load a network to AUTO using the default list of device candidates.
	// The following lines are equivalent:
	ov::CompiledModel model0 = core.compile_model(model);
	ov::CompiledModel model1 = core.compile_model(model, "AUTO");

	// You can also specify the devices to be used by AUTO in its selection process.
	// The following lines are equivalent:
	ov::CompiledModel model3 = core.compile_model(model, "AUTO:GPU,CPU");
	ov::CompiledModel model4 = core.compile_model(model, "AUTO", ov::device::priorities("GPU,CPU"));

	// the AUTO plugin is pre-configured (globally) with the explicit option:
	core.set_property("AUTO", ov::device::priorities("GPU,CPU"));
//! [part0]
	return 0;
}
