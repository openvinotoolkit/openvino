#include <openvino/runtime/core.hpp>

int main() {
//! [dynamic_shape]

// Read model
ov::Core core;
auto model = core.read_model("model.xml");

model->reshape({{ov::Dimension(-1), ov::Dimension(-1)}});

// compile model and create infer request
auto compiled_model = core.compile_model(model, "GPU");
auto infer_request = compiled_model.create_infer_request();
auto input = model->get_parameters().at(0);

// ...

// create input tensor with specific shape
ov::Tensor input_tensor(input->get_element_type(), {2, 177});

// ...

infer_request.set_tensor(input, input_tensor);
infer_request.infer();

//! [dynamic_shape]

return 0;
}
