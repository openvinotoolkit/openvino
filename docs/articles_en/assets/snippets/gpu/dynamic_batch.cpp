#include <openvino/runtime/core.hpp>

int main() {
size_t C = 3;
size_t H = 224;
size_t W = 224;

//! [dynamic_batch]

// Read model
ov::Core core;
auto model = core.read_model("model.xml");

model->reshape({{ov::Dimension(1, 10), ov::Dimension(C), ov::Dimension(H), ov::Dimension(W)}});  // {1..10, C, H, W}

// compile model and create infer request
auto compiled_model = core.compile_model(model, "GPU");
auto infer_request = compiled_model.create_infer_request();
auto input = model->get_parameters().at(0);

// ...

// create input tensor with specific batch size
ov::Tensor input_tensor(input->get_element_type(), {2, C, H, W});

// ...

infer_request.set_tensor(input, input_tensor);
infer_request.infer();

//! [dynamic_batch]

return 0;
}
