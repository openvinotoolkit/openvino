#include <openvino/runtime/core.hpp>


int main() {
    {
        //! [undefined_shape]
        // Read model
        ov::Core core;
        auto model = core.read_model("model.xml");

        model->reshape({{-1, -1, -1, -1}});  // {?, ?, ?, ?}

        // compile model and create infer request
        auto compiled_model = core.compile_model(model, "GPU");
        auto infer_request = compiled_model.create_infer_request();
        auto input = model->get_parameters().at(0);

        // ...

        // create input tensor with specific batch size
        ov::Tensor input_tensor(input->get_element_type(), {10, 20, 30, 40});

        // ...

        infer_request.set_tensor(input, input_tensor);
        infer_request.infer();
        //! [undefined_shape]
    }

    {
        //! [defined_upper_bound]
        ov::Core core;
        auto model = core.read_model("model.xml");

        model->reshape({{ov::Dimension(1, 10), ov::Dimension(1, 20), ov::Dimension(1, 30), ov::Dimension(1, 40)}});
        //! [defined_upper_bound]
    }
    return 0;
}
