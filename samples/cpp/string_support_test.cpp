#include <openvino/openvino.hpp>
#include <vector>
#include <string>



#include <openvino/op/op.hpp>


class StringTensorOp : public ov::op::Op {
public:
    OPENVINO_OP("StringTensorOp");

    StringTensorOp () = default;

    StringTensorOp(ov::OutputVector inputs)
        : ov::op::Op(inputs) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // The conditions below is a part of CPU hack to support a wrapped string tensor as a u8 tensor with a pointer
        if(get_input_size() == 1 && get_input_element_type(0) == ov::element::u8) {
            std::cerr << "[ DEBUG ] StringTensorOp::validate_and_infer_types detected CPU hack at input, still providing element::string in the output\n";
        }
        if(get_output_size() == 1 && get_output_element_type(0) == ov::element::u8) {
            std::cerr << "[ DEBUG ] StringTensorOp::validate_and_infer_types detected CPU hack at output, not overriding output type\n";
            return;
        }
        // Default behviour without applied hack at the output
        set_output_type(0, ov::element::string, get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        auto result = std::make_shared<StringTensorOp>(inputs);
        return result;
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    bool has_evaluate() const {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
        std::cerr << "Input raw tensor type: " << inputs[0].get_element_type() << " and shape " << inputs[0].get_shape() << "\n";

        auto pitensor = &inputs[0];

        if(pitensor->get_element_type() == ov::element::u8 && pitensor->get_byte_size() == sizeof(void*)) {
            std::cerr << "[ DEBUG ] StringTensorOp::evaluate detected CPU hack at input, unwrapping\n";
            auto data = *reinterpret_cast<const void* const*>(pitensor->data());
            if(data != nullptr) {
                pitensor = reinterpret_cast<const ov::Tensor*>(data);
            }
        }

        auto input_tensor = *pitensor;

        std::cerr << "Input tensor type after optional unwrapping: " << input_tensor.get_element_type() << " and shape " << input_tensor.get_shape() << "\n";
        std::cerr << "Output raw tensor type: " << outputs[0].get_element_type() << " and shape " << outputs[0].get_shape() << "\n";

        auto potensor = &outputs[0];

        if(potensor->get_element_type() == ov::element::u8 && potensor->get_byte_size() == sizeof(void*)) {
            std::cerr << "[ DEBUG ] StringTensorOp::evaluate detected CPU hack at output, unwrapping\n";
            auto data = *reinterpret_cast<void* const*>(potensor->data());
            if(data != nullptr) {
                potensor = reinterpret_cast<ov::Tensor*>(data);
            }
        }

        auto output_tensor = *potensor;

        std::cerr << "Output tensor type after optional unwrapping: " << output_tensor.get_element_type() << " and shape " << output_tensor.get_shape() << "\n";
        output_tensor.set_shape(input_tensor.get_shape());
        std::cerr << "set_shaped output tensor to " << output_tensor.get_shape() << "\n";

        for(size_t i = 0; i < input_tensor.get_size(); ++i) {
            std::cerr << "Produced element: " << (output_tensor.data<std::string>()[i] = "StringTensorOp(" + input_tensor.data<std::string>()[i] + ")") << "\n";
        }

        return true;
    }
};


void print_tensor_string_values (ov::Tensor tensor) {
    auto data = tensor.data<std::string>();

    for (size_t i = 0; i < tensor.get_size(); ++i) {
        std::cout << '"' << data[i] << "\"\n";
    }
}

ov::Tensor init_tensor () {
    static std::vector<std::string> strings = {"one", "two", "three", "four", "five", "six"};
    return ov::Tensor(ov::element::string, ov::Shape{2, 3}, &strings[0]);
}

void test_tensor () {
    auto tensor = init_tensor();
    std::cout << "--------\n";
    print_tensor_string_values(tensor);
    tensor.data<std::string>()[3] = "new string at 4th position";
    std::cout << "--------\n";
    print_tensor_string_values(tensor);
    tensor.set_shape(ov::Shape{2, 1});
    std::cout << "--------\n";
    print_tensor_string_values(tensor);
}

std::shared_ptr<ov::Model> test_model () {
    // Make a string tensor of shape [2, 3]
    std::cout << "--------\n";
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::string, ov::PartialShape{-1, -1});
    parameter->get_output_tensor(0).set_names({"input_tensor_name"});
    auto custom_op = std::make_shared<StringTensorOp>(ov::OutputVector{parameter});
    auto result = std::make_shared<ov::op::v0::Result>(custom_op);
    result->get_input_tensor(0).set_names({"output_tensor_name"});
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
    ov::save_model(model, "model.xml");

    ov::Core core;
    core.add_extension(ov::OpExtension<StringTensorOp>());
    auto model_read = core.read_model("model.xml");
    std::cout << "Parameter/Result of a model after serialization:\n";
    std::cout << model_read->input().get_element_type() << "\n";
    std::cout << model_read->output().get_element_type() << "\n";

    return model_read;
}

void test_compile_and_infer (std::shared_ptr<ov::Model> model) {
    std::cout << "--------\n";
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU");

    std::cout << "Compiled model input/output type: " << compiled.input().get_element_type() << "/" << compiled.output().get_element_type() << "\n";

    std::cout << "--------\n";
    auto request = compiled.create_infer_request();

    request.set_input_tensor(init_tensor()); // required for CPU hack, otherwise output tensor won't be correctly allocated
    auto output_tensor = request.get_output_tensor();   // trigger wrapping logic before infer request
    request.infer();

    std::cout << "Input tensor after inference:\n";
    print_tensor_string_values(init_tensor());
    std::cout << "Output tensor after inference:\n";
    print_tensor_string_values(request.get_output_tensor());
}


int main () {
    test_tensor();
    auto model = test_model();
    test_compile_and_infer(model);
}
