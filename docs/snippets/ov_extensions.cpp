// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/openvino.hpp>
#include <openvino/frontend/node_context.hpp>
#include <openvino/frontend/paddle/node_context.hpp>
//! [add_extension_header]
//#include <openvino/core/op_extension.hpp>
//! [add_extension_header]
//! [add_frontend_extension_header]
#include <openvino/frontend/extension.hpp>
//! [add_frontend_extension_header]

//! [frontend_extension_Identity_header]
#include <openvino/frontend/extension.hpp>
//! [frontend_extension_Identity_header]

//! [frontend_extension_ThresholdedReLU_header]
#include <openvino/opsets/opset11.hpp>
//! [frontend_extension_ThresholdedReLU_header]

//! [frontend_extension_framework_map_macro_headers]
#include <openvino/frontend/extension/op.hpp>
#include <openvino/frontend/onnx/extension/op.hpp>
#include <openvino/frontend/tensorflow/extension/op.hpp>
#include <openvino/frontend/paddle/extension/op.hpp>
//! [frontend_extension_framework_map_macro_headers]

#include <identity.hpp>

//! [frontend_extension_CustomOperation]
class CustomOperation : public ov::op::Op {

    std::string attr1;
    int attr2;

public:

    OPENVINO_OP("CustomOperation");

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("attr1", attr1);
        visitor.on_attribute("attr2", attr2);
        return true;
    }

    // ... implement other required methods
    //! [frontend_extension_CustomOperation]
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector&) const override { return nullptr; }
};

//! [frontend_extension_framework_map_macro_CustomOp]
class CustomOp : public ov::op::Op {
    std::string m_mode;
    int m_axis;

public:
    OPENVINO_OP("CustomOp");
    OPENVINO_FRAMEWORK_MAP(onnx, "CustomOp", { {"mode", "mode"} }, { {"axis", -1} });
    OPENVINO_FRAMEWORK_MAP(tensorflow, "CustomOpV3", { {"axis", "axis"} }, { {"mode", "linear"} });
    OPENVINO_FRAMEWORK_MAP(paddle, {"X"}, {"Out"}, "CustomOp", { {"mode", "mode"} }, { {"axis", -1} });

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("mode", m_mode);
        visitor.on_attribute("axis", m_axis);
        return true;
    }

    // ... implement other required methods
//! [frontend_extension_framework_map_macro_CustomOp]
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector&) const override { return nullptr; }
};

//! [frontend_extension_framework_map_CustomElu]
class CustomElu : public ov::op::Op {
private:
    float m_alpha;
    float m_beta;

public:
    OPENVINO_OP("CustomElu");

    CustomElu() = default;

    CustomElu(const ov::Output<ov::Node>& input, float alpha, float beta) : Op({input}), m_alpha(alpha), m_beta(beta) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("alpha", m_alpha);
        visitor.on_attribute("beta", m_beta);
        return true;
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomElu>(inputs[0], m_alpha, m_beta);
    }
};
//! [frontend_extension_framework_map_CustomElu]


int main() {
{
//! [add_extension]
ov::Core core;

// Use operation type to add operation extension
core.add_extension<TemplateExtension::Identity>();

// or you can add operation extension object which is equivalent form
core.add_extension(ov::OpExtension<TemplateExtension::Identity>());
//! [add_extension]
}
{
ov::Core core;

//! [add_frontend_extension]
// Register mapping for new frontends: FW's "TemplateIdentity" operation to TemplateExtension::Identity
core.add_extension(ov::frontend::OpExtension<TemplateExtension::Identity>("Identity"));

// Register more sophisticated mapping with decomposition
core.add_extension(ov::frontend::ConversionExtension(
    "Identity",
    [](const ov::frontend::NodeContext& context) {
        // Arbitrary decomposition code here
        // Return a vector of operation outputs
        return ov::OutputVector{ std::make_shared<TemplateExtension::Identity>(context.get_input(0)) };
    }));
//! [add_frontend_extension]
}
{
//! [frontend_extension_Identity]
auto extension1 = ov::frontend::OpExtension<TemplateExtension::Identity>("Identity");

// or even simpler if original FW type and OV type of operations match, that is "Identity"
auto extension2 = ov::frontend::OpExtension<TemplateExtension::Identity>();
//! [frontend_extension_Identity]

//! [frontend_extension_read_model]
ov::Core core;
// Add arbitrary number of extensions before calling read_model method
core.add_extension(ov::frontend::OpExtension<TemplateExtension::Identity>());
core.read_model("/path/to/model.onnx");
//! [frontend_extension_read_model]

//! [frontend_extension_MyRelu]
core.add_extension(ov::frontend::OpExtension<>("Relu", "MyRelu"));
//! [frontend_extension_MyRelu]

//! [frontend_extension_CustomOperation_as_is]
core.add_extension(ov::frontend::OpExtension<CustomOperation>());
//! [frontend_extension_CustomOperation_as_is]

//! [frontend_extension_CustomOperation_as_is_paddle]
core.add_extension(ov::frontend::OpExtension<CustomOperation>({"A", "B", "C"}, {"X", "Y"}));
//! [frontend_extension_CustomOperation_as_is_paddle]

//! [frontend_extension_CustomOperation_rename]
core.add_extension(ov::frontend::OpExtension<CustomOperation>(
    std::map<std::string, std::string>{ {"attr1", "fw_attr1"}, {"attr2", "fw_attr2"} },
    {}
));
//! [frontend_extension_CustomOperation_rename]

//! [frontend_extension_CustomOperation_rename_paddle]
core.add_extension(ov::frontend::OpExtension<CustomOperation>(
    {"A", "B", "C"},
    {"X", "Y"},
    std::map<std::string, std::string>{ {"attr1", "fw_attr1"}, {"attr2", "fw_attr2"} },
    {}
));
//! [frontend_extension_CustomOperation_rename_paddle]


//! [frontend_extension_CustomOperation_rename_set]
core.add_extension(ov::frontend::OpExtension<CustomOperation>(
    std::map<std::string, std::string>{ {"attr1", "fw_attr1"} },
    { {"attr2", 5} }
));
//! [frontend_extension_CustomOperation_rename_set]

//! [frontend_extension_CustomOperation_rename_set_paddle]
core.add_extension(ov::frontend::OpExtension<CustomOperation>(
    {"A", "B", "C"},
    {"X", "Y"},
    std::map<std::string, std::string>{ {"attr1", "fw_attr1"} },
    { {"attr2", 5} }
));
//! [frontend_extension_CustomOperation_rename_set_paddle]

{
//! [frontend_extension_framework_map_CustomElu_mapping]
auto extension = std::make_shared<ov::frontend::OpExtension<CustomElu>>("aten::elu",
                                                                        std::map<std::string, size_t>{{"alpha", 1}},
                                                                        std::map<std::string, ov::Any>{{"beta", 1.0f}});
//! [frontend_extension_framework_map_CustomElu_mapping]
}


//! [frontend_extension_ThresholdedReLU]
core.add_extension(ov::frontend::ConversionExtension(
    "ThresholdedRelu",
    [](const ov::frontend::NodeContext& node) {
        auto greater = std::make_shared<ov::opset11::Greater>(
            node.get_input(0),
            ov::opset11::Constant::create(ov::element::f32, {}, {node.get_attribute<float>("alpha")}));
        auto casted = std::make_shared<ov::opset11::Convert>(greater, ov::element::f32);
        return ov::OutputVector{ std::make_shared<ov::opset11::Multiply>(node.get_input(0), casted) };
    }));
//! [frontend_extension_ThresholdedReLU]

//! [frontend_extension_paddle_TopK]
core.add_extension(ov::frontend::ConversionExtension("top_k_v2", [](const ov::frontend::NodeContext& node) {
    auto x = node.get_input("X");
    const auto k_expected = node.get_attribute<int>("k", 1);
    auto k_expected_node = ov::opset11::Constant::create(ov::element::i32, {}, {k_expected});

    auto axis = node.get_attribute<int32_t>("axis", -1);
    bool sorted = node.get_attribute<bool>("sorted", true);
    bool largest = node.get_attribute<bool>("largest", true);

    std::string sort_type = sorted ? "value" : "none";
    std::string mode = largest ? "max" : "min";

    auto node_topk = std::make_shared<ov::opset11::TopK>(x, k_expected_node, axis, mode, sort_type);

    ov::frontend::paddle::NamedOutputs named_outputs;
    named_outputs["Out"] = ov::OutputVector{node_topk->output(0)};
    named_outputs["Indices"] = ov::OutputVector{node_topk->output(1)};

    return named_outputs;
}));
//! [frontend_extension_paddle_TopK]

//! [frontend_extension_tf_TopK]
core.add_extension(ov::frontend::ConversionExtension("TopKV2", [](const ov::frontend::NodeContext& node) {
    auto input = node.get_input(0);
    auto k_input = node.get_input(1);
    bool sorted = node.get_attribute<bool>("sorted", true);    
    auto mode = ov::opset11::TopK::Mode::MAX;
    auto sort_type = sorted ? ov::opset11::TopK::SortType::SORT_VALUES : ov::opset11::TopK::SortType::SORT_INDICES;
    auto top_k = std::make_shared<ov::opset11::TopK>(input, k_input, -1, mode, sort_type, ov::element::i32, true);
    return ov::frontend::NamedOutputVector{{"values", top_k->output(0)}, {"indices", top_k->output(1)}};
}));
//! [frontend_extension_tf_TopK]

}
{
//! [add_extension_lib]
ov::Core core;
// Load extensions library to ov::Core
core.add_extension("openvino_template_extension.so");
//! [add_extension_lib]
}

{
//! [frontend_extension_framework_map_macro_add_extension]
ov::Core core;
core.add_extension(ov::OpExtension<CustomOp>());
//! [frontend_extension_framework_map_macro_add_extension]
}
return 0;
}
