// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/openvino.hpp>
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
#include <openvino/opsets/opset8.hpp>
//! [frontend_extension_ThresholdedReLU_header]

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

//! [frontend_extension_CustomOperation_rename]
core.add_extension(ov::frontend::OpExtension<CustomOperation>(
    { {"attr1", "fw_attr1"}, {"attr2", "fw_attr2"} },
    {}
));
//! [frontend_extension_CustomOperation_rename]

//! [frontend_extension_CustomOperation_rename_set]
core.add_extension(ov::frontend::OpExtension<CustomOperation>(
    { {"attr1", "fw_attr1"} },
    { {"attr2", 5} }
));
//! [frontend_extension_CustomOperation_rename_set]

//! [frontend_extension_ThresholdedReLU]
core.add_extension(ov::frontend::ConversionExtension(
    "ThresholdedReLU",
    [](const ov::frontend::NodeContext& node) {
        auto greater = std::make_shared<ov::opset8::Greater>(
            node.get_input(0),
            ov::opset8::Constant::create(ov::element::f32, {}, {node.get_attribute<float>("alpha")}));
        auto casted = std::make_shared<ov::opset8::Convert>(greater, ov::element::f32);
        return ov::OutputVector{ std::make_shared<ov::opset8::Multiply>(node.get_input(0), casted) };
    }));
//! [frontend_extension_ThresholdedReLU]
}
{
//! [add_extension_lib]
ov::Core core;
// Load extensions library to ov::Core
core.add_extension("openvino_template_extension.so");
//! [add_extension_lib]
}
return 0;
}
