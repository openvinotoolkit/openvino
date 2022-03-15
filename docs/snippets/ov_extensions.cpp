// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/openvino.hpp>
//! [add_extension]
#include <openvino/core/op_extension.hpp>
//! [add_extension]
//! [add_frontend_extension]
#include <openvino/frontend/extension.hpp>
//! [add_frontend_extension]

#include <identity.hpp>


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
core.add_extension(ov::frontend::OpExtension<TemplateExtension::Identity>("TemplateIdentity"));

// Register more sophisticated mapping with decomposition
core.add_extension(ov::frontend::ConversionExtension(
        "TemplateIdentity",
        [](const ov::frontend::NodeContext& context) {
            // Arbitrary decomposition code here
            return ov::OutputVector{ std::make_shared<TemplateExtension::Identity>(context.get_input(0)) };
        }));
//! [add_frontend_extension]
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
