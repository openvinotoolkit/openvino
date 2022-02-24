// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/openvino.hpp>
#include <openvino/core/op_extension.hpp>
#include <identity.hpp>

int main() {
{
//! [add_extension]
ov::Core core;
// Use operation type to add operation extension 
core.add_extension<TemplateExtension::Identity>();
// or you can add operation extension to this method
core.add_extension(ov::OpExtension<TemplateExtension::Identity>());
//! [add_extension]
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
