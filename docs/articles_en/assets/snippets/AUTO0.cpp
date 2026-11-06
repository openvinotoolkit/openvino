// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//! [py_ov_property_import_header]
#include <openvino/openvino.hpp>
#include <openvino/runtime/auto/properties.hpp>
//! [py_ov_property_import_header]

int main() {
{
//! [part0]
ov::Core core;

// Read a network in IR, PaddlePaddle, or ONNX format:
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

// compile a model on AUTO using the default list of device candidates.
// The following lines are equivalent:
ov::CompiledModel model0 = core.compile_model(model);
ov::CompiledModel model1 = core.compile_model(model, "AUTO");

// Optional
// You can also specify the devices to be used by AUTO.
// The following lines are equivalent:
ov::CompiledModel model3 = core.compile_model(model, "AUTO:GPU,CPU");
ov::CompiledModel model4 = core.compile_model(model, "AUTO", ov::device::priorities("GPU,CPU"));

//Optional
// the AUTO plugin is pre-configured (globally) with the explicit option:
core.set_property("AUTO", ov::device::priorities("GPU,CPU"));
//! [part0]
}
{
//! [compile_for_all]
ov::Core core;
auto model = core.read_model("model.xml");
auto compiled_model = core.compile_model(model,
                                         "AUTO",
                                         ov::device::priorities("GPU,CPU"),
                                         ov::cache_dir("model_cache"),
                                         ov::intel_auto::compile_for_all(true));
//! [compile_for_all]
}
    return 0;
}
