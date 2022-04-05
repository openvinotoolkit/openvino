// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//! [include]
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gna/properties.hpp>
//! [include]

int main() {
const std::string model_path = "model.xml";
//! [ov_gna_exec_mode_hw_with_sw_fback]
ov::Core core;
auto model = core.read_model(model_path);
auto compiled_model = core.compile_model(model, "GNA",
   ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::HW_WITH_SW_FBACK));
//! [ov_gna_exec_mode_hw_with_sw_fback]
}