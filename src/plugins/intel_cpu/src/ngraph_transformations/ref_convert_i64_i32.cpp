// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "ref_convert_i64_i32.hpp"
#include <openvino/opsets/opset10.hpp>
#include "transformations/utils/utils.hpp"
#include "cpu_types.h"


bool ov::intel_cpu::RefConvertI64ToI32::run_on_model(const std::shared_ptr<ov::Model> &model) {
    const auto orderedOps = model->get_ordered_ops();
    for (const auto& op : orderedOps) {
        if (TypeFromName(op->get_type_name()) == Type::Unknown) {
            for (auto& output : op->outputs()) {
                if (output.get_element_type() == ov::element::i64 || output.get_element_type() == ov::element::u64) {
                    auto targetInputs = output.get_target_inputs();
                    auto convert = std::make_shared<ov::opset10::Convert>(output, ov::element::i32);

                    auto& rt_info = convert->get_rt_info();
                    rt_info["convert_i64_i32"] = "";
                    for (const auto& targetInput : targetInputs) {
                        targetInput.replace_source_output(convert);
                    }

                    auto& convertTensor = convert->output(0).get_tensor();
                    const auto newName = ov::op::util::get_ie_output_name(output);
                    if (ov::descriptor::get_ov_tensor_legacy_name(convertTensor).empty()) {
                        ov::descriptor::set_ov_tensor_legacy_name(convertTensor, newName);
                    }
                    if (!output.get_names().empty()) {
                        convertTensor.set_names(output.get_names());
                    }
                }
            }
        }
    }

    return true;
}
