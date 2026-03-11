// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kokoro_model_transforms.hpp"

#include <string>
#include <vector>

#include "npuw/logging.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/select.hpp"

namespace ov {
namespace npuw {
namespace kokoro {

void guard_angle_divide(std::shared_ptr<ov::Model>& model) {
    // Find Divide nodes from aten::angle translation.
    // In Kokoro-82M IR: "__module.decoder.generator/aten::angle/Divide"
    std::vector<std::shared_ptr<ov::Node>> divides_to_patch;

    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::v1::Divide>(node) == nullptr) {
            continue;
        }
        const auto& name = node->get_friendly_name();
        if (name.find("aten::angle") != std::string::npos) {
            divides_to_patch.push_back(node);
        }
    }

    if (divides_to_patch.empty()) {
        LOG_VERB("guard_angle_divide: no aten::angle/Divide nodes found, skipping");
        return;
    }

    LOG_INFO("guard_angle_divide: patching " << divides_to_patch.size() << " Divide node(s)");

    for (auto& divide_node : divides_to_patch) {
        // The aten::angle decomposition produces: Divide(imag, real) -> Atan
        //   input(0) = imag  (numerator)
        //   input(1) = real  (denominator)
        //
        // When both are zero, Divide(0, 0) = NaN per IEEE 754.
        // We guard by replacing the denominator with:
        //   Select(both_zero, epsilon, real)
        // so that Divide(0, eps) = 0 and Atan(0) = 0 — the correct angle
        // for a zero-amplitude spectral bin.
        auto imag_output = divide_node->input_value(0);  // numerator
        auto real_output = divide_node->input_value(1);  // denominator

        auto elem_type = real_output.get_element_type();

        // Create scalar zero constant for comparison
        auto zero = std::make_shared<ov::op::v0::Constant>(elem_type, ov::Shape{}, 0.0f);
        zero->set_friendly_name(divide_node->get_friendly_name() + "/guard_zero");

        // Small positive value so that Divide(0, eps) ≈ 0 and Atan(0) = 0.
        constexpr float eps_val = 1.0e-7f;
        auto eps = std::make_shared<ov::op::v0::Constant>(elem_type, ov::Shape{}, eps_val);
        eps->set_friendly_name(divide_node->get_friendly_name() + "/guard_eps");

        // Check if both inputs are exactly zero
        auto real_is_zero = std::make_shared<ov::op::v1::Equal>(real_output, zero);
        real_is_zero->set_friendly_name(divide_node->get_friendly_name() + "/guard_real_eq_zero");

        auto imag_is_zero = std::make_shared<ov::op::v1::Equal>(imag_output, zero);
        imag_is_zero->set_friendly_name(divide_node->get_friendly_name() + "/guard_imag_eq_zero");

        auto both_zero = std::make_shared<ov::op::v1::LogicalAnd>(real_is_zero, imag_is_zero);
        both_zero->set_friendly_name(divide_node->get_friendly_name() + "/guard_both_zero");

        // Select(condition=both_zero, then=eps, else=real)
        // When both are zero -> use eps as denominator (safe)
        // Otherwise -> use original real (unchanged behavior)
        auto guarded_real = std::make_shared<ov::op::v1::Select>(both_zero, eps, real_output);
        guarded_real->set_friendly_name(divide_node->get_friendly_name() + "/guard_select");

        // Replace Divide's denominator input with the guarded version
        divide_node->input(1).replace_source_output(guarded_real);

        LOG_DEBUG("guard_angle_divide: patched '" << divide_node->get_friendly_name()
                                                  << "' (elem_type=" << elem_type << ", eps=" << eps_val << ")");
    }

    model->validate_nodes_and_infer_types();
}

}  // namespace kokoro
}  // namespace npuw
}  // namespace ov
