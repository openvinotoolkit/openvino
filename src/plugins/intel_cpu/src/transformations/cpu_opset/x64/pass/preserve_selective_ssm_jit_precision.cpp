// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preserve_selective_ssm_jit_precision.hpp"

#include <cstddef>
#include <memory>

#include "nodes/kernels/x64/selective_ssm_jit_config.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {
namespace {

constexpr size_t data_input_count = 6;
constexpr size_t state_input_index = 5;
constexpr size_t state_rank = 4;

bool is_supported_data_precision(const ov::element::Type& precision) {
    return any_of(precision, ov::element::f16, ov::element::bf16);
}

bool has_supported_data_inputs(const std::shared_ptr<ov::Node>& node) {
    const auto precision = node->get_input_element_type(0);
    if (!is_supported_data_precision(precision)) {
        return false;
    }

    for (size_t input_index = 1; input_index < data_input_count; ++input_index) {
        if (node->get_input_element_type(input_index) != precision) {
            return false;
        }
    }
    return true;
}

bool has_supported_state_size(const std::shared_ptr<ov::Node>& node) {
    const auto& state_shape = node->get_input_partial_shape(state_input_index);
    if (state_shape.rank().is_dynamic() || state_shape.rank().get_length() != state_rank ||
        state_shape[state_rank - 1].is_dynamic()) {
        return false;
    }

    const auto size = state_shape[state_rank - 1].get_length();
    return size > 0 && static_cast<size_t>(size) <= kernel::max_selective_ssm_jit_state_size;
}

}  // namespace

PreserveSelectiveSSMJitPrecision::PreserveSelectiveSSMJitPrecision() {
    MATCHER_SCOPE(PreserveSelectiveSSMJitPrecision);

    const auto selective_ssm =
        ov::pass::pattern::wrap_type<ov::op::internal::SelectiveSSM, ov::op::internal::PagedSelectiveSSM>();
    const ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& matcher) {
        const auto& node = matcher.get_match_root();
        if (!has_supported_data_inputs(node) || !has_supported_state_size(node)) {
            return false;
        }

        const auto precision = node->get_input_element_type(0);
        ov::disable_conversion(node, precision, ov::element::f32);
        for (size_t input_index = 0; input_index < data_input_count; ++input_index) {
            ov::disable_conversion(node->get_input_node_shared_ptr(input_index), precision, ov::element::f32);
        }
        return false;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(selective_ssm, matcher_name), callback);
}

}  // namespace ov::intel_cpu
