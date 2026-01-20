// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "increase_position.hpp"

#include "intel_gpu/op/gemm.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

namespace ov::intel_gpu {

IncreasePrecision::IncreasePrecision() {}

bool IncreasePrecision::insert_converts_before_if_needed(const std::shared_ptr<ov::Node>& node,
                                                                const ov::element::Type desired_et,
                                                                size_t& input_idx,
                                                                const std::vector<size_t>& skip_inputs) {
    bool is_changed = false;
    for (const auto& input : node->inputs()) {
        const auto& incoming_output = input.get_source_output();
        const auto& incoming_node = incoming_output.get_node_shared_ptr();
        const auto input_et = incoming_output.get_element_type();

        if (input_et == desired_et)
            continue;

        if (std::find(skip_inputs.begin(), skip_inputs.end(), input.get_index()) != skip_inputs.end()) {
            continue;
        }

        auto in_convert = ov::as_type_ptr<ov::op::v0::Convert>(incoming_node);

        if (in_convert && in_convert->get_users().size() == 1 && input_et.bitwidth() <= desired_et.bitwidth()) {
            auto convert = std::make_shared<ov::op::v0::Convert>(incoming_node->input_value(0), desired_et);
            convert->set_friendly_name(in_convert->get_friendly_name() + "_increase_precision_" + std::to_string(input_idx));
            copy_runtime_info(incoming_node, convert);
            ov::replace_node(incoming_node, convert);
        } else {
            auto convert = std::make_shared<ov::op::v0::Convert>(incoming_output, desired_et);
            convert->set_friendly_name(incoming_node->get_friendly_name() + "_increase_precision_" + std::to_string(input_idx));
            copy_runtime_info(incoming_node, convert);
            input.replace_source_output(convert);
        }

        input_idx++;
        is_changed = true;
    }

    return is_changed;
}

void IncreasePrecision::insert_converts_after_if_needed(const std::shared_ptr<ov::Node>& node,
                                                            const ov::element::Type original_et, size_t& output_idx) {
    for (const auto& output : node->outputs()) {
        for (const auto& out_inputs : output.get_target_inputs()) {
            auto out_node = out_inputs.get_node()->shared_from_this();

            auto convert = std::make_shared<ov::op::v0::Convert>(output, original_et);
            auto convert_name = out_node->get_friendly_name() + "_restore_precision_" + std::to_string(output_idx);
            convert->set_friendly_name(convert_name);
            copy_runtime_info(node, convert);
            out_inputs.replace_source_output(convert);
            output_idx++;
        }
    }
}
}  // namespace ov::intel_gpu