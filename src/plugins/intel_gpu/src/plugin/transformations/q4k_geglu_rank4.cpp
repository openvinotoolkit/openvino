// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "q4k_geglu_rank4.hpp"

#include <map>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov::intel_gpu {
namespace {

bool is_rank3_activation_reshape(const std::shared_ptr<ov::op::v1::Reshape>& reshape) {
    const auto input_shape = reshape->get_input_partial_shape(0);
    const auto output_shape = reshape->get_output_partial_shape(0);
    if (input_shape.rank().is_dynamic() || input_shape.rank().get_length() != 4 || output_shape.rank().is_dynamic() || output_shape.rank().get_length() != 3 ||
        !reshape->get_special_zero() || !input_shape[1].is_static() || input_shape[1].get_length() != 1 || !input_shape[3].is_static()) {
        return false;
    }

    const auto target = ov::as_type_ptr<ov::op::v0::Constant>(reshape->input_value(1).get_node_shared_ptr());
    if (!target || target->get_shape() != ov::Shape{3})
        return false;

    const auto target_dims = target->cast_vector<int64_t>();
    return target_dims[0] == 0 && target_dims[1] == -1 && target_dims[2] == input_shape[3].get_length();
}

}  // namespace

bool Q4KGeGLURank4::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    const auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    std::map<ov::Output<ov::Node>, ov::Output<ov::Node>> rank4_inputs;
    for (const auto& node : model->get_ordered_ops()) {
        const auto hint = node->get_rt_info().find("gguf_gpu_rank4_hint");
        if (!ov::is_type<ov::op::v0::MatMul>(node) || hint == node->get_rt_info().end() || !hint->second.is<bool>() || !hint->second.as<bool>()) {
            continue;
        }

        const auto input = node->input_value(0);
        const auto input_shape = input.get_partial_shape();
        if (input_shape.rank().is_dynamic() || input_shape.rank().get_length() != 3)
            continue;

        auto rank4_input = rank4_inputs.find(input);
        if (rank4_input == rank4_inputs.end()) {
            const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(input.get_node_shared_ptr());
            const auto reshape_input = convert ? convert->input_value(0) : input;
            const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(reshape_input.get_node_shared_ptr());
            ov::Output<ov::Node> expanded_input;
            if (reshape && is_rank3_activation_reshape(reshape)) {
                expanded_input = reshape->input_value(0);
                if (convert) {
                    auto rank4_convert = std::make_shared<ov::op::v0::Convert>(expanded_input, convert->get_destination_type());
                    ov::copy_runtime_info(convert, rank4_convert);
                    rank4_convert->set_friendly_name(convert->get_friendly_name());
                    expanded_input = rank4_convert;
                }
            } else {
                auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input, axis);
                ov::copy_runtime_info(input.get_node_shared_ptr(), unsqueeze);
                expanded_input = unsqueeze;
            }
            rank4_input = rank4_inputs.emplace(input, expanded_input).first;
        }

        node->input(0).replace_source_output(rank4_input->second);
        node->validate_and_infer_types();
        changed = true;
    }
    return changed;
}

}  // namespace ov::intel_gpu
