// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// ggml_top_k(a, k): the indices of the k largest values along ne[0] (the OV last axis).
// The caller supplies k; legacy decoders encode it in the output's last dimension.
OutputVector translate_top_k(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    const int groups = context.get_attribute<int>("expert_groups", 1);
    if (groups > 1) {
        using namespace ov::op;
        const int used = context.get_attribute<int>("expert_groups_used");
        const auto shape = input.get_partial_shape();
        const auto experts = shape[shape.rank().get_length() - 1].get_length();
        FRONT_END_GENERAL_CHECK(experts % groups == 0 && experts / groups >= 2 && used > 0 && used <= groups,
                                "Invalid grouped expert selection dimensions");
        const auto grouped_shape =
            v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{-1, groups, experts / groups});
        auto grouped = std::make_shared<v1::Reshape>(input, grouped_shape, false);
        auto two = v0::Constant::create(ov::element::i64, {}, {2});
        auto best =
            std::make_shared<v11::TopK>(grouped, two, -1, v11::TopK::Mode::MAX, v11::TopK::SortType::SORT_VALUES);
        auto axis = v0::Constant::create(ov::element::i64, {}, {-1});
        // llama.cpp scores each group by its two highest (possibly biased) expert probabilities.
        auto scores = std::make_shared<v1::ReduceSum>(best->output(0), axis, false);
        auto selected =
            make_topk_indices(scores, v0::Constant::create(ov::element::i64, {}, {used}), -1, v11::TopK::Mode::MAX);
        auto zero = v0::Constant::create(ov::element::boolean, {}, {false});
        auto one = v0::Constant::create(ov::element::boolean, {}, {true});
        auto mask = std::make_shared<v3::Broadcast>(zero, std::make_shared<v3::ShapeOf>(scores));
        auto updates = std::make_shared<v3::Broadcast>(one, std::make_shared<v3::ShapeOf>(selected));
        auto selected_mask = std::make_shared<v3::ScatterElementsUpdate>(mask, selected, updates, axis);
        auto expanded = std::make_shared<v0::Unsqueeze>(selected_mask, axis);
        auto negative = v0::Constant::create(input.get_element_type(), {}, {-std::numeric_limits<float>::infinity()});
        auto filtered = std::make_shared<v1::Select>(expanded, grouped, negative);
        input = std::make_shared<v1::Reshape>(filtered, std::make_shared<v3::ShapeOf>(input), false);
    }

    const auto k_attribute = context.get_attribute_as_any("k");
    int64_t k;
    if (!k_attribute.empty()) {
        k = k_attribute.as<int64_t>();
    } else {
        const auto shape = context.get_output_shape();
        const auto rank = shape.rank();
        FRONT_END_OP_CONVERSION_CHECK(
            rank.is_static() && rank.get_length() > 0 && shape[rank.get_length() - 1].is_static(),
            "TOP_K requires 'k' or a static last output dimension");
        k = shape[rank.get_length() - 1].get_length();
    }

    auto k_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {k});
    auto indices = make_topk_indices(input, k_node, -1, ov::op::v11::TopK::Mode::MAX);

    return rename_outputs_with_suffix({std::move(indices)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
