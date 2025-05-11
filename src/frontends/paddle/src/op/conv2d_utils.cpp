// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv2d_utils.hpp"

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
ov::op::PadType get_auto_pad(const NodeContext& node) {
    // Default value means use explicitly provided padding values.
    ov::op::PadType pad_type{ov::op::PadType::NOTSET};
    auto padding_algorithm = node.get_attribute<std::string>("padding_algorithm");
    static std::unordered_map<std::string, ov::op::PadType> auto_pad_values{
        {"VALID", ov::op::PadType::VALID},
        {"SAME", ov::op::PadType::SAME_UPPER},
        {"NOTSET", ov::op::PadType::NOTSET},
    };

    const auto pad_val_it = auto_pad_values.find(padding_algorithm);

    if (pad_val_it == auto_pad_values.end()) {
        pad_type = ov::op::PadType::NOTSET;
    } else {
        pad_type = pad_val_it->second;
    }

    return pad_type;
}

std::pair<CoordinateDiff, CoordinateDiff> get_pads(const NodeContext& node, const size_t kernel_rank) {
    CoordinateDiff pads(kernel_rank, 0);

    auto pads_int32 = node.get_attribute<std::vector<int32_t>>("paddings");
    pads = CoordinateDiff{std::begin(pads_int32), std::end(pads_int32)};
    CoordinateDiff pads_begin;
    CoordinateDiff pads_end;

    if (pads.size() == kernel_rank * 2) {
        for (size_t i = 0; i < pads.size(); i++) {
            if (i & 0x01) {
                pads_end.push_back(pads[i]);
            } else {
                pads_begin.push_back(pads[i]);
            }
        }
        return {pads_begin, pads_end};
    } else {
        // No paddings provided or only one side values provided, which means same
        // padding at both begin and end of axis.
        return {pads, pads};
    }
}

std::pair<CoordinateDiff, CoordinateDiff> get_pads(const NodeContext& node) {
    const auto data_rank = node.get_input("Input").get_partial_shape().rank();
    PADDLE_OP_CHECK(node, data_rank.get_length() > 2, "the rank of conv input must > 2");
    const auto data_spatial_dims = data_rank.get_length() - 2;

    return get_pads(node, data_spatial_dims);
}
std::shared_ptr<Node> get_reshaped_filter(const Output<Node>& filters, const int32_t groups) {
    /*  filters' layout is [O,I,W,H].
     *  Divide O with groups:
     *      grouped_O = O / groups
     *  The final grouped filters' layout is [groups, grouped_O, I, W, H]
     */
    const std::vector<size_t> o_indices{0};
    auto filter_o_node = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(filters, o_indices);

    const std::vector<size_t> ihw_indices{1, 2, 3};
    auto filter_ihw_node = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(filters, ihw_indices);

    auto groups_node = opset6::Constant::create(element::i64, Shape{1}, {groups});
    auto grouped_o_node = std::make_shared<opset6::Divide>(filter_o_node, groups_node);
    auto target_filter_shape =
        std::make_shared<opset6::Concat>(OutputVector{groups_node, grouped_o_node, filter_ihw_node}, 0);
    return std::make_shared<opset6::Reshape>(filters, target_filter_shape, false);
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
