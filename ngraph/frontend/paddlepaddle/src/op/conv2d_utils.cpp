// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>

#include "conv2d_utils.hpp"
#include "node_context.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                ngraph::op::PadType get_auto_pad(const NodeContext& node)
                {
                    // Default value means use explicitly provided padding values.
                    ngraph::op::PadType pad_type{ngraph::op::PadType::NOTSET};
                    auto padding_algorithm = node.get_attribute<std::string>("padding_algorithm");
                    static std::unordered_map<std::string, ngraph::op::PadType> auto_pad_values{
                        {"VALID", ngraph::op::PadType::VALID},
                        {"SAME", ngraph::op::PadType::SAME_UPPER},
                        {"NOTSET", ngraph::op::PadType::NOTSET},
                    };

                    const auto pad_val_it = auto_pad_values.find(padding_algorithm);

                    if (pad_val_it == auto_pad_values.end())
                    {
                        pad_type = ngraph::op::PadType::NOTSET;
                    }
                    else
                    {
                        pad_type = pad_val_it->second;
                    }

                    return pad_type;
                }

                std::pair<CoordinateDiff, CoordinateDiff> get_pads(const NodeContext& node,
                                                                   const size_t kernel_rank)
                {
                    CoordinateDiff pads(kernel_rank, 0);

                    auto pads_int32 = node.get_attribute<std::vector<int32_t>>("paddings");
                    pads = CoordinateDiff{std::begin(pads_int32), std::end(pads_int32)};
                    CoordinateDiff pads_begin;
                    CoordinateDiff pads_end;

                    if (pads.size() == kernel_rank * 2)
                    {
                        for (size_t i = 0; i < pads.size(); i++)
                        {
                            if (i & 0x01)
                            {
                                pads_end.push_back(pads[i]);
                            }
                            else
                            {
                                pads_begin.push_back(pads[i]);
                            }
                        }
                        return {pads_begin, pads_end};
                    }
                    else
                    {
                        // No paddings provided or only one side values provided, which means same
                        // padding at both begin and end of axis.
                        return {pads, pads};
                    }
                }

                std::pair<CoordinateDiff, CoordinateDiff> get_pads(const NodeContext& node)
                {
                    const auto data_rank = node.get_ng_input("Input").get_partial_shape().rank();
                    PDPD_ASSERT(data_rank.get_length() > 2, "the rank of conv input must > 2");
                    const auto data_spatial_dims = data_rank.get_length() - 2;

                    return get_pads(node, data_spatial_dims);
                }
                std::shared_ptr<Node> get_reshaped_filter(const Output<Node>& filters,
                                                          const int32_t groups)
                {
                    auto shape_of_filters = std::make_shared<opset6::ShapeOf>(filters);

                    auto num_begin = opset6::Constant::create(element::i64, Shape{1}, {0});
                    auto num_end = opset6::Constant::create(element::i64, Shape{1}, {1});
                    auto num_node = std::make_shared<opset6::StridedSlice>(shape_of_filters,
                                                                           num_begin,
                                                                           num_end,
                                                                           std::vector<int64_t>{0},
                                                                           std::vector<int64_t>{0});

                    auto hw_begin = opset6::Constant::create(element::i64, Shape{1}, {1});
                    auto hw_end = opset6::Constant::create(element::i64, Shape{1}, {4});
                    auto filter_hw_node =
                        std::make_shared<opset6::StridedSlice>(shape_of_filters,
                                                               hw_begin,
                                                               hw_end,
                                                               std::vector<int64_t>{0},
                                                               std::vector<int64_t>{0});

                    auto groups_node = opset6::Constant::create(element::i64, Shape{1}, {groups});
                    auto grouped_num_node = std::make_shared<opset6::Divide>(num_node, groups_node);
                    auto target_filter_shape = std::make_shared<opset6::Concat>(
                        OutputVector{groups_node, grouped_num_node, filter_hw_node}, 0);
                    return std::make_shared<opset6::Reshape>(filters, target_filter_shape, false);
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
