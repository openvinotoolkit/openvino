// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/arg_min_max_factory.hpp"
#include "default_opset.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/validation_util.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace utils
        {
            ArgMinMaxFactory::ArgMinMaxFactory(const Node& node)
                : m_keep_dims{node.get_attribute_value<std::int64_t>("keepdims", 1)}
                , m_input_node{node.get_ng_inputs().at(0)}
                , m_axis{node.get_attribute_value<std::int64_t>("axis", 0)}
                , m_select_last_index{
                      node.get_attribute_value<std::int64_t>("select_last_index", 0)}
            {
            }

            std::shared_ptr<ngraph::Node> ArgMinMaxFactory::make_arg_max() const
            {
                return make_topk_subgraph(default_opset::TopK::Mode::MAX);
            }

            std::shared_ptr<ngraph::Node> ArgMinMaxFactory::make_arg_min() const
            {
                return make_topk_subgraph(default_opset::TopK::Mode::MIN);
            }

            std::shared_ptr<ngraph::Node>
                ArgMinMaxFactory::make_topk_subgraph(default_opset::TopK::Mode mode) const
            {
                const auto k_node =
                    default_opset::Constant::create(ngraph::element::i64, Shape{}, {1});

                if (m_select_last_index == 1)
                {
                    // Example (ArgMin):
                    // The goal is to get the index of the last occurence of the
                    // minimum value present in given input tensor.
                    //
                    // Input:           [1, 2, 1, 3, 4, 4]
                    // Expected output: [2]
                    //
                    // Top-K is always returning the "most-left" result. The trick is to
                    // reverse input to find the "most-right" occurence which is equal to
                    // the last occurence in the original input.
                    // reverse = [4, 4, 3, 1, 2, 1]
                    //
                    // Run TopK on reversed tensor, in the example output with index values
                    // is equal to:
                    // topk->output(1) = 3
                    //
                    // Using ShapeOf and Gather on input obtain length of the input tensor
                    // along axis, in the example this is equal to:
                    // dims_on_axis = 6
                    //
                    // Now using two Substract ops calculate resulting index:
                    // res_index = dims_on_axis - topk->output(1) = 6 - 3 = 3
                    // result = res_index - 1 = 3 - 1 = 2

                    const auto axis_node =
                        default_opset::Constant::create(ngraph::element::i64, Shape{1}, {m_axis});
                    const auto reverse = std::make_shared<opset1::Reverse>(
                        m_input_node, axis_node, opset1::Reverse::Mode::INDEX);

                    const auto topk = std::make_shared<default_opset::TopK>(
                        reverse, k_node, m_axis, mode, default_opset::TopK::SortType::NONE);

                    const auto data_shape = std::make_shared<default_opset::ShapeOf>(m_input_node);
                    const auto dims_on_axis = std::make_shared<default_opset::Gather>(
                        data_shape,
                        axis_node,
                        default_opset::Constant::create(ngraph::element::i64, Shape{}, {0}));

                    const auto res_index = std::make_shared<default_opset::Subtract>(
                        dims_on_axis,
                        std::make_shared<default_opset::Convert>(topk->output(1), element::i64));
                    const auto result = std::make_shared<default_opset::Subtract>(
                        res_index,
                        default_opset::Constant::create(ngraph::element::i64, Shape{1}, {1}));

                    if (m_keep_dims == 0)
                    {
                        const auto axis_to_remove = default_opset::Constant::create(
                            element::u64, Shape{}, {topk->get_axis()});

                        return std::make_shared<default_opset::Squeeze>(result, axis_to_remove);
                    }

                    return result;
                }

                const auto topk = std::make_shared<default_opset::TopK>(
                    m_input_node, k_node, m_axis, mode, default_opset::TopK::SortType::NONE);

                const auto result =
                    std::make_shared<default_opset::Convert>(topk->output(1), element::i64);

                if (m_keep_dims == 0)
                {
                    const auto axis_to_remove =
                        default_opset::Constant::create(element::u64, Shape{}, {topk->get_axis()});

                    return std::make_shared<default_opset::Squeeze>(result, axis_to_remove);
                }

                return result;
            }
        } // namespace utils
    }     // namespace onnx_import
} // namespace ngraph
