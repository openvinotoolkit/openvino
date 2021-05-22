// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/shape.hpp"
#include "op/pad.hpp"
#include "utils/convpool.hpp"

namespace
{
    ngraph::op::PadMode get_pad_mode(std::string mode)
    {
        ngraph::op::PadMode pad_mode;

        if (mode == "constant")
        {
            pad_mode = ngraph::op::PadMode::CONSTANT;
        }
        else if (mode == "reflect")
        {
            pad_mode = ngraph::op::PadMode::REFLECT;
        }
        else if (mode == "edge")
        {
            pad_mode = ngraph::op::PadMode::EDGE;
        }
        else
        {
            throw ngraph::ngraph_error("Unsupported padding mode: [" + mode + "]");
        }

        return pad_mode;
    }
} // namespace
namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector pad(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);

                    const auto data_rank = node.get_ng_inputs().at(0).get_partial_shape().rank();
                    CHECK_VALID_NODE(
                        node, data_rank.is_static(), "Data rank must be static for pad op");
                    const auto data_rank_value = data_rank.get_length();

                    double value = node.get_attribute_value<double>("value", 0);
                    const std::string mode =
                        node.get_attribute_value<std::string>("mode", "constant");
                    ngraph::op::PadMode pad_mode = get_pad_mode(mode);

                    const auto paddings = convpool::get_pads(node, data_rank_value);
                    ngraph::CoordinateDiff padding_below = paddings.first;
                    ngraph::CoordinateDiff padding_above = paddings.second;

                    return {std::make_shared<default_opset::Pad>(
                        data,
                        std::make_shared<default_opset::Constant>(
                            element::i64, ngraph::Shape{padding_below.size()}, padding_below),
                        std::make_shared<default_opset::Constant>(
                            element::i64, ngraph::Shape{padding_above.size()}, padding_above),
                        std::make_shared<default_opset::Constant>(
                            data.get_element_type(), ngraph::Shape{}, std::vector<double>{value}),
                        pad_mode)};
                }

            } // namespace set_1
            namespace set_11
            {
                OutputVector pad(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    auto pads = node.get_ng_inputs().at(1);
                    Output<ngraph::Node> values;
                    Output<ngraph::Node> padding_begin;
                    Output<ngraph::Node> padding_end;

                    if (node.get_ng_inputs().size() == 3)
                    {
                        values = node.get_ng_inputs().at(2);
                    }
                    else
                    {
                        values = default_opset::Constant::create(
                            data.get_element_type(), ngraph::Shape{}, {0});
                    }

                    if (ngraph::op::is_constant(pads.get_node()))
                    {
                        std::vector<std::int64_t> pads_vector =
                            ngraph::as_type_ptr<default_opset::Constant>(pads.get_node_shared_ptr())
                                ->get_vector<std::int64_t>();

                        std::size_t const half_size = pads_vector.size() / 2;
                        std::vector<std::int64_t> padding_begin_values(
                            pads_vector.begin(), pads_vector.begin() + half_size);
                        std::vector<std::int64_t> padding_end_values(
                            pads_vector.begin() + half_size, pads_vector.end());

                        padding_begin = default_opset::Constant::create(
                            element::i64, ngraph::Shape{half_size}, padding_begin_values);
                        padding_end = default_opset::Constant::create(
                            element::i64, ngraph::Shape{half_size}, padding_end_values);
                    }
                    else
                    {
                        OutputVector padding = builder::opset1::split(pads, 2, 0);

                        padding_begin = padding.at(0);
                        padding_end = padding.at(1);
                    }

                    const std::string mode =
                        node.get_attribute_value<std::string>("mode", "constant");
                    ngraph::op::PadMode pad_mode = get_pad_mode(mode);

                    return {std::make_shared<default_opset::Pad>(
                        data, padding_begin, padding_end, values, pad_mode)};
                }

            } // namespace set_11

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
