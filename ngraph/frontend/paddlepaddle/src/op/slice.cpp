// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "slice.hpp"
#include <ngraph/opsets/opset6.hpp>
#include <paddlepaddle_frontend/utility.hpp>

using namespace ngraph;
using namespace ngraph::frontend;

namespace pdpd
{
    namespace op
    {
        NamedOutputs slice(const NodeContext& node)
        {
            auto data = node.get_ng_input("Input");
            auto axes = node.get_attribute<std::vector<int32_t>>("axes");
            // TODO: support tensor type #55266
            auto starts = node.get_attribute<std::vector<int32_t>>("starts");
            // TODO: support tensor type #55266
            auto ends = node.get_attribute<std::vector<int32_t>>("ends");
            auto data_rank = data.get_partial_shape().rank();
            size_t shape_size = data_rank.get_length();
            std::vector<int32_t> fixedStarts(shape_size, 0);
            std::vector<int32_t> fixedEnds(shape_size, INT_MAX);

            int n = 0;
            for (auto i : axes)
            {
                PDPD_ASSERT(i < (int32_t)shape_size, "slice: axes must be less than the X rank.");
                fixedStarts[i] = starts[n];
                fixedEnds[i] = ends[n];
                n++;
            }

            auto startsNode = opset6::Constant::create(element::i32, {shape_size}, fixedStarts);
            auto endsNode = opset6::Constant::create(element::i32, {shape_size}, fixedEnds);
            auto stridesNode = opset6::Constant::create(
                element::i32, {shape_size}, std::vector<int32_t>(shape_size, 1));
            return node.default_single_output_mapping(
                {std::make_shared<opset6::StridedSlice>(data,
                                                        startsNode,
                                                        endsNode,
                                                        stridesNode,
                                                        std::vector<int64_t>(shape_size, 0),
                                                        std::vector<int64_t>(shape_size, 0))},
                {"Out"});
        }

    } // namespace op
} // namespace pdpd
