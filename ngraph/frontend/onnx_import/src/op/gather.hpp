// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/validation_util.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline OutputVector gather(const Node& node)
                {
                    OutputVector ng_inputs{node.get_ng_inputs()};
                    auto data = ng_inputs.at(0);
                    auto indices = ng_inputs.at(1);
                    auto axis = node.get_attribute_value<int64_t>("axis", 0);
                    const auto valid_axis = ngraph::normalize_axis(
                        node.get_description(), axis, data.get_partial_shape().rank());

                    return {std::make_shared<default_opset::Gather>(
                        data,
                        indices,
                        default_opset::Constant::create(element::i64, Shape{}, {valid_axis}))};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
