// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>

#include "default_opset.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/op/mvn.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/validation_util.hpp"
#include "op/mean_variance_normalization.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector mean_variance_normalization(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    bool across_channels =
                        node.get_attribute_value<std::int64_t>("across_channels", 0);
                    bool normalize_variance =
                        node.get_attribute_value<std::int64_t>("normalize_variance", 1);

                    return {std::make_shared<ngraph::opset5::MVN>(
                        data, across_channels, normalize_variance)};
                }

            } // namespace set_1

            namespace set_9
            {
                OutputVector mean_variance_normalization(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    auto axes =
                        node.get_attribute_value<std::vector<std::int64_t>>("axes", {0, 2, 3});
                    const std::vector<std::size_t> normalized_axes = ngraph::normalize_axes(
                        node.get_description(), axes, data.get_partial_shape().rank());
                    auto const_axes = default_opset::Constant::create(
                        element::i64, Shape{normalized_axes.size()}, normalized_axes);
                    return {std::make_shared<ngraph::op::v6::MVN>(
                        data, const_axes, true, 1e-09, ngraph::op::MVNEpsMode::OUTSIDE_SQRT)};
                }

            } // namespace set_9

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
