// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "op/org.openvinotoolkit/fake_quantize.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector fake_quantize(const onnx_import::Node& node)
                {
                    const auto inputs = node.get_ng_inputs();
                    const auto X = inputs.at(0);
                    const auto input_low = inputs.at(1);
                    const auto input_high = inputs.at(2);
                    const auto output_low = inputs.at(3);
                    const auto output_high = inputs.at(4);

                    const auto levels = node.get_attribute_value<std::size_t>("levels");

                    return {std::make_shared<default_opset::FakeQuantize>(
                        X, input_low, input_high, output_low, output_high, levels)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
