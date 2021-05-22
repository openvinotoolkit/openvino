// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/shape.hpp"
#include "op/reshape.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector reshape(const Node& node)
                {
                    OutputVector ng_inputs{node.get_ng_inputs()};
                    const auto data = ng_inputs.at(0);

                    Output<ngraph::Node> pattern;
                    bool special_zero = true;
                    // Since opset 5 the target shape is provided as input
                    if (ng_inputs.size() == 2)
                    {
                        pattern = ng_inputs.at(1);
                    }
                    else
                    {
                        const auto output_shape =
                            node.get_attribute_value<std::vector<int64_t>>("shape", {});

                        // Added in onnx reshape version 14
                        special_zero = !node.get_attribute_value<int64_t>("allowzero", 0);

                        pattern = default_opset::Constant::create(
                            element::i64, Shape{output_shape.size()}, output_shape);
                    }

                    return {std::make_shared<default_opset::Reshape>(data, pattern, special_zero)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
