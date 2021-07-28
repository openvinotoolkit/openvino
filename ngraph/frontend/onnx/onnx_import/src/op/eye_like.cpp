// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "exceptions.hpp"
#include "op/eye_like.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector eye_like(const Node& node)
                {
                    const auto input = node.get_ng_inputs().at(0);
                    const auto& input_shape = input.get_shape();

                    std::int64_t dtype;
                    element::Type target_type;

                    std::int64_t shift = node.get_attribute_value<std::int64_t>("k", 0);
                    if (node.has_attribute("dtype"))
                    {
                        dtype = node.get_attribute_value<std::int64_t>("dtype");
                        target_type = common::get_ngraph_element_type(dtype);
                    }
                    else
                    {
                        target_type = input.get_element_type();
                    }

                    CHECK_VALID_NODE(node,
                                     input_shape.size() == 2,
                                     "The provided shape rank: ",
                                     input_shape.size(),
                                     " is unsupported, only 2D shapes are supported");

                    std::shared_ptr<ngraph::Node> eye_like_matrix =
                        common::shifted_square_identity(input_shape, target_type, shift);

                    return {eye_like_matrix};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
