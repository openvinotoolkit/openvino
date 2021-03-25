// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/validation_util.hpp"
#include "op/concat.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector concat(const Node& node)
                {
                    OutputVector inputs{node.get_ng_inputs()};
                    std::int64_t axis = node.get_attribute_value<std::int64_t>("axis");
                    return {std::make_shared<default_opset::Concat>(inputs, axis)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
