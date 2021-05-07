// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/mod.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "op/mod.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector mod(const Node& node)
                {
                    Output<ngraph::Node> dividend{node.get_ng_inputs().at(0)};
                    Output<ngraph::Node> divisor{node.get_ng_inputs().at(1)};

                    std::int64_t fmod = node.get_attribute_value<std::int64_t>("fmod", 0);
                    CHECK_VALID_NODE(
                        node, fmod == 1, "Only 'fmod=1' mode is supported for mod operator.");

                    return {std::make_shared<default_opset::Mod>(dividend, divisor)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
