//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <memory>

#include "core/null_node.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "op/dropout.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                OutputVector
                    build_dropout(const Node& node, float drop_probability, bool training_mode)
                {
                    CHECK_VALID_NODE(
                        node,
                        drop_probability == 0 || !training_mode,
                        "Training mode is not supported for Dropout op if drop_probability is not "
                        "equal 0");

                    const auto input_data = node.get_ng_inputs().at(0);
                    const bool return_mask = node.get_outputs_size() > 1;

                    if (return_mask)
                    {
                        const auto mask = std::make_shared<default_opset::Broadcast>(
                            default_opset::Constant::create(
                                ngraph::element::boolean, Shape{}, {true}),
                            std::make_shared<default_opset::ShapeOf>(input_data));
                        return {input_data, mask};
                    }
                    else
                    {
                        return {input_data};
                    }
                }
            }

            namespace set_12
            {
                OutputVector dropout(const Node& node)
                {
                    const auto ng_inputs = node.get_ng_inputs();
                    // seed attribute is ignored because traning mode is not supported anyway

                    // default values of inputs
                    double ratio = 0.5f;
                    bool training_mode = false;

                    if (ng_inputs.size() > 1)
                    {
                        if (!ngraph::op::is_null(ng_inputs.at(1)))
                        {
                            CHECK_VALID_NODE(
                                node,
                                ngraph::op::is_constant(ng_inputs.at(1).get_node_shared_ptr()),
                                "Not constant (or omitted) ratio input is not supported.");
                            ratio = as_type_ptr<default_opset::Constant>(
                                        ng_inputs.at(1).get_node_shared_ptr())
                                        ->cast_vector<double>()[0];
                        }
                    }
                    if (ng_inputs.size() > 2)
                    {
                        if (!ngraph::op::is_null(ng_inputs.at(2)))
                        {
                            CHECK_VALID_NODE(
                                node,
                                ngraph::op::is_constant(ng_inputs.at(2).get_node_shared_ptr()),
                                "Not constant (or omitted) training_mode input is not supported.");
                            training_mode = as_type_ptr<default_opset::Constant>(
                                                ng_inputs.at(2).get_node_shared_ptr())
                                                ->cast_vector<bool>()[0];
                        }
                    }
                    return build_dropout(node, ratio, training_mode);
                }
            } // namespace set_12

            namespace set_7
            {
                OutputVector dropout(const Node& node)
                {
                    // "is_test" attribute was removed
                    const bool training_mode = false;
                    const auto ratio = node.get_attribute_value<float>("ratio", 0.5f);

                    return build_dropout(node, ratio, training_mode);
                }
            } // namespace set_7

            namespace set_1
            {
                OutputVector dropout(const Node& node)
                {
                    // legacy consumed_inputs attribute ignored
                    const bool training_mode = !node.get_attribute_value<int64_t>("is_test", 0);
                    const auto ratio = node.get_attribute_value<float>("ratio", 0.5f);

                    return build_dropout(node, ratio, training_mode);
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
