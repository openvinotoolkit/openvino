//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <string>
#include <vector>

#include "default_opset.hpp"
#include "gru.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/shape.hpp"
#include "utils/recurrent.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                namespace
                {
                    struct GRUInputMap : public recurrent::OpInputMap
                    {
                        GRUInputMap(const Node& node, std::size_t gates_count)
                            : OpInputMap(node, gates_count)
                        {
                            bool linear_before_reset = static_cast<bool>(
                                node.get_attribute_value<std::int64_t>("linear_before_reset", 0));

                            // Override bias, since we need separated W and R biases for `h` gate.
                            if (linear_before_reset)
                            {
                                const auto& ng_inputs = node.get_ng_inputs();
                                const auto el_type = ng_inputs.at(0)->get_output_element_type(0);

                                if (ng_inputs.size() > 3 && !ng_inputs.at(3)->is_null())
                                {
                                    auto bias = ng_inputs.at(3);
                                    // gates_count * 2 since B is: [Wb, Rb]
                                    const int split_parts = 2 * 3;
                                    const auto split_bias =
                                        builder::opset1::split(bias, split_parts, 1);
                                    const auto wr_z_bias = split_bias.at(0) + split_bias.at(3);
                                    const auto wr_r_bias = split_bias.at(1) + split_bias.at(4);
                                    // The result has shape: [num_directions, 4 * hidden_size]
                                    // and data layout:
                                    //       [
                                    //          [Wb_z + Rb_z],
                                    //          [Wb_r + Rb_r],
                                    //          [Wb_h],
                                    //          [Rb_h],
                                    //          // num_directions times
                                    //       ]
                                    m_map[recurrent::OpInput::B] =
                                        std::make_shared<default_opset::Concat>(
                                            NodeVector{wr_z_bias,
                                                       wr_r_bias,
                                                       split_bias.at(2),
                                                       split_bias.at(5)},
                                            1);
                                }
                                else
                                {
                                    const std::size_t hidden_size =
                                        m_map[recurrent::OpInput::R]->get_shape().back();
                                    const std::size_t num_directions =
                                        m_map[recurrent::OpInput::W]->get_shape().front();

                                    m_map[recurrent::OpInput::B] =
                                        std::make_shared<default_opset::Constant>(
                                            el_type,
                                            Shape{num_directions, (gates_count + 1) * hidden_size},
                                            0.f);
                                }
                            }
                        }

                        virtual ~GRUInputMap() = default;
                    };

                    struct GRUAttributes : public recurrent::OpAttributes
                    {
                        GRUAttributes(const Node& node)
                            : OpAttributes(node)
                            , m_linear_before_reset{static_cast<bool>(
                                  node.get_attribute_value<std::int64_t>("linear_before_reset", 0))}
                        {
                            m_activations = node.get_attribute_value<std::vector<std::string>>(
                                "activations", {"sigmoid", "tanh"});
                        }

                        virtual ~GRUAttributes() = default;

                        bool m_linear_before_reset;
                    };
                }

                NodeVector gru(const Node& node)
                {
                    constexpr std::size_t gates_count = 3;
                    GRUInputMap input_map{node, gates_count};
                    GRUAttributes attributes{node};

                    recurrent::RecurrentSequence sequence_op(input_map, attributes.m_direction);
                    auto results =
                        sequence_op.run_sequence([&attributes](const recurrent::OpInputMap& args,
                                                               const Output<ngraph::Node>& in_Xt,
                                                               const Output<ngraph::Node> H_t) {

                            const GRUInputMap& gru_args = dynamic_cast<const GRUInputMap&>(args);

                            return std::make_shared<default_opset::GRUCell>(
                                in_Xt,
                                H_t,
                                gru_args.at(recurrent::OpInput::W),
                                gru_args.at(recurrent::OpInput::R),
                                gru_args.at(recurrent::OpInput::B),
                                attributes.m_hidden_size,
                                attributes.m_activations,
                                attributes.m_activations_alpha,
                                attributes.m_activations_beta,
                                attributes.m_clip_threshold,
                                attributes.m_linear_before_reset);
                        });
                    return results;
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
