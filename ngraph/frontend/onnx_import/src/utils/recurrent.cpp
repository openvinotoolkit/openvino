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

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "default_opset.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/check.hpp"
#include "ngraph/enum_names.hpp"
#include "onnx_import/core/null_node.hpp"
#include "utils/recurrent.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace recurrent
        {
            OpInputMap::OpInputMap(const onnx_import::Node& node, std::size_t gates_count)
            {
                const auto& ng_inputs = node.get_ng_inputs();

                m_map[OpInput::X] = builder::opset1::reorder_axes(ng_inputs.at(0), {1, 0, 2});
                m_map[OpInput::W] = ng_inputs.at(1);
                m_map[OpInput::R] = ng_inputs.at(2);

                const auto el_type = ng_inputs.at(0).get_element_type();

                const auto x_pshape = m_map[OpInput::X].get_partial_shape();
                const auto w_pshape = m_map[OpInput::W].get_partial_shape();
                const auto r_pshape = m_map[OpInput::R].get_partial_shape();
                NGRAPH_CHECK(x_pshape.rank().is_static() && x_pshape[0].is_static() &&
                                 x_pshape[1].is_static(),
                             "RecurrentSequence input X must have static \"seq_length\" and "
                             "\"batch_size\" dimensions.");
                NGRAPH_CHECK(w_pshape.rank().is_static() && w_pshape[0].is_static(),
                             "RecurrentSequence input W must have static \"num_directions\" "
                             "(outermost) dimension.");
                NGRAPH_CHECK(r_pshape.rank().is_static() && r_pshape[2].is_static(),
                             "RecurrentSequence input R must have static \"hidden_size\" "
                             "(innermost) dimension.");

                const std::size_t hidden_size = m_map[OpInput::R].get_shape().back();
                const std::size_t batch_size = m_map[OpInput::X].get_shape().at(0);
                const std::size_t num_directions = m_map[OpInput::W].get_shape().front();

                if (ng_inputs.size() > 3 && !ngraph::op::is_null(ng_inputs.at(3)))
                {
                    auto bias = ng_inputs.at(3);
                    auto split_bias = builder::opset1::split(bias, 2, 1);
                    NGRAPH_SUPPRESS_DEPRECATED_START
                    m_map[OpInput::B] =
                        std::make_shared<ngraph::op::v1::Add>(split_bias.at(0), split_bias.at(1));
                    NGRAPH_SUPPRESS_DEPRECATED_END
                }
                else
                {
                    m_map[OpInput::B] = std::make_shared<default_opset::Constant>(
                        el_type, Shape{num_directions, gates_count * hidden_size}, 0.f);
                }
                if (ng_inputs.size() > 4 && !ngraph::op::is_null(ng_inputs.at(4)))
                {
                    m_map[OpInput::SEQ_LENGTHS] = ng_inputs.at(4);
                }
                else
                {
                    m_map[OpInput::SEQ_LENGTHS] = std::make_shared<default_opset::Constant>(
                        element::i32, Shape{batch_size}, m_map[OpInput::X].get_shape().at(1));
                }
                // The initial value of the hidden.
                if (ng_inputs.size() > 5 && !ngraph::op::is_null(ng_inputs.at(5)))
                {
                    m_map[OpInput::INIT_H] =
                        builder::opset1::reorder_axes(ng_inputs.at(5), {1, 0, 2});
                }
                else
                {
                    m_map[OpInput::INIT_H] = std::make_shared<default_opset::Constant>(
                        el_type, Shape{batch_size, num_directions, hidden_size}, 0.f);
                }
            }

            OpInputMap::OpInputMap(container_type&& map)
                : m_map(std::move(map))
            {
            }

            Output<ngraph::Node>& OpInputMap::at(const OpInput& key) { return m_map.at(key); }
            const Output<ngraph::Node>& OpInputMap::at(const OpInput& key) const
            {
                return m_map.at(key);
            }

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ATTRIBUTES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            OpAttributes::OpAttributes(const Node& node)
                : m_hidden_size{node.get_attribute_value<std::int64_t>("hidden_size")}
                , m_clip_threshold{node.get_attribute_value<float>("clip", 0.f)}
                // Recurrent Operators which have more activation functions should override
                // this value in constructor of respective Attributes' struct.
                , m_activations{node.get_attribute_value<std::vector<std::string>>("activations",
                                                                                   {"tanh"})}
                // Default values for activation functions are same as for corresponding
                // ONNX operator.
                , m_activations_alpha{node.get_attribute_value<std::vector<float>>(
                      "activation_alpha", std::vector<float>{})}
                , m_activations_beta{node.get_attribute_value<std::vector<float>>(
                      "activation_beta", std::vector<float>{})}
            {
                m_clip_threshold = std::abs(m_clip_threshold);
                std::string direction =
                    ngraph::to_lower(node.get_attribute_value<std::string>("direction", "forward"));
                m_direction = ngraph::as_enum<ngraph::op::RecurrentSequenceDirection>(direction);
            }

        } // recurrent
    }     // onnx_import
} // ngraph
