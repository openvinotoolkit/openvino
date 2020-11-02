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

#include <memory>

#include "ngraph/builder/reshape.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/op/rnn.hpp"
#include "onnx_import/utils/recurrent.hpp"

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
                    struct RNNInputMap : public recurrent::OpInputMap
                    {
                        RNNInputMap(const onnx_import::Node& node, std::size_t gates_count)
                            : OpInputMap(node, gates_count)
                        {
                        }

                        virtual ~RNNInputMap() = default;
                    };

                    struct RNNAttributes : public recurrent::OpAttributes
                    {
                        RNNAttributes(const Node& node)
                            : OpAttributes(node)
                        {
                        }

                        virtual ~RNNAttributes() = default;
                    };
                }

                OutputVector rnn(const Node& node)
                {
                    constexpr std::size_t gates_count = 1;
                    RNNInputMap input_map{node, gates_count};
                    RNNAttributes attributes{node};

                    auto rnn_sequence = std::make_shared<default_opset::RNNSequence>(
                        input_map.at(recurrent::OpInput::X),
                        input_map.at(recurrent::OpInput::INIT_H),
                        input_map.at(recurrent::OpInput::SEQ_LENGTHS),
                        input_map.at(recurrent::OpInput::W),
                        input_map.at(recurrent::OpInput::R),
                        input_map.at(recurrent::OpInput::B),
                        attributes.m_hidden_size,
                        attributes.m_direction,
                        attributes.m_activations,
                        attributes.m_activations_alpha,
                        attributes.m_activations_beta,
                        attributes.m_clip_threshold);

                    const auto Y = rnn_sequence->output(0);
                    const auto Y_h = rnn_sequence->output(1);

                    return {builder::opset1::reorder_axes(Y, {2, 1, 0, 3}),
                            builder::opset1::reorder_axes(Y_h, {1, 0, 2})};
                }
            } // namespace set_1
        }     // namespace op
    }         // namespace onnx_import
} // namespace ngraph
