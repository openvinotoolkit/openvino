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

#pragma once

#include <functional>
#include <map>
#include <memory>

#include "core/node.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace recurrent
        {
            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INPUT NODES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            ///
            /// \brief      This class describes a recurrent operation input name
            ///
            enum class NGRAPH_API OpInput
            {
                X,           // Packed input sequences.
                             // Shape: [seq_length, batch_size, input_size]
                W,           // Weight tensor for the gates.
                             // Shape: [num_directions, gates_count*hidden_size, input_size]
                R,           // The recurrence weight tensor.
                             // Shape: [num_directions, gates_count*hidden_size, hidden_size]
                B,           // The bias tensor for gates.
                             // Shape [num_directions, gates_count*hidden_size]
                SEQ_LENGTHS, // The lengths of the sequences in a batch. Shape [batch_size]
                INIT_H,      // The initial value of the hidden.
                             // Shape [num_directions, batch_size, hidden_size]
            };

            ///
            /// \brief      { struct_description }
            ///
            struct NGRAPH_API OpInputMap
            {
                using container_type = std::map<OpInput, std::shared_ptr<ngraph::Node>>;

                explicit OpInputMap(const onnx_import::Node& node, std::size_t gates_count);
                OpInputMap(container_type&& map);
                virtual ~OpInputMap() = default;

                std::shared_ptr<ngraph::Node>& at(const OpInput& key);
                const std::shared_ptr<ngraph::Node>& at(const OpInput& key) const;

                container_type m_map;
            };

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ATTRIBUTES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            struct NGRAPH_API OpAttributes
            {
                explicit OpAttributes(const Node& node);
                virtual ~OpAttributes() = default;

                ngraph::op::RecurrentSequenceDirection m_direction;
                std::int64_t m_hidden_size;
                float m_clip_threshold;
                std::vector<std::string> m_activations;
                std::vector<float> m_activations_alpha;
                std::vector<float> m_activations_beta;
            };

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Helper classes~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            ///
            /// \brief      Callable object defining recurrent cell computations.
            ///
            /// Function returns node output representing cell hidden state after cell
            /// computations. The arguments are:
            ///     * input node map.
            ///     * the cell input data
            ///     * the cell hidden state from previous step.
            ///
            using RecurrentCellFunction = std::function<Output<ngraph::Node>(
                const OpInputMap&, const Output<ngraph::Node>&, const Output<ngraph::Node>)>;

            ///
            /// \brief      This class describes a recurrent sequence.
            ///
            class NGRAPH_API RecurrentSequence
            {
            public:
                ///
                /// \brief      Constructs a RecurrentSequence class object.
                ///
                /// \param[in]  args       The map with recurrent sequence operator inputs.
                /// \param[in]  attrs      The structure containing operator attributes.
                /// \param[in]  direction  The sequence direction mode {FORWARD, REVERSE,
                ///                        BIDIRECTIONAL}.
                ///
                RecurrentSequence(OpInputMap& args,
                                  ngraph::op::RecurrentSequenceDirection direction);

                ///
                /// \brief      Carry out all steps of recurrent sequence with provided cell kernel.
                ///
                /// \param[in]  kernel  The cell kernel function.
                ///
                /// \return     The node vector containing results from all sequence steps.
                ///
                NodeVector run_sequence(const RecurrentCellFunction& kernel);

            private:
                ///
                /// \brief      Gets the masked value according to sequence lenght in a batch.
                ///
                /// \note       Zeros out values or sets them to default value for inputs with
                ///             sequence lenght shorter than currently procssed time step.
                ///
                /// \param[in]  data           The input value.
                /// \param[in]  time_step      The current time step denoting sequence lenght.
                /// \param[in]  batch_axis     The batch axis index of data tensor.
                /// \param[in]  default_value  The default value for masked elements.
                ///
                /// \return     The masked value.
                ///
                std::shared_ptr<ngraph::Node> get_masked_node(
                    const Output<ngraph::Node>& data,
                    std::int32_t time_step,
                    std::size_t batch_axis = 0,
                    const Output<ngraph::Node>& default_value = Output<ngraph::Node>()) const;

                ///
                /// \brief      Split and squeeze input data to remove 'num_direction' dimension.
                ///
                /// \param[in]  node        The node to update.
                /// \param[in]  is_reverse  Indicates if configure to reverse pass.
                ///
                /// \return     Updated node for forward/reverse pass.
                ///
                std::shared_ptr<ngraph::Node> prepare_input(Output<ngraph::Node> node,
                                                            bool is_reverse) const;

                ///
                /// \brief      Perform computation through all input sequence steps in single mode.
                ///
                /// \param[in]  kernel      The cell kernel function.
                /// \param[in]  is_reverse  Indicates if carry out reverse or forward pass.
                ///
                /// \return     The node vector with pass results.
                ///
                NodeVector recurrent_sequence_pass(const RecurrentCellFunction& kernel,
                                                   bool is_reverse = false);

                OpInputMap& m_args;
                ngraph::op::RecurrentSequenceDirection m_direction;
            };

        } // recurrent
    }     // onnx_import
} // ngraph
