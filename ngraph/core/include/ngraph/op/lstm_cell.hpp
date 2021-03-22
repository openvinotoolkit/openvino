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

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/util/activation_functions.hpp"
#include "ngraph/op/util/fused_op.hpp"
#include "ngraph/op/util/rnn_cell_base.hpp"

namespace ngraph
{
    namespace op
    {
        enum class LSTMWeightsFormat
        {
            FICO, // IE
            ICOF, // PyTorch
            IFCO, // DNNL, TF, MxNet
            IFOC, // Caffe
            IOFC, // ONNX
        };

        namespace v0
        {
            ///
            /// \brief      Class for single lstm cell node.
            ///
            /// \note       Following implementation supports:
            ///             \li \c peepholes Gers & Schmidhuber (2000)
            ///             https://ieeexplore.ieee.org/document/861302
            ///             \li Coupling input and forget gates.
            ///
            /// \note       It calculates following equations:
            ///
            ///             it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
            ///             ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
            ///             ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
            ///             Ct = ft (.) Ct-1 + it (.) ct
            ///             ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
            ///             Ht = ot (.) h(Ct)
            ///
            ///             *       - Is a dot product,
            ///             (.)     - is a Hadamard product (element-wise),
            ///             f, g, h - are activation functions.
            ///
            /// \note       This class represents only single *cell* (for current time step) and not
            ///             the whole LSTM Sequence layer
            ///
            /// \sa         LSTMSequence, RNNCell, GRUCell
            ///
            class NGRAPH_API LSTMCell : public util::RNNCellBase
            {
            public:
                static constexpr NodeTypeInfo type_info{"LSTMCell", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                LSTMCell();
                ///
                /// \brief      Constructs LSTMCell node.
                ///
                /// \param[in]  X                     The input tensor with shape: [batch_size,
                ///                                   input_size].
                /// \param[in]  initial_hidden_state  The hidden state tensor at current time step
                ///                                   with shape: [batch_size, hidden_size].
                /// \param[in]  initial_cell_state    The cell state tensor at current time step
                ///                                   with shape: [batch_size, hidden_size].
                /// \param[in]  W                     The gate weights tensor with shape:
                ///                                   [4*hidden_size, input_size].
                /// \param[in]  R                     The recurrence weights tensor with shape:
                ///                                   [4*hidden_size, hidden_size].
                /// \param[in]  hidden_size           The number of hidden units for recurrent cell.
                /// \param[in]  weights_format        The order of gates in weights tensors. The
                ///                                   default format is IFCO since it is used by
                ///                                   DNNL.
                /// \param[in]  activations           The vector of activation functions used inside
                ///                                   recurrent cell.
                /// \param[in]  activations_alpha     The vector of alpha parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  activations_beta      The vector of beta parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  clip                  The value defining clipping range [-clip,
                ///                                   clip] on input of activation functions.
                /// \param[in]  input_forget          Controls coupling input and forget gates.
                ///
                LSTMCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& initial_cell_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         std::size_t hidden_size,
                         LSTMWeightsFormat weights_format = LSTMWeightsFormat::IFCO,
                         const std::vector<std::string>& activations =
                             std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                         const std::vector<float>& activations_alpha = {},
                         const std::vector<float>& activations_beta = {},
                         float clip = 0.f,
                         bool input_forget = false);

                ///
                /// \brief      Constructs LSTMCell node.
                ///
                /// \param[in]  X                     The input tensor with shape: [batch_size,
                ///                                   input_size].
                /// \param[in]  initial_hidden_state  The hidden state tensor at current time step
                ///                                   with shape: [batch_size, hidden_size].
                /// \param[in]  initial_cell_state    The cell state tensor at current time step
                ///                                   with shape: [batch_size, hidden_size].
                /// \param[in]  W                     The weight tensor with shape: [4*hidden_size,
                ///                                   input_size].
                /// \param[in]  R                     The recurrence weight tensor with shape:
                ///                                   [4*hidden_size, hidden_size].
                /// \param[in]  B                     The bias tensor for gates with shape:
                ///                                   [4*hidden_size].
                /// \param[in]  hidden_size           The number of hidden units for recurrent cell.
                /// \param[in]  weights_format        The order of gates in weights tensors. The
                ///                                   default format is IFCO since it is used by
                ///                                   DNNL.
                /// \param[in]  activations           The vector of activation functions used inside
                ///                                   recurrent cell.
                /// \param[in]  activations_alpha     The vector of alpha parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  activations_beta      The vector of beta parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  clip                  The value defining clipping range [-clip,
                ///                                   clip] on input of activation functions.
                /// \param[in]  input_forget          Controls coupling input and forget gates.
                ///
                LSTMCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& initial_cell_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         const Output<Node>& B,
                         std::size_t hidden_size,
                         LSTMWeightsFormat weights_format = LSTMWeightsFormat::IFCO,
                         const std::vector<std::string>& activations =
                             std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                         const std::vector<float>& activations_alpha = {},
                         const std::vector<float>& activations_beta = {},
                         float clip = 0.f,
                         bool input_forget = false);

                ///
                /// \brief      Constructs LSTMCell node.
                ///
                /// \param[in]  X                     The input tensor with shape: [batch_size,
                ///                                   input_size].
                /// \param[in]  initial_hidden_state  The hidden state tensor at current time step
                ///                                   with shape: [batch_size, hidden_size].
                /// \param[in]  initial_cell_state    The cell state tensor at current time step
                ///                                   with shape: [batch_size, hidden_size].
                /// \param[in]  W                     The weight tensor with shape: [4*hidden_size,
                ///                                   input_size].
                /// \param[in]  R                     The recurrence weight tensor with shape:
                ///                                   [4*hidden_size, hidden_size].
                /// \param[in]  B                     The bias tensor for gates with shape:
                ///                                   [4*hidden_size].
                /// \param[in]  P                     The weight tensor for peepholes with shape:
                ///                                   [3*hidden_size] - 3 equals to only iof gates.
                ///                                   The order is: input, output, forget gates.
                /// \param[in]  hidden_size           The number of hidden units for recurrent cell.
                /// \param[in]  weights_format        The order of gates in weights tensors. The
                ///                                   default format is IFCO since it is used by
                ///                                   DNNL.
                /// \param[in]  activations           The vector of activation functions used inside
                ///                                   recurrent cell.
                /// \param[in]  activations_alpha     The vector of alpha parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  activations_beta      The vector of beta parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  clip                  The value defining clipping range [-clip,
                ///                                   clip] on input of activation functions.
                /// \param[in]  input_forget          Controls coupling input and forget gates.
                ///
                LSTMCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& initial_cell_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         const Output<Node>& B,
                         const Output<Node>& P,
                         std::size_t hidden_size,
                         LSTMWeightsFormat weights_format = LSTMWeightsFormat::IFCO,
                         const std::vector<std::string>& activations =
                             std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                         const std::vector<float>& activations_alpha = {},
                         const std::vector<float>& activations_beta = {},
                         float clip = 0.f,
                         bool input_forget = false);

                virtual void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool get_input_forget() const { return m_input_forget; }
                LSTMWeightsFormat get_weights_format() const { return m_weights_format; }

            private:
                ///
                /// \brief      Creates the default bias input initialized with zeros.
                ///
                /// \return     The object of Output class.
                ///
                Output<Node> get_default_bias_input() const;

                ///
                /// \brief      Creates the default peepholes input initialized with zeros.
                ///
                /// \return     The object of Output class.
                ///
                Output<Node> get_default_peepholes_input() const;
                ///
                /// \brief The Activation function f.
                ///
                util::ActivationFunction m_activation_f;
                ///
                /// \brief The Activation function g.
                ///
                util::ActivationFunction m_activation_g;
                ///
                /// \brief The Activation function h.
                ///
                util::ActivationFunction m_activation_h;
                ///
                /// \brief      Controls whether to couple input and forget gates.
                ///
                bool m_input_forget = false;

                ///
                /// \brief The order of gates in weights tensors.
                ///
                LSTMWeightsFormat m_weights_format;

                static constexpr std::size_t s_gates_count{4};
                static constexpr std::size_t s_peepholes_count{3};
            };
        } // v0

        namespace v4
        {
            ///
            /// \brief      Class for single lstm cell node.
            ///
            /// \note       Following implementation supports:
            ///             \li \c peepholes Gers & Schmidhuber (2000)
            ///             https://ieeexplore.ieee.org/document/861302
            ///             \li Coupling input and forget gates.
            ///
            /// \note       It calculates following equations:
            ///
            ///             it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
            ///             ft = f(Xt*(Wf^T) + Ht-1*(Rf^T)  + Wbf + Rbf)
            ///             ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
            ///             Ct = ft (.) Ct-1 + it (.) ct
            ///             ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
            ///             Ht = ot (.) h(Ct)
            ///
            ///             *       - Is a dot product,
            ///             (.)     - is a Hadamard product (element-wise),
            ///             f, g, h - are activation functions.
            ///
            /// \note       This class represents only single *cell* (for current time step) and not
            ///             the whole LSTM Sequence layer
            ///
            /// \sa         LSTMSequence, RNNCell, GRUCell
            ///
            class NGRAPH_API LSTMCell : public util::RNNCellBase
            {
            public:
                static constexpr NodeTypeInfo type_info{"LSTMCell", 4};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                LSTMCell();
                ///
                /// \brief      Constructs LSTMCell node.
                ///
                /// \param[in]  X                     The input tensor with shape: [batch_size,
                ///                                   input_size].
                /// \param[in]  initial_hidden_state  The hidden state tensor at current time step
                ///                                   with shape: [batch_size, hidden_size].
                /// \param[in]  initial_cell_state    The cell state tensor at current time step
                ///                                   with shape: [batch_size, hidden_size].
                /// \param[in]  W                     The gate weights tensor with shape:
                ///                                   [4*hidden_size, input_size].
                /// \param[in]  R                     The recurrence weights tensor with shape:
                ///                                   [4*hidden_size, hidden_size].
                /// \param[in]  hidden_size           The number of hidden units for recurrent cell.
                /// \param[in]  activations           The vector of activation functions used inside
                ///                                   recurrent cell.
                /// \param[in]  activations_alpha     The vector of alpha parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  activations_beta      The vector of beta parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  clip                  The value defining clipping range [-clip,
                ///                                   clip] on input of activation functions.
                LSTMCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& initial_cell_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         std::size_t hidden_size,
                         const std::vector<std::string>& activations =
                             std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                         const std::vector<float>& activations_alpha = {},
                         const std::vector<float>& activations_beta = {},
                         float clip = 0.f);

                ///
                /// \brief      Constructs LSTMCell node.
                ///
                /// \param[in]  X                     The input tensor with shape: [batch_size,
                ///                                   input_size].
                /// \param[in]  initial_hidden_state  The hidden state tensor at current time step
                ///                                   with shape: [batch_size, hidden_size].
                /// \param[in]  initial_cell_state    The cell state tensor at current time step
                ///                                   with shape: [batch_size, hidden_size].
                /// \param[in]  W                     The weight tensor with shape: [4*hidden_size,
                ///                                   input_size].
                /// \param[in]  R                     The recurrence weight tensor with shape:
                ///                                   [4*hidden_size, hidden_size].
                /// \param[in]  B                     The bias tensor for gates with shape:
                ///                                   [4*hidden_size].
                /// \param[in]  hidden_size           The number of hidden units for recurrent cell.
                /// \param[in]  activations           The vector of activation functions used inside
                ///                                   recurrent cell.
                /// \param[in]  activations_alpha     The vector of alpha parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  activations_beta      The vector of beta parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  clip                  The value defining clipping range [-clip,
                ///                                   clip] on input of activation functions.
                ///
                LSTMCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& initial_cell_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         const Output<Node>& B,
                         std::size_t hidden_size,
                         const std::vector<std::string>& activations =
                             std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                         const std::vector<float>& activations_alpha = {},
                         const std::vector<float>& activations_beta = {},
                         float clip = 0.f);

                void validate_and_infer_types() override;

                bool visit_attributes(AttributeVisitor& visitor) override;
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

            private:
                ///
                /// \brief      Creates the default bias input initialized with zeros.
                ///
                /// \return     The object of Output class.
                ///
                Output<Node> get_default_bias_input() const;

                ///
                /// \brief The Activation function f.
                ///
                util::ActivationFunction m_activation_f;
                ///
                /// \brief The Activation function g.
                ///
                util::ActivationFunction m_activation_g;
                ///
                /// \brief The Activation function h.
                ///
                util::ActivationFunction m_activation_h;

                static constexpr std::size_t s_gates_count{4};
            };
        } // v4
    }     // namespace op

    NGRAPH_API
    std::ostream& operator<<(std::ostream& s, const op::LSTMWeightsFormat& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::LSTMWeightsFormat>
        : public EnumAttributeAdapterBase<op::LSTMWeightsFormat>
    {
    public:
        AttributeAdapter(op::LSTMWeightsFormat& value)
            : EnumAttributeAdapterBase<op::LSTMWeightsFormat>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<op::LSTMWeightsFormat>", 1};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
} // namespace ngraph
