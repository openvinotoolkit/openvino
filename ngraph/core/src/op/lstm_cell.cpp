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

#include <cmath>
#include <functional>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/lstm_cell.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

constexpr NodeTypeInfo op::LSTMCell::type_info;

op::LSTMCell::LSTMCell()
    : m_input_forget(false)
    , m_weights_format(LSTMWeightsFormat::IFCO)
{
    m_activations = {"sigmoid", "tanh", "tanh"};
    m_activation_f = get_activation_function(0);
    m_activation_g = get_activation_function(1);
    m_activation_h = get_activation_function(2);
}

op::LSTMCell::LSTMCell(const Output<Node>& X,
                       const Output<Node>& initial_hidden_state,
                       const Output<Node>& initial_cell_state,
                       const Output<Node>& W,
                       const Output<Node>& R,
                       size_t hidden_size,
                       op::LSTMWeightsFormat weights_format,
                       const vector<string>& activations,
                       const vector<float>& activations_alpha,
                       const vector<float>& activations_beta,
                       float clip,
                       bool input_forget)
    : FusedOp({X, initial_hidden_state, initial_cell_state, W, R})
    , RNNCellBase(hidden_size, clip, activations, activations_alpha, activations_beta)
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_activation_h{get_activation_function(2)}
    , m_input_forget{input_forget}
    , m_weights_format{weights_format}
{
    set_argument(5, get_default_bias_input());
    set_argument(6, get_default_peepholes_input());
    constructor_validate_and_infer_types();
}

op::LSTMCell::LSTMCell(const Output<Node>& X,
                       const Output<Node>& initial_hidden_state,
                       const Output<Node>& initial_cell_state,
                       const Output<Node>& W,
                       const Output<Node>& R,
                       const Output<Node>& B,
                       size_t hidden_size,
                       op::LSTMWeightsFormat weights_format,
                       const vector<string>& activations,
                       const vector<float>& activations_alpha,
                       const vector<float>& activations_beta,
                       float clip,
                       bool input_forget)
    : FusedOp({X, initial_hidden_state, initial_cell_state, W, R, B})
    , RNNCellBase(hidden_size, clip, activations, activations_alpha, activations_beta)
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_activation_h{get_activation_function(2)}
    , m_input_forget{input_forget}
    , m_weights_format{weights_format}
{
    set_argument(6, get_default_peepholes_input());
    constructor_validate_and_infer_types();
}

op::LSTMCell::LSTMCell(const Output<Node>& X,
                       const Output<Node>& initial_hidden_state,
                       const Output<Node>& initial_cell_state,
                       const Output<Node>& W,
                       const Output<Node>& R,
                       const Output<Node>& B,
                       const Output<Node>& P,
                       size_t hidden_size,
                       op::LSTMWeightsFormat weights_format,
                       const vector<string>& activations,
                       const vector<float>& activations_alpha,
                       const vector<float>& activations_beta,
                       float clip,
                       bool input_forget)
    : FusedOp({X, initial_hidden_state, initial_cell_state, W, R, B, P})
    , RNNCellBase(hidden_size, clip, activations, activations_alpha, activations_beta)
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_activation_h{get_activation_function(2)}
    , m_input_forget{input_forget}
    , m_weights_format{weights_format}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::LSTMCell::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("hidden_size", m_hidden_size);
    visitor.on_attribute("activations", m_activations);
    visitor.on_attribute("activations_alpha", m_activations_alpha);
    visitor.on_attribute("activations_beta", m_activations_beta);
    visitor.on_attribute("clip", m_clip);

    visitor.on_attribute("input_forget", m_input_forget);
    visitor.on_attribute("weights_format", m_weights_format);
    return true;
}

void op::LSTMCell::pre_validate_and_infer_types()
{
    set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    set_output_type(1, get_input_element_type(0), PartialShape::dynamic());
    if (is_dynamic())
    {
        return;
    }

    const auto& x_pshape = get_input_partial_shape(0);
    const auto& ht_pshape = get_input_partial_shape(1);
    const auto& ct_pshape = get_input_partial_shape(2);
    const auto& w_pshape = get_input_partial_shape(3);
    const auto& r_pshape = get_input_partial_shape(4);

    NODE_VALIDATION_CHECK(this,
                          (x_pshape.is_static() || w_pshape.is_static() || r_pshape.is_static() ||
                           ht_pshape.is_static() || ct_pshape.is_static()),
                          "LSTMCell supports only static input tensors.");

    const Shape& x_shape{x_pshape.to_shape()};

    const size_t batch_size = x_shape.at(0);
    const size_t input_size = x_shape.at(1);

    const Shape& w_shape{w_pshape.to_shape()};
    const Shape& r_shape{r_pshape.to_shape()};
    const Shape& ht_shape{ht_pshape.to_shape()};
    const Shape& ct_shape{ct_pshape.to_shape()};

    NODE_VALIDATION_CHECK(this,
                          (w_shape == Shape{s_gates_count * get_hidden_size(), input_size}),
                          "Input tensor W must have shape (",
                          s_gates_count * get_hidden_size(),
                          ", ",
                          input_size,
                          "). Actual shape is:",
                          w_shape,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          (r_shape == Shape{s_gates_count * get_hidden_size(), get_hidden_size()}),
                          "Input tensor R must have shape (",
                          s_gates_count * get_hidden_size(),
                          ", ",
                          get_hidden_size(),
                          "). Actual shape is:",
                          r_shape,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          (ht_shape == Shape{batch_size, get_hidden_size()}),
                          "Input tensor initial_hidden_state must have shape (",
                          batch_size,
                          ", ",
                          get_hidden_size(),
                          "). Actual shape is:",
                          ht_shape,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          (ct_shape == Shape{batch_size, get_hidden_size()}),
                          "Input tensor initial_cell_state must have shape (",
                          batch_size,
                          ", ",
                          get_hidden_size(),
                          "). Actual shape is:",
                          ct_shape,
                          ".");

    const auto& b_pshape = get_input_partial_shape(5);
    const auto& p_pshape = get_input_partial_shape(6);

    NODE_VALIDATION_CHECK(this,
                          (b_pshape.is_static() || p_pshape.is_static()),
                          "LSTMCell supports only static input tensors.");

    const Shape& b_shape{b_pshape.to_shape()};
    const Shape& p_shape{p_pshape.to_shape()};

    NODE_VALIDATION_CHECK(this,
                          (b_shape == Shape{s_gates_count * get_hidden_size()}),
                          "Input tensor B must have shape (",
                          s_gates_count * get_hidden_size(),
                          "). Actual shape is:",
                          b_shape,
                          ".");

    NODE_VALIDATION_CHECK(this,
                          (p_shape == Shape{s_peepholes_count * get_hidden_size()}),
                          "Input tensor P must have shape (",
                          s_peepholes_count * get_hidden_size(),
                          "). Actual shape is:",
                          p_shape,
                          ".");
}

OutputVector op::LSTMCell::decompose_op() const
{
    // ------ VARIABLE'S NAMES AND ACRONYM DEFINITIONS ------
    // The names used below are analogous to the one used in ONNX documentation.
    //
    // ------ ACRONYMS ------
    // i - input gate
    // o - output gate
    // f - forget gate
    // c - cell gate
    // t - time step (t-1 means previous time step)
    // Wb - W bias vectors for input, output, forget, and cell gates.
    // Rb - R bias vectors for input, output, forget, and cell gates.
    // P  - The peephole weights for input, output and forget gates.
    // ------ VARIABLE NAMES ------
    // X       - The input data tensor. Shape: [batch_size, input_size].
    // W       - The weight matrix for input, forget, cell and output gates
    //           Shape: [4*hidden_size, input_size]
    // R       - The recurrence weight matrix for input, forget, cell and output gates.
    //           Shape: [4*hidden_size, hidden_size].
    // H_t     - The hidden state tensor at current time step. Shape: [batch_size, hidden_size].
    // C_t     - The cell state tensor at current time step. Shape: [batch_size, hidden_size].
    // bias    - The sum of biases (weight and recurrence) for input, forget, cell and output gates.
    //           Shape: [4 * hidden_size]
    // p_[iof] - The peephole weight vector for respectively: input, output, and forget gates.
    //           Each peephole has shape [hidden_size].
    //
    // (.) - Denotes element-wise multiplication.
    // *   - Denotes dot product.
    //
    // ---- Equations ----
    // f, g, h - are activation functions.
    // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    // Ct = ft (.) Ct-1 + it (.) ct
    // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    // Ht = ot (.) h(Ct)
    // --------------------

    Output<Node> X = input_value(0);
    Output<Node> H_t = input_value(1);
    Output<Node> C_t = input_value(2);
    Output<Node> W = input_value(3);
    Output<Node> R = input_value(4);
    Output<Node> bias = input_value(5);
    OutputVector p_iof = builder::split(input_value(6), s_peepholes_count);

    // Converting to IFCO format since it's DNNL default.
    if (m_weights_format != op::LSTMWeightsFormat::IFCO)
    {
        W = convert_node_format(W);
        R = convert_node_format(R);
        bias = convert_node_format(bias);
    }

    const auto& p_i = p_iof.at(0);
    const auto& p_o = p_iof.at(1);
    const auto& p_f = p_iof.at(2);

    // Xt*(W^T) -- for [iofc] gates.
    auto Xt_W = make_shared<op::Dot>(X, builder::opset1::transpose(W));
    // Ht-1*(R^T)  -- for [iofc] gates.
    auto Ht_R = make_shared<op::Dot>(H_t, builder::opset1::transpose(R));
    // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb  -- for [iofc] gates.
    auto gates = add(Xt_W, add(Ht_R, bias));

    OutputVector split_gates = builder::split(gates, 4, -1);
    auto i_t = split_gates.at(0);
    auto f_t = split_gates.at(1);
    auto c_t = split_gates.at(2);
    auto o_t = split_gates.at(3);

    // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    i_t = m_activation_f(clip(add(i_t, mul(p_i, C_t))));
    if (m_input_forget)
    {
        // Couple input with forget gate: 1 - i_t
        f_t = sub(op::Constant::create(i_t.get_element_type(),
                                       i_t.get_shape(),
                                       vector<float>(shape_size(i_t.get_shape()), 1.f)),
                  i_t);
    }
    else
    {
        // f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        f_t = m_activation_f(clip(add(f_t, mul(p_f, C_t))));
    }
    // ft (.) Ct-1 + it (.) ct
    auto C = add(mul(f_t, C_t), mul(i_t, m_activation_g(clip(c_t))));
    // f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    o_t = m_activation_f(clip(add(o_t, mul(p_o, C))));
    // ot (.) h(Ct)
    auto H = mul(o_t, m_activation_h(clip(C)));

    return {H, C};
}

Output<Node> op::LSTMCell::get_default_bias_input() const
{
    return Output<Node>{op::Constant::create(
        get_input_element_type(0), Shape{s_gates_count * get_hidden_size()}, vector<float>{0.f})};
}

Output<Node> op::LSTMCell::get_default_peepholes_input() const
{
    return Output<Node>{op::Constant::create(get_input_element_type(0),
                                             Shape{s_peepholes_count * get_hidden_size()},
                                             vector<float>{0.f})};
}

shared_ptr<Node> op::LSTMCell::convert_node_format(const Output<Node>& node) const
{
    static const std::map<op::LSTMWeightsFormat, std::vector<size_t>> gate_order_conversion_map{
        {op::LSTMWeightsFormat::FICO, {1, 0, 2, 3}},
        {op::LSTMWeightsFormat::ICOF, {0, 3, 1, 2}},
        {op::LSTMWeightsFormat::IFOC, {0, 1, 3, 2}},
        {op::LSTMWeightsFormat::IOFC, {0, 2, 3, 1}},
    };

    OutputVector splitted_node = builder::split(node, s_gates_count);
    OutputVector nodes_in_new_format;
    nodes_in_new_format.reserve(s_gates_count);
    for (const auto& axis : gate_order_conversion_map.at(m_weights_format))
    {
        nodes_in_new_format.push_back(splitted_node.at(axis));
    }
    return make_shared<op::Concat>(nodes_in_new_format, 0);
}

shared_ptr<Node> op::LSTMCell::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 5)
    {
        return make_shared<LSTMCell>(new_args.at(0),
                                     new_args.at(1),
                                     new_args.at(2),
                                     new_args.at(3),
                                     new_args.at(4),
                                     get_hidden_size(),
                                     get_weights_format(),
                                     get_activations(),
                                     get_activations_alpha(),
                                     get_activations_beta(),
                                     get_clip(),
                                     m_input_forget);
    }
    else if (new_args.size() == 6)
    {
        return make_shared<LSTMCell>(new_args.at(0),
                                     new_args.at(1),
                                     new_args.at(2),
                                     new_args.at(3),
                                     new_args.at(4),
                                     new_args.at(5),
                                     get_hidden_size(),
                                     get_weights_format(),
                                     get_activations(),
                                     get_activations_alpha(),
                                     get_activations_beta(),
                                     get_clip(),
                                     m_input_forget);
    }
    else if (new_args.size() == 7)
    {
        return make_shared<LSTMCell>(new_args.at(0),
                                     new_args.at(1),
                                     new_args.at(2),
                                     new_args.at(3),
                                     new_args.at(4),
                                     new_args.at(5),
                                     new_args.at(6),
                                     get_hidden_size(),
                                     get_weights_format(),
                                     get_activations(),
                                     get_activations_alpha(),
                                     get_activations_beta(),
                                     get_clip(),
                                     m_input_forget);
    }
    else
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
}

namespace ngraph
{
    template <>
    EnumNames<op::LSTMWeightsFormat>& EnumNames<op::LSTMWeightsFormat>::get()
    {
        static auto enum_names =
            EnumNames<op::LSTMWeightsFormat>("op::LSTMWeightsFormat",
                                             {{"fico", op::LSTMWeightsFormat::FICO},
                                              {"icof", op::LSTMWeightsFormat::ICOF},
                                              {"ifco", op::LSTMWeightsFormat::IFCO},
                                              {"ifoc", op::LSTMWeightsFormat::IFOC},
                                              {"iofc", op::LSTMWeightsFormat::IOFC}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::LSTMWeightsFormat>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::LSTMWeightsFormat& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph
