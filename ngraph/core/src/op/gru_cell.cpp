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

#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/gru_cell.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

constexpr NodeTypeInfo op::v3::GRUCell::type_info;

op::v3::GRUCell::GRUCell()
    : m_linear_before_reset(false)
{
    m_activations = {"sigmoid", "tanh"};
    m_activation_f = get_activation_function(0);
    m_activation_g = get_activation_function(1);
}

op::v3::GRUCell::GRUCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         size_t hidden_size)
    : GRUCell(X,
              initial_hidden_state,
              W,
              R,
              hidden_size,
              vector<string>{"sigmoid", "tanh"},
              vector<float>{},
              vector<float>{},
              0.f,
              false)
{
}

op::v3::GRUCell::GRUCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         size_t hidden_size,
                         const vector<string>& activations,
                         const vector<float>& activations_alpha,
                         const vector<float>& activations_beta,
                         float clip,
                         bool linear_before_reset)
    : FusedOp({X, initial_hidden_state, W, R})
    , RNNCellBase(hidden_size, clip, activations, activations_alpha, activations_beta)
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_linear_before_reset{linear_before_reset}
{
    add_default_bias_input();
    constructor_validate_and_infer_types();
}

op::v3::GRUCell::GRUCell(const Output<Node>& X,
                         const Output<Node>& initial_hidden_state,
                         const Output<Node>& W,
                         const Output<Node>& R,
                         const Output<Node>& B,
                         size_t hidden_size,
                         const vector<string>& activations,
                         const vector<float>& activations_alpha,
                         const vector<float>& activations_beta,
                         float clip,
                         bool linear_before_reset)
    : FusedOp({X, initial_hidden_state, W, R, B})
    , RNNCellBase(hidden_size, clip, activations, activations_alpha, activations_beta)
    , m_activation_f{get_activation_function(0)}
    , m_activation_g{get_activation_function(1)}
    , m_linear_before_reset{linear_before_reset}
{
    constructor_validate_and_infer_types();
}

bool op::v3::GRUCell::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("linear_before_reset", m_linear_before_reset);
    return op::util::RNNCellBase::visit_attributes(visitor);
}

void op::v3::GRUCell::pre_validate_and_infer_types()
{
    if (is_dynamic())
    {
        return;
    }

    const auto& x_pshape = get_input_partial_shape(0);
    const auto& ht_pshape = get_input_partial_shape(1);
    const auto& w_pshape = get_input_partial_shape(2);
    const auto& r_pshape = get_input_partial_shape(3);
    const auto& b_pshape = get_input_partial_shape(4);

    const Shape& x_shape{x_pshape.to_shape()};

    const size_t batch_size = x_shape.at(0);
    const size_t input_size = x_shape.at(1);

    const Shape& w_shape{w_pshape.to_shape()};
    const Shape& r_shape{r_pshape.to_shape()};
    const Shape& ht_shape{ht_pshape.to_shape()};

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
                          w_shape,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          (ht_shape == Shape{batch_size, get_hidden_size()}),
                          "Input tensor initial_hidden_state must have shape (",
                          batch_size,
                          ", ",
                          get_hidden_size(),
                          "). Actual shape is:",
                          w_shape,
                          ".");

    const Shape& b_shape{b_pshape.to_shape()};
    NODE_VALIDATION_CHECK(
        this,
        (b_shape == Shape{(s_gates_count + m_linear_before_reset) * get_hidden_size()}),
        "Input tensor B must have shape (",
        (s_gates_count + m_linear_before_reset) * get_hidden_size(),
        "). Actual shape is:",
        b_shape,
        ".");
}

OutputVector op::v3::GRUCell::decompose_op() const
{
    // ------ VARIABLE'S NAMES AND ACRONYM DEFINITIONS ------
    // The names used below are analogous to the one used in ONNX documentation.
    //
    // ------ ACRONYMS ------
    // z_t - update gate at current time step
    // r_t - reset gate at current time step
    // h_t - hidden gate at current time step
    // t - time step (t-1 means previous time step)
    // X        The input data tensor. Shape: [batch_size, input_size].
    // W[zrh] - The weight tensor for update, reset and hidden gates.
    //          Shape: [gates_count * hidden_size, input_size].
    // R[zrh] - The recurrence weight tensor for update, reset and hidden gates.
    //          Shape: [gates_count * hidden_size, hidden_size].
    // H_t    - The hidden state tensor at current time step. Shape: [batch_size, hidden_size].
    // B      - The sum of biases (weight and recurrence) for update, reset and hidden gates.
    //          If linear_before_reset := true then biases for hidden gates are placed separately
    //          (weight and recurrence).
    //          Shape: [gates_count * hidden_size] when linear_before_reset := false
    //          Shape: [(gates_count + 1) * hidden_size] when linear_before_reset := true
    // Wb[zrh] - W bias vectors for update, reset and hidden gates.
    // Rb[zrh] - R bias vectors for update, reset and hidden gates.

    // (.) - Denotes element-wise multiplication.
    // *   - Denotes dot product.

    // ---- Equations ----
    // f, g  - are activation functions
    // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # when linear_before_reset := false
    //                                                      # (default)
    // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset := true
    // Ht = (1 - zt) (.) ht + zt (.) Ht-1
    // -------------------

    Output<Node> X = input_value(0);
    Output<Node> H_t = input_value(1);
    Output<Node> W = input_value(2);
    Output<Node> R = input_value(3);
    Output<Node> B = input_value(4);

    // Xt*(W^T)
    auto Xt_W = make_shared<op::Dot>(X, builder::opset1::transpose(W));
    auto R_transpose = builder::opset1::transpose(R);
    // Ht-1*(R^T)
    auto Ht_R = make_shared<op::Dot>(H_t, R_transpose);

    // split to gates:
    OutputVector Xt_W_zrh = builder::split(Xt_W, 3, 1);
    OutputVector R_zrh = builder::split(R_transpose, 3, 1);
    OutputVector Ht_R_zrh = builder::split(Ht_R, 3, 1);
    OutputVector biases_zrh = m_linear_before_reset ? builder::split(B, 4) : builder::split(B, 3);

    // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    auto z_t = m_activation_f(clip(add(Xt_W_zrh[0], add(Ht_R_zrh[0], biases_zrh[0]))));
    // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    auto r_t = m_activation_f(clip(add(Xt_W_zrh[1], add(Ht_R_zrh[1], biases_zrh[1]))));

    Output<Node> h_t;
    if (m_linear_before_reset)
    {
        // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
        auto Ht_Rh_Rbh = add(Ht_R_zrh[2], biases_zrh[3]);
        h_t = m_activation_g(clip(add(Xt_W_zrh[2], add(mul(r_t, Ht_Rh_Rbh), biases_zrh[2]))));
    }
    else
    {
        // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
        auto rt_Ht = mul(r_t, H_t);
        auto rt_Ht_Rh = make_shared<op::Dot>(rt_Ht, R_zrh[2]);
        // Tensor shape: [batch_size, hidden_size]
        h_t = m_activation_g(clip(add(Xt_W_zrh[2], add(rt_Ht_Rh, biases_zrh[2]))));
    }

    auto one = op::Constant::create(z_t->get_element_type(),
                                    z_t->get_shape(),
                                    vector<float>(shape_size(z_t->get_shape()), 1.f));
    // Ht = (1 - zt) (.) ht + zt (.) Ht-1
    H_t = add(mul(sub(one, z_t), h_t), mul(z_t, H_t));
    return {H_t.get_node_shared_ptr()};
}

void op::v3::GRUCell::add_default_bias_input()
{
    Output<Node> B = op::Constant::create(
        get_input_element_type(0),
        Shape{(s_gates_count + m_linear_before_reset) * get_hidden_size()},
        vector<float>((s_gates_count + m_linear_before_reset) * get_hidden_size(), 0.f));
    set_argument(4, B);
}

shared_ptr<Node> op::v3::GRUCell::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 4)
    {
        return make_shared<GRUCell>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    get_hidden_size(),
                                    get_activations(),
                                    get_activations_alpha(),
                                    get_activations_beta(),
                                    get_clip(),
                                    m_linear_before_reset);
    }
    else if (new_args.size() == 5)
    {
        return make_shared<GRUCell>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    new_args.at(4),
                                    get_hidden_size(),
                                    get_activations(),
                                    get_activations_alpha(),
                                    get_activations_beta(),
                                    get_clip(),
                                    m_linear_before_reset);
    }
    else
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
}
