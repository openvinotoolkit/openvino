// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/op/lstm_cell.hpp"
#include "openvino/reference/add.hpp"
#include "openvino/reference/clamp.hpp"
#include "openvino/reference/matmul.hpp"
#include "openvino/reference/multiply.hpp"
#include "openvino/reference/relu.hpp"
#include "openvino/reference/sigmoid.hpp"
#include "openvino/reference/split.hpp"
#include "openvino/reference/subtract.hpp"
#include "openvino/reference/tanh.hpp"

namespace ov {
namespace reference {
template <typename T>
void lstm_cell(const T* X,
               const Shape& X_shape,
               const T* H,
               const Shape& H_shape,
               const T* C,
               const Shape& C_shape,
               const T* W,
               const Shape& W_shape,
               const T* R,
               const Shape& R_shape,
               const T* B,
               const Shape& B_shape,
               T* out_Ht,
               T* out_Ct,
               const std::string& activation_f,
               const std::string& activation_g,
               const std::string& activation_h,
               float clip) {
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
    // H_t     - The hidden state tensor at current time step. Shape: [batch_size,
    // hidden_size].
    // C_t     - The cell state tensor at current time step. Shape: [batch_size,
    // hidden_size].
    // bias    - The sum of biases (weight and recurrence) for input, forget, cell and
    // output gates.
    //           Shape: [4 * hidden_size]
    // p_[iof] - The peephole weight vector for respectively: input, output, and forget
    // gates.
    //           Each peephole has shape [hidden_size].
    //
    // (.) - Denotes element-wise multiplication.
    // *   - Denotes dot product.
    //
    // ---- Equations ----
    // f, g, h - are activation functions.
    // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
    // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
    // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
    // Ct = ft (.) Ct-1 + it (.) ct
    // Ht = ot (.) h(Ct)
    // --------------------
    Shape gate_shape{X_shape[0], H_shape[1]};
    Shape all_gates_shape{X_shape[0], 4 * H_shape[1]};
    auto gate_shape_size = X_shape[0] * H_shape[1];
    auto all_gates_shape_size = gate_shape_size * 4;
    // Xt*(W^T)
    std::vector<T> Xt_W(all_gates_shape_size);
    reference::matmul(X, W, Xt_W.data(), X_shape, W_shape, all_gates_shape, false, true);

    // Ht-1*(R^T)
    std::vector<T> Ht_R(all_gates_shape_size);
    reference::matmul(H, R, Ht_R.data(), H_shape, R_shape, all_gates_shape, false, true);

    // Ht-1*(R^T) + Wb + Rb
    std::vector<T> Ht_R_B(all_gates_shape_size);
    reference::add(Ht_R.data(), B, Ht_R_B.data(), all_gates_shape, B_shape, op::AutoBroadcastType::NUMPY);

    // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
    std::vector<T> XHB(all_gates_shape_size);
    reference::add(Xt_W.data(),
                   Ht_R_B.data(),
                   XHB.data(),
                   all_gates_shape,
                   all_gates_shape,
                   op::AutoBroadcastType::NUMPY);

    std::vector<std::vector<T>> X_W_fico(4, std::vector<T>(all_gates_shape_size / 4));
    std::vector<char*> pointers = {reinterpret_cast<char*>(X_W_fico[0].data()),
                                   reinterpret_cast<char*>(X_W_fico[1].data()),
                                   reinterpret_cast<char*>(X_W_fico[2].data()),
                                   reinterpret_cast<char*>(X_W_fico[3].data())};
    // split on gates
    reference::split(reinterpret_cast<char*>(XHB.data()), all_gates_shape, sizeof(T), 1, 4, pointers.data());

    auto clip_activation = [&clip](std::vector<T>& gate, const std::string& activation, bool enable_clip = true) {
        if (clip > 0.f && enable_clip) {
            reference::clamp(gate.data(), gate.data(), static_cast<T>(-clip), static_cast<T>(clip), gate.size());
        }
        if (activation == "relu") {
            reference::relu(gate.data(), gate.data(), gate.size());
        } else if (activation == "sigmoid") {
            reference::sigmoid(gate.data(), gate.data(), gate.size());
        } else if (activation == "tanh") {
            ov::reference::tanh(gate.data(), gate.data(), gate.size());
        } else {
            OPENVINO_THROW("Activation function " + activation + " is not supported.");
        }
    };

    // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
    clip_activation(X_W_fico[0], activation_f);
    // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
    clip_activation(X_W_fico[1], activation_f);
    // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    clip_activation(X_W_fico[2], activation_g);
    // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
    clip_activation(X_W_fico[3], activation_f);

    std::vector<T> mul1(gate_shape_size);
    std::vector<T> mul2(gate_shape_size);
    std::vector<T> Ct(gate_shape_size);
    // ft (.) Ct-1
    reference::multiply(X_W_fico[0].data(), C, mul1.data(), gate_shape, C_shape, op::AutoBroadcastType::NUMPY);
    // it (.) ct
    reference::multiply(X_W_fico[1].data(),
                        X_W_fico[2].data(),
                        mul2.data(),
                        gate_shape,
                        gate_shape,
                        op::AutoBroadcastType::NUMPY);
    // Ct = ft (.) Ct-1 + it (.) ct
    reference::add(mul1.data(), mul2.data(), Ct.data(), gate_shape, gate_shape, op::AutoBroadcastType::NUMPY);
    std::memcpy(out_Ct, Ct.data(), Ct.size() * sizeof(T));
    clip_activation(Ct, activation_h, false);

    // Ht = ot (.) h(Ct)
    reference::multiply(X_W_fico[3].data(), Ct.data(), out_Ht, gate_shape, gate_shape, op::AutoBroadcastType::NUMPY);
}

template <typename T>
void lstm_cell_v1(const T* X,
                  const Shape& X_shape,
                  const T* H,
                  const Shape& H_shape,
                  const T* C,
                  const Shape& C_shape,
                  const T* W,
                  const Shape& W_shape,
                  const T* R,
                  const Shape& R_shape,
                  const T* B,
                  const Shape& B_shape,
                  const T* P,
                  const Shape& P_shape,
                  T* out_Ht,
                  T* out_Ct,
                  const std::string& activation_f,
                  const std::string& activation_g,
                  const std::string& activation_h,
                  float clip,
                  const ov::op::LSTMWeightsFormat weight_format,
                  bool input_forget) {
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
    // H_t     - The hidden state tensor at current time step. Shape: [batch_size,
    // hidden_size].
    // C_t     - The cell state tensor at current time step. Shape: [batch_size,
    // hidden_size].
    // bias    - The sum of biases (weight and recurrence) for input, forget, cell and
    // output gates.
    //           Shape: [4 * hidden_size]
    // p_[iof] - The peephole weight vector for respectively: input, output, and forget
    // gates.
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
    // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    // Ct = ft (.) Ct-1 + it (.) ct
    // Ht = ot (.) h(Ct)
    // --------------------
    Shape gate_shape{X_shape[0], H_shape[1]};
    Shape all_gates_shape{X_shape[0], 4 * H_shape[1]};
    Shape P_gate_shape{H_shape[1]};
    auto P_gate_size = H_shape[1];
    auto gate_shape_size = X_shape[0] * H_shape[1];
    auto all_gates_shape_size = gate_shape_size * 4;

    if (weight_format != ov::op::LSTMWeightsFormat::FICO) {
        OPENVINO_THROW("Only LSTMWeightFormat = FICO is supported.");
    }
    // Xt*(W^T)
    std::vector<T> Xt_W(all_gates_shape_size);
    reference::matmul(X, W, Xt_W.data(), X_shape, W_shape, all_gates_shape, false, true);

    // Ht-1*(R^T)
    std::vector<T> Ht_R(all_gates_shape_size);
    reference::matmul(H, R, Ht_R.data(), H_shape, R_shape, all_gates_shape, false, true);

    // Ht-1*(R^T) + Wb + Rb
    std::vector<T> Ht_R_B(all_gates_shape_size);
    reference::add(Ht_R.data(), B, Ht_R_B.data(), all_gates_shape, B_shape, op::AutoBroadcastType::NUMPY);

    // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
    std::vector<T> XHB(all_gates_shape_size);
    reference::add(Xt_W.data(),
                   Ht_R_B.data(),
                   XHB.data(),
                   all_gates_shape,
                   all_gates_shape,
                   op::AutoBroadcastType::NUMPY);

    std::vector<std::vector<T>> X_W_fico(4, std::vector<T>(all_gates_shape_size / 4));
    std::vector<char*> pointers = {reinterpret_cast<char*>(X_W_fico[0].data()),
                                   reinterpret_cast<char*>(X_W_fico[1].data()),
                                   reinterpret_cast<char*>(X_W_fico[2].data()),
                                   reinterpret_cast<char*>(X_W_fico[3].data())};
    // split on gates
    reference::split(reinterpret_cast<char*>(XHB.data()), all_gates_shape, sizeof(T), 1, 4, pointers.data());

    auto clip_activation = [&clip](std::vector<T>& gate, const std::string& activation, bool enable_clip = true) {
        if (clip > 0.f && enable_clip) {
            reference::clamp(gate.data(), gate.data(), static_cast<T>(-clip), static_cast<T>(clip), gate.size());
        }
        if (activation == "relu") {
            reference::relu(gate.data(), gate.data(), gate.size());
        } else if (activation == "sigmoid") {
            reference::sigmoid(gate.data(), gate.data(), gate.size());
        } else if (activation == "tanh") {
            ov::reference::tanh(gate.data(), gate.data(), gate.size());
        } else {
            OPENVINO_THROW("Activation function " + activation + " is not supported.");
        }
    };

    // Split P on gates f, i, o
    std::vector<std::vector<T>> P_fio(3, std::vector<T>(P_gate_size));

    std::vector<char*> P_pointers = {reinterpret_cast<char*>(P_fio[0].data()),
                                     reinterpret_cast<char*>(P_fio[1].data()),
                                     reinterpret_cast<char*>(P_fio[2].data())};

    reference::split(reinterpret_cast<const char*>(P), P_shape, sizeof(T), 0, 3, P_pointers.data());

    // Pf (.) Ct-1
    std::vector<T> PfCt_1(gate_shape_size);
    reference::multiply(P_fio[0].data(), C, PfCt_1.data(), P_gate_shape, C_shape, op::AutoBroadcastType::NUMPY);

    // Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf + Pf (.) Ct-1
    std::vector<T> XHBPf(gate_shape_size);
    reference::add(X_W_fico[0].data(), PfCt_1.data(), XHBPf.data(), gate_shape, C_shape, op::AutoBroadcastType::NUMPY);

    // Pi (.) Ct-1
    std::vector<T> PiCt_1(gate_shape_size);
    reference::multiply(P_fio[1].data(), C, PiCt_1.data(), P_gate_shape, C_shape, op::AutoBroadcastType::NUMPY);

    // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    clip_activation(XHBPf, activation_f);

    // it calculation per input_forget condition
    std::vector<T> XHBPi(gate_shape_size);
    if (input_forget) {
        // it = (1 - ft)
        std::vector<T> ones(gate_shape_size, T(1));
        reference::subtract(ones.data(),
                            XHBPf.data(),
                            XHBPi.data(),
                            gate_shape,
                            gate_shape,
                            op::AutoBroadcastType::NUMPY);
    } else {
        // Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi + Pi (.) Ct-1
        reference::add(X_W_fico[1].data(),
                       PiCt_1.data(),
                       XHBPi.data(),
                       gate_shape,
                       C_shape,
                       op::AutoBroadcastType::NUMPY);
        // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
        clip_activation(XHBPi, activation_f);
    }

    // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    clip_activation(X_W_fico[2], activation_g);

    std::vector<T> mul1(gate_shape_size);
    std::vector<T> mul2(gate_shape_size);
    std::vector<T> Ct(gate_shape_size);
    // ft (.) Ct-1
    reference::multiply(XHBPf.data(), C, mul1.data(), gate_shape, C_shape, op::AutoBroadcastType::NUMPY);
    // it (.) ct
    reference::multiply(XHBPi.data(),
                        X_W_fico[2].data(),
                        mul2.data(),
                        gate_shape,
                        gate_shape,
                        op::AutoBroadcastType::NUMPY);

    // input_forget=true: Ct = ft (.) Ct-1 + (1 - ft)(.) ct
    // input_forget=false: Ct = ft (.) Ct-1 + it (.) ct
    reference::add(mul1.data(), mul2.data(), Ct.data(), gate_shape, gate_shape, op::AutoBroadcastType::NUMPY);
    std::memcpy(out_Ct, Ct.data(), Ct.size() * sizeof(T));

    // Po (.) Ct
    std::vector<T> PoCt(gate_shape_size);
    reference::multiply(P_fio[2].data(),
                        Ct.data(),
                        PoCt.data(),
                        P_gate_shape,
                        gate_shape,
                        op::AutoBroadcastType::NUMPY);

    // Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo + Po (.) Ct
    std::vector<T> XHBPo(gate_shape_size);
    reference::add(X_W_fico[3].data(), PoCt.data(), XHBPo.data(), gate_shape, gate_shape, op::AutoBroadcastType::NUMPY);

    // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    clip_activation(XHBPo, activation_f);

    clip_activation(Ct, activation_h, false);

    // Ht = ot (.) h(Ct)
    reference::multiply(XHBPo.data(), Ct.data(), out_Ht, gate_shape, gate_shape, op::AutoBroadcastType::NUMPY);
}
}  // namespace reference
}  // namespace ov
