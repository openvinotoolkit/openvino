// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

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
void gru_cell(const T* X,
              const Shape& X_shape,
              const T* H,
              const Shape& H_shape,
              const T* W,
              const Shape& W_shape,
              const T* R,
              const Shape& R_shape,
              const T* B,
              const Shape& B_shape,
              T* dst_data,
              const std::string& activation_f,
              const std::string& activation_g,
              float clip,
              bool linear_before_reset,
              const T* A = nullptr) {
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
    // H_t    - The hidden state tensor at current time step. Shape: [batch_size,
    // hidden_size].
    // B      - The sum of biases (weight and recurrence) for update, reset and hidden
    // gates.
    //          If linear_before_reset := true then biases for hidden gates are placed
    //          separately
    //          (weight and recurrence).
    //          Shape: [gates_count * hidden_size] when linear_before_reset := false
    //          Shape: [(gates_count + 1) * hidden_size] when linear_before_reset :=
    //          true
    // Wb[zrh] - W bias vectors for update, reset and hidden gates.
    // Rb[zrh] - R bias vectors for update, reset and hidden gates.
    // A       - Attentional update gate.
    // (.) - Denotes element-wise multiplication.
    // *   - Denotes dot product.

    // ---- Equations ----
    // f, g  - are activation functions
    // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # when linear_before_reset
    // := false
    //                                                      # (default)
    // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset
    // := true
    // zt' = (1-A) (.) zt
    // Ht = (1 - zt') (.) ht + zt' (.) Ht-1
    // -------------------

    Shape gate_shape{X_shape[0], H_shape[1]};           // [batch_size, hidden_size]
    Shape all_gates_shape{X_shape[0], 3 * H_shape[1]};  // [batch_size, 3*hidden_size]
    Shape bias_shape{H_shape[1], H_shape[1]};           // [hidden_size, hidden_size]
    auto gate_shape_size = X_shape[0] * H_shape[1];     // batch_size * hidden_size
    auto all_gates_shape_size = gate_shape_size * 3;
    auto bias_shape_size = H_shape[1] * H_shape[1];

    // Xt*(W^T)
    std::vector<T> Xt_W(all_gates_shape_size);
    reference::matmul(X, W, Xt_W.data(), X_shape, W_shape, all_gates_shape, false, true);

    // Ht-1*(R^T)
    std::vector<T> Ht_R(all_gates_shape_size);
    reference::matmul(H, R, Ht_R.data(), H_shape, R_shape, all_gates_shape, false, true);

    std::vector<std::vector<T>> X_W_zrh(3, std::vector<T>(gate_shape_size));
    std::vector<char*> pointers_XW = {reinterpret_cast<char*>(X_W_zrh[0].data()),
                                      reinterpret_cast<char*>(X_W_zrh[1].data()),
                                      reinterpret_cast<char*>(X_W_zrh[2].data())};
    std::vector<std::vector<T>> R_zrh(3, std::vector<T>(bias_shape_size));
    std::vector<char*> pointers_R = {reinterpret_cast<char*>(R_zrh[0].data()),
                                     reinterpret_cast<char*>(R_zrh[1].data()),
                                     reinterpret_cast<char*>(R_zrh[2].data())};
    std::vector<std::vector<T>> Ht_R_zrh(3, std::vector<T>(gate_shape_size));
    std::vector<char*> pointers_H_R = {reinterpret_cast<char*>(Ht_R_zrh[0].data()),
                                       reinterpret_cast<char*>(Ht_R_zrh[1].data()),
                                       reinterpret_cast<char*>(Ht_R_zrh[2].data())};

    size_t num_b_splits = linear_before_reset ? 4 : 3;
    std::vector<std::vector<T>> biases_zrh(num_b_splits, std::vector<T>(B_shape[0] / num_b_splits));
    std::vector<char*> pointers_biases = {reinterpret_cast<char*>(biases_zrh[0].data()),
                                          reinterpret_cast<char*>(biases_zrh[1].data()),
                                          reinterpret_cast<char*>(biases_zrh[2].data())};
    if (linear_before_reset) {
        pointers_biases.push_back(reinterpret_cast<char*>(biases_zrh[3].data()));
    }

    // split on gates
    reference::split(reinterpret_cast<char*>(Xt_W.data()), all_gates_shape, sizeof(T), 1, 3, pointers_XW.data());
    reference::split(reinterpret_cast<const char*>(R), R_shape, sizeof(T), 0, 3, pointers_R.data());
    reference::split(reinterpret_cast<char*>(Ht_R.data()), all_gates_shape, sizeof(T), 1, 3, pointers_H_R.data());
    reference::split(reinterpret_cast<const char*>(B), B_shape, sizeof(T), 0, num_b_splits, pointers_biases.data());

    auto clip_activation = [&clip](std::vector<T>& gate, const std::string& activation) {
        if (clip > 0.f) {
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

    // calculate z_t
    // steps:
    // Ht-1*(Rz^T) + Wbz + Rbz
    // Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz
    // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    std::vector<T> z_t(gate_shape_size);
    reference::add(Ht_R_zrh[0].data(),
                   biases_zrh[0].data(),
                   z_t.data(),
                   gate_shape,
                   {B_shape[0] / num_b_splits},
                   op::AutoBroadcastType::NUMPY);
    reference::add(X_W_zrh[0].data(), z_t.data(), z_t.data(), gate_shape, gate_shape, op::AutoBroadcastType::NUMPY);

    clip_activation(z_t, activation_f);

    T one[] = {1};
    if (A) {  // Attention score input provided
        const Shape a_shape{gate_shape[0], 1};
        std::vector<T> a_t(gate_shape[0]);
        reference::subtract(one, A, a_t.data(), {1}, a_shape, op::AutoBroadcastType::NUMPY);
        reference::multiply(a_t.data(), z_t.data(), z_t.data(), a_shape, gate_shape, op::AutoBroadcastType::NUMPY);
    }

    // calculate r_t
    // steps:
    // Ht-1*(Rr^T) + Wbr + Rbr
    // Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr
    // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    std::vector<T> r_t(gate_shape_size);
    reference::add(Ht_R_zrh[1].data(),
                   biases_zrh[1].data(),
                   r_t.data(),
                   gate_shape,
                   {B_shape[0] / num_b_splits},
                   op::AutoBroadcastType::NUMPY);
    reference::add(X_W_zrh[1].data(), r_t.data(), r_t.data(), gate_shape, gate_shape, op::AutoBroadcastType::NUMPY);
    clip_activation(r_t, activation_f);

    // calculate h_t
    std::vector<T> h_t(gate_shape_size);
    if (linear_before_reset) {
        // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
        reference::add(Ht_R_zrh[2].data(),
                       biases_zrh[3].data(),
                       h_t.data(),
                       gate_shape,
                       {B_shape[0] / num_b_splits},
                       op::AutoBroadcastType::NUMPY);
        reference::multiply(r_t.data(), h_t.data(), h_t.data(), gate_shape, gate_shape, op::AutoBroadcastType::NUMPY);
        reference::add(h_t.data(),
                       biases_zrh[2].data(),
                       h_t.data(),
                       gate_shape,
                       {B_shape[0] / num_b_splits},
                       op::AutoBroadcastType::NUMPY);
        reference::add(X_W_zrh[2].data(), h_t.data(), h_t.data(), gate_shape, gate_shape, op::AutoBroadcastType::NUMPY);
    } else {
        // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
        reference::multiply(r_t.data(), H, h_t.data(), gate_shape, H_shape, op::AutoBroadcastType::NUMPY);
        std::vector<T> matmul(gate_shape_size);
        reference::matmul(h_t.data(), R_zrh[2].data(), matmul.data(), gate_shape, bias_shape, gate_shape, false, true);
        reference::add(matmul.data(),
                       biases_zrh[2].data(),
                       h_t.data(),
                       gate_shape,
                       {B_shape[0] / num_b_splits},
                       op::AutoBroadcastType::NUMPY);
        reference::add(X_W_zrh[2].data(), h_t.data(), h_t.data(), gate_shape, gate_shape, op::AutoBroadcastType::NUMPY);
    }
    clip_activation(h_t, activation_g);
    // Ht = (1 - zt) (.) ht + zt (.) Ht-1
    std::vector<T> mul1(gate_shape_size);
    std::vector<T> mul2(gate_shape_size);
    reference::subtract(one, z_t.data(), mul1.data(), {1}, gate_shape, op::AutoBroadcastType::NUMPY);
    reference::multiply(mul1.data(), h_t.data(), mul1.data(), gate_shape, gate_shape, op::AutoBroadcastType::NUMPY);
    reference::multiply(z_t.data(), H, mul2.data(), gate_shape, gate_shape, op::AutoBroadcastType::NUMPY);
    reference::add(mul1.data(), mul2.data(), dst_data, gate_shape, gate_shape, op::AutoBroadcastType::NUMPY);
}
}  // namespace reference
}  // namespace ov
