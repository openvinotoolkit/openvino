// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/reference/concat.hpp"
#include "openvino/reference/gru_cell.hpp"
#include "openvino/reference/lstm_cell.hpp"
#include "openvino/reference/rnn_cell.hpp"
#include "openvino/reference/split.hpp"
#include "reverse_sequence.hpp"

namespace ov {
namespace reference {
enum class CellType {
    RNN,
    GRU,
    LSTM,
    LSTM_v1,
    AUGRU,
};

struct CellArgs {
    std::string activation_f;                                                   // RNN
    std::string activation_g;                                                   // RNN/GRU
    std::string activation_h;                                                   // RNN/GRU/LSTM
    float clip;                                                                 // RNN/GRU/LSTM
    bool linear_before_reset = false;                                           // GRU
    ov::op::LSTMWeightsFormat weight_format = ov::op::LSTMWeightsFormat::FICO;  // LSTM_v1
    bool input_forget = false;                                                  // LSTM_v1
};

template <typename T, typename U>
void cell_pass(CellType type,
               const std::vector<const char*>& inputs,
               const std::vector<Shape>& shapes,
               const std::vector<char*>& outputs,
               const CellArgs& args,
               bool is_reverse) {
    auto squeeze_axis = [](const Shape& shape, size_t axis) -> Shape {
        Shape new_shape(shape.size() - 1);
        for (size_t i = 0, j = 0; i < shape.size(); ++i) {
            if (i != axis) {
                new_shape[j] = shape[i];
                j++;
            }
        }
        return new_shape;
    };

    size_t x_shape_size = shape_size(shapes[0]);

    // split X
    size_t num_splits = shapes[0].at(1);
    size_t part_size = x_shape_size / num_splits * sizeof(T);
    std::vector<char> in_seqs(x_shape_size * sizeof(T));
    std::vector<char*> in_seqs_pointers(num_splits);

    // in case of seq_lengths input was provided and filled with values !=
    // max_seq_lengths
    // we have to fill some of the outputs with zeros (apply mask)
    size_t batch = shapes[0][0];
    size_t hidden_size = shapes[2][2];
    int64_t max_seq_lengths = num_splits;
    const auto* seq_len_values = reinterpret_cast<const U*>(inputs[1]);
    bool enable_mask = false;
    for (size_t i = 0; i < batch; ++i) {
        enable_mask |= (max_seq_lengths != seq_len_values[i]);
    }

    std::vector<char> temp_buffer(x_shape_size * sizeof(T));
    if (is_reverse) {
        reference::reverse_sequence<T, U>(reinterpret_cast<const T*>(inputs[0]),
                                          reinterpret_cast<T*>(temp_buffer.data()),
                                          shapes[0],
                                          0,
                                          1,
                                          seq_len_values);
    } else {
        memcpy(temp_buffer.data(), inputs[0], x_shape_size * sizeof(T));
    }
    for (size_t i = 0; i < num_splits; ++i)
        in_seqs_pointers[i] = in_seqs.data() + i * part_size;

    reference::split(temp_buffer.data(), shapes[0], sizeof(T), 1, num_splits, in_seqs_pointers.data());

    // split A
    std::vector<char> a_seqs;
    if (type == CellType::AUGRU) {
        const auto a_shape_size = shape_size(shapes[6]);
        a_seqs.resize(a_shape_size * sizeof(T));
        std::vector<char*> a_pointers(num_splits);
        for (size_t i = 0; i < num_splits; ++i) {
            a_pointers[i] = a_seqs.data() + i * batch * sizeof(T);
        }
        reference::split(inputs[6], shapes[6], sizeof(T), 1, num_splits, a_pointers.data());
    }

    Shape part_shape{batch, 1, hidden_size};
    size_t part_shape_size = shape_size(part_shape);
    std::vector<std::vector<char>> h_list(num_splits, std::vector<char>(part_shape_size * sizeof(T), 0));
    std::vector<std::vector<char>> c_list(num_splits, std::vector<char>(part_shape_size * sizeof(T), 0));

    // use outputs as a buffer for temporarily values
    char* H_i = outputs[1];
    std::memcpy(H_i, inputs[2], shape_size(shapes[2]) * sizeof(T));

    char* C_i = nullptr;  // LSTMCell only
    if ((type == CellType::LSTM) || (type == CellType::LSTM_v1)) {
        C_i = outputs[2];
        std::memcpy(C_i, inputs[3], shape_size(shapes[3]) * sizeof(T));
    }

    for (size_t time_step = 0; time_step < num_splits; ++time_step) {
        if (type == CellType::LSTM) {
            reference::lstm_cell<T>(reinterpret_cast<const T*>(in_seqs.data() + time_step * part_size),
                                    squeeze_axis(shapes[0], 1),
                                    reinterpret_cast<const T*>(H_i),
                                    squeeze_axis(shapes[2], 1),
                                    reinterpret_cast<const T*>(C_i),
                                    squeeze_axis(shapes[3], 1),
                                    reinterpret_cast<const T*>(inputs[4]),
                                    squeeze_axis(shapes[4], 0),
                                    reinterpret_cast<const T*>(inputs[5]),
                                    squeeze_axis(shapes[5], 0),
                                    reinterpret_cast<const T*>(inputs[6]),
                                    squeeze_axis(shapes[6], 0),
                                    reinterpret_cast<T*>(outputs[1]),
                                    reinterpret_cast<T*>(outputs[2]),
                                    args.activation_f,
                                    args.activation_g,
                                    args.activation_h,
                                    args.clip);
        } else if (type == CellType::LSTM_v1) {
            reference::lstm_cell_v1<T>(reinterpret_cast<const T*>(in_seqs.data() + time_step * part_size),
                                       squeeze_axis(shapes[0], 1),
                                       reinterpret_cast<const T*>(H_i),
                                       squeeze_axis(shapes[2], 1),
                                       reinterpret_cast<const T*>(C_i),
                                       squeeze_axis(shapes[3], 1),
                                       reinterpret_cast<const T*>(inputs[4]),
                                       squeeze_axis(shapes[4], 0),
                                       reinterpret_cast<const T*>(inputs[5]),
                                       squeeze_axis(shapes[5], 0),
                                       reinterpret_cast<const T*>(inputs[6]),
                                       squeeze_axis(shapes[6], 0),
                                       reinterpret_cast<const T*>(inputs[7]),
                                       squeeze_axis(shapes[7], 0),
                                       reinterpret_cast<T*>(outputs[1]),
                                       reinterpret_cast<T*>(outputs[2]),
                                       args.activation_f,
                                       args.activation_g,
                                       args.activation_h,
                                       args.clip,
                                       args.weight_format,
                                       args.input_forget);
        } else if (type == CellType::RNN) {
            reference::rnn_cell<T>(reinterpret_cast<const T*>(in_seqs.data() + time_step * part_size),
                                   squeeze_axis(shapes[0], 1),
                                   reinterpret_cast<const T*>(H_i),
                                   squeeze_axis(shapes[2], 1),
                                   reinterpret_cast<const T*>(inputs[3]),
                                   squeeze_axis(shapes[3], 0),
                                   reinterpret_cast<const T*>(inputs[4]),
                                   squeeze_axis(shapes[4], 0),
                                   reinterpret_cast<const T*>(inputs[5]),
                                   squeeze_axis(shapes[5], 0),
                                   reinterpret_cast<T*>(outputs[1]),
                                   args.activation_f,
                                   args.clip);
        } else if (type == CellType::GRU) {
            reference::gru_cell<T>(reinterpret_cast<const T*>(in_seqs.data() + time_step * part_size),
                                   squeeze_axis(shapes[0], 1),
                                   reinterpret_cast<const T*>(H_i),
                                   squeeze_axis(shapes[2], 1),
                                   reinterpret_cast<const T*>(inputs[3]),
                                   squeeze_axis(shapes[3], 0),
                                   reinterpret_cast<const T*>(inputs[4]),
                                   squeeze_axis(shapes[4], 0),
                                   reinterpret_cast<const T*>(inputs[5]),
                                   squeeze_axis(shapes[5], 0),
                                   reinterpret_cast<T*>(outputs[1]),
                                   args.activation_f,
                                   args.activation_g,
                                   args.clip,
                                   args.linear_before_reset);
        } else if (type == CellType::AUGRU) {
            reference::gru_cell<T>(reinterpret_cast<const T*>(in_seqs.data() + time_step * part_size),
                                   squeeze_axis(shapes[0], 1),
                                   reinterpret_cast<const T*>(H_i),
                                   squeeze_axis(shapes[2], 1),
                                   reinterpret_cast<const T*>(inputs[3]),
                                   squeeze_axis(shapes[3], 0),
                                   reinterpret_cast<const T*>(inputs[4]),
                                   squeeze_axis(shapes[4], 0),
                                   reinterpret_cast<const T*>(inputs[5]),
                                   squeeze_axis(shapes[5], 0),
                                   reinterpret_cast<T*>(outputs[1]),
                                   args.activation_f,
                                   args.activation_g,
                                   args.clip,
                                   args.linear_before_reset,
                                   reinterpret_cast<const T*>(a_seqs.data() + time_step * batch * sizeof(T)));
        }

        if (enable_mask) {
            size_t part_size_single_batch = part_shape_size / batch * sizeof(T);
            for (size_t i = 0; i < batch; ++i) {
                auto shift = i * part_size_single_batch;
                if ((time_step + 1) > static_cast<size_t>(seq_len_values[i])) {
                    continue;
                }
                std::memcpy(h_list[time_step].data() + shift, outputs[1] + shift, part_size_single_batch);
                if ((type == CellType::LSTM) || (type == CellType::LSTM_v1)) {
                    std::memcpy(c_list[time_step].data() + shift, outputs[2] + shift, part_size_single_batch);
                }
            }
            if ((num_splits - time_step) > 1) {
                std::memcpy(outputs[1], h_list[time_step].data(), part_shape_size * sizeof(T));
                if ((type == CellType::LSTM) || (type == CellType::LSTM_v1)) {
                    std::memcpy(outputs[2], c_list[time_step].data(), part_shape_size * sizeof(T));
                }
            } else {
                for (size_t i = 0; i < batch; ++i) {
                    size_t idx = seq_len_values[i] - 1;
                    auto shift = i * part_size_single_batch;
                    if (idx >= 0 && idx < h_list.size()) {
                        std::memcpy(outputs[1] + shift, h_list[idx].data() + shift, part_size_single_batch);
                        if ((type == CellType::LSTM) || (type == CellType::LSTM_v1)) {
                            std::memcpy(outputs[2] + shift, c_list[idx].data() + shift, part_size_single_batch);
                        }
                    } else {
                        std::memset(outputs[1] + shift, 0, part_size_single_batch);
                        if ((type == CellType::LSTM) || (type == CellType::LSTM_v1)) {
                            std::memset(outputs[2] + shift, 0, part_size_single_batch);
                        }
                    }
                }
            }
        } else {
            std::memcpy(h_list[time_step].data(), outputs[1], part_shape_size * sizeof(T));
        }
    }
    // The tensor that concats all the intermediate output values of the hidden.
    // It has shape [batch_size, seq_length, hidden_size]
    std::vector<Shape> in_shapes(num_splits, part_shape);
    std::vector<const char*> to_concat_pointers(num_splits);
    Shape out_shape{batch, num_splits, hidden_size};

    for (size_t i = 0; i < num_splits; ++i)
        to_concat_pointers[i] = h_list[i].data();

    reference::concat(to_concat_pointers, outputs[0], in_shapes, out_shape, 1, sizeof(T));

    if (is_reverse)  // enable_mask
    {
        temp_buffer.resize(shape_size(out_shape) * sizeof(T));
        reference::reverse_sequence<T, U>(reinterpret_cast<const T*>(outputs[0]),
                                          reinterpret_cast<T*>(temp_buffer.data()),
                                          out_shape,
                                          0,
                                          1,
                                          seq_len_values);
        std::memcpy(outputs[0], temp_buffer.data(), shape_size(out_shape) * sizeof(T));
    }
}

template <typename T, typename U>
void lstm_sequence(const char* X,
                   const Shape& X_shape,
                   const char* H,
                   const Shape& H_shape,
                   const char* C,
                   const Shape& C_shape,
                   const char* seq_lengths,
                   const Shape& seq_lengths_shape,
                   const char* W,
                   const Shape& W_shape,
                   const char* R,
                   const Shape& R_shape,
                   const char* B,
                   const Shape& B_shape,
                   char* Y,
                   char* Ho,
                   char* Co,
                   const std::string& activation_f,
                   const std::string& activation_g,
                   const std::string& activation_h,
                   float clip,
                   op::RecurrentSequenceDirection direction) {
    OutputVector results;
    if (direction == op::RecurrentSequenceDirection::FORWARD || direction == op::RecurrentSequenceDirection::REVERSE) {
        CellArgs args;
        args.activation_f = activation_f;
        args.activation_g = activation_g;
        args.activation_h = activation_h;
        args.clip = clip;
        std::vector<const char*> inputs = {X, seq_lengths, H, C, W, R, B};
        std::vector<char*> outputs = {Y, Ho, Co};
        std::vector<Shape> shapes = {X_shape, seq_lengths_shape, H_shape, C_shape, W_shape, R_shape, B_shape};
        cell_pass<T, U>(CellType::LSTM,
                        inputs,
                        shapes,
                        outputs,
                        args,
                        direction == op::RecurrentSequenceDirection::REVERSE);
    } else if (direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) {
        // Split bidirectional case to forward + reverse passes.
        // split inputs
        std::vector<std::vector<char>> H_split(2, std::vector<char>(sizeof(T) * shape_size(H_shape) / 2));
        std::vector<std::vector<char>> C_split(2, std::vector<char>(sizeof(T) * shape_size(C_shape) / 2));
        std::vector<std::vector<char>> W_split(2, std::vector<char>(sizeof(T) * shape_size(W_shape) / 2));
        std::vector<std::vector<char>> R_split(2, std::vector<char>(sizeof(T) * shape_size(R_shape) / 2));
        std::vector<std::vector<char>> B_split(2, std::vector<char>(sizeof(T) * shape_size(B_shape) / 2));
        char* h_pointers[2] = {H_split[0].data(), H_split[1].data()};
        char* c_pointers[2] = {C_split[0].data(), C_split[1].data()};
        char* w_pointers[2] = {W_split[0].data(), W_split[1].data()};
        char* r_pointers[2] = {R_split[0].data(), R_split[1].data()};
        char* b_pointers[2] = {B_split[0].data(), B_split[1].data()};
        reference::split(H, H_shape, sizeof(T), 1, 2, h_pointers);
        reference::split(C, C_shape, sizeof(T), 1, 2, c_pointers);
        reference::split(W, W_shape, sizeof(T), 0, 2, w_pointers);
        reference::split(R, R_shape, sizeof(T), 0, 2, r_pointers);
        reference::split(B, B_shape, sizeof(T), 0, 2, b_pointers);
        std::vector<char> forward_res_y(sizeof(T) * H_shape[0] * H_shape[2] * X_shape[1]);
        std::vector<char> reverse_res_y(sizeof(T) * H_shape[0] * H_shape[2] * X_shape[1]);
        std::vector<std::vector<char>> forward_res(2, std::vector<char>(sizeof(T) * H_shape[0] * H_shape[2]));
        std::vector<std::vector<char>> reverse_res(2, std::vector<char>(sizeof(T) * H_shape[0] * H_shape[2]));

        CellArgs args;
        args.activation_f = activation_f;
        args.activation_g = activation_g;
        args.activation_h = activation_h;
        args.clip = clip;
        std::vector<Shape> shapes = {X_shape, seq_lengths_shape, H_shape, C_shape, W_shape, R_shape, B_shape};
        // update H,C,W,R,B shapes after split
        shapes[2][1] = 1;
        shapes[3][1] = 1;
        for (size_t i = 4; i < shapes.size(); ++i) {
            shapes[i][0] = 1;
        }
        // forward pass
        cell_pass<T, U>(CellType::LSTM,
                        {X, seq_lengths, h_pointers[0], c_pointers[0], w_pointers[0], r_pointers[0], b_pointers[0]},
                        shapes,
                        {forward_res_y.data(), forward_res[0].data(), forward_res[1].data()},
                        args,
                        false);
        // reverse pass
        cell_pass<T, U>(CellType::LSTM,
                        {X, seq_lengths, h_pointers[1], c_pointers[1], w_pointers[1], r_pointers[1], b_pointers[1]},
                        shapes,
                        {reverse_res_y.data(), reverse_res[0].data(), reverse_res[1].data()},
                        args,
                        true);

        // Stack together respective outputs from both forward and reverse passes.
        std::vector<Shape> in_shapes_y = {{H_shape[0], 1, X_shape[1], H_shape[2]},
                                          {H_shape[0], 1, X_shape[1], H_shape[2]}};
        std::vector<Shape> in_shapes_h_c = {{H_shape[0], 1, H_shape[2]}, {H_shape[0], 1, H_shape[2]}};
        Shape output_shape_y{H_shape[0], 2, X_shape[1], H_shape[2]};
        Shape output_shape_h_c{H_shape[0], 2, H_shape[2]};

        reference::concat({forward_res_y.data(), reverse_res_y.data()}, Y, in_shapes_y, output_shape_y, 1, sizeof(T));
        reference::concat({forward_res[0].data(), reverse_res[0].data()},
                          Ho,
                          in_shapes_h_c,
                          output_shape_h_c,
                          1,
                          sizeof(T));
        reference::concat({forward_res[1].data(), reverse_res[1].data()},
                          Co,
                          in_shapes_h_c,
                          output_shape_h_c,
                          1,
                          sizeof(T));
    }
}

template <typename T, typename U>
void gru_sequence(const char* X,
                  const Shape& X_shape,
                  const char* H,
                  const Shape& H_shape,
                  const char* seq_lengths,
                  const Shape& seq_lengths_shape,
                  const char* W,
                  const Shape& W_shape,
                  const char* R,
                  const Shape& R_shape,
                  const char* B,
                  const Shape& B_shape,
                  char* Y,
                  char* Ho,
                  const std::string& activation_f,
                  const std::string& activation_g,
                  const float clip,
                  const op::RecurrentSequenceDirection direction,
                  const bool linear_before_reset,
                  const char* A = nullptr) {
    OutputVector results;
    if (direction == op::RecurrentSequenceDirection::FORWARD || direction == op::RecurrentSequenceDirection::REVERSE) {
        CellArgs args;
        args.activation_f = activation_f;
        args.activation_g = activation_g;
        args.linear_before_reset = linear_before_reset;
        args.clip = clip;
        std::vector<const char*> inputs = {X, seq_lengths, H, W, R, B};
        std::vector<char*> outputs = {Y, Ho};
        std::vector<Shape> shapes = {X_shape, seq_lengths_shape, H_shape, W_shape, R_shape, B_shape};
        if (A) {
            inputs.push_back(A);
            Shape a_shape = X_shape;
            a_shape[2] = 1;
            shapes.push_back(a_shape);
            cell_pass<T, U>(CellType::AUGRU, inputs, shapes, outputs, args,
                            false);  // only forward direction
        } else {
            cell_pass<T, U>(CellType::GRU,
                            inputs,
                            shapes,
                            outputs,
                            args,
                            direction == op::RecurrentSequenceDirection::REVERSE);
        }
    } else if (direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) {
        // Split bidirectional case to forward + reverse passes.
        // split inputs
        std::vector<std::vector<char>> H_split(2, std::vector<char>(sizeof(T) * shape_size(H_shape) / 2));
        std::vector<std::vector<char>> W_split(2, std::vector<char>(sizeof(T) * shape_size(W_shape) / 2));
        std::vector<std::vector<char>> R_split(2, std::vector<char>(sizeof(T) * shape_size(R_shape) / 2));
        std::vector<std::vector<char>> B_split(2, std::vector<char>(sizeof(T) * shape_size(B_shape) / 2));
        char* h_pointers[2] = {H_split[0].data(), H_split[1].data()};
        char* w_pointers[2] = {W_split[0].data(), W_split[1].data()};
        char* r_pointers[2] = {R_split[0].data(), R_split[1].data()};
        char* b_pointers[2] = {B_split[0].data(), B_split[1].data()};
        reference::split(H, H_shape, sizeof(T), 1, 2, h_pointers);
        reference::split(W, W_shape, sizeof(T), 0, 2, w_pointers);
        reference::split(R, R_shape, sizeof(T), 0, 2, r_pointers);
        reference::split(B, B_shape, sizeof(T), 0, 2, b_pointers);
        std::vector<char> forward_res_y(sizeof(T) * H_shape[0] * H_shape[2] * X_shape[1]);
        std::vector<char> forward_res_h(sizeof(T) * H_shape[0] * H_shape[2]);
        std::vector<char> reverse_res_y(sizeof(T) * H_shape[0] * H_shape[2] * X_shape[1]);
        std::vector<char> reverse_res_h(sizeof(T) * H_shape[0] * H_shape[2]);

        CellArgs args;
        args.activation_f = activation_f;
        args.activation_g = activation_g;
        args.linear_before_reset = linear_before_reset;
        args.clip = clip;
        std::vector<Shape> shapes = {X_shape, seq_lengths_shape, H_shape, W_shape, R_shape, B_shape};
        // update H,W,R,B shapes after split
        shapes[2][1] = 1;
        for (size_t i = 3; i < shapes.size(); ++i) {
            shapes[i][0] = 1;
        }

        // forward pass
        cell_pass<T, U>(CellType::GRU,
                        {X, seq_lengths, h_pointers[0], w_pointers[0], r_pointers[0], b_pointers[0]},
                        shapes,
                        {forward_res_y.data(), forward_res_h.data()},
                        args,
                        false);
        // reverse pass
        cell_pass<T, U>(CellType::GRU,
                        {X, seq_lengths, h_pointers[1], w_pointers[1], r_pointers[1], b_pointers[1]},
                        shapes,
                        {reverse_res_y.data(), reverse_res_h.data()},
                        args,
                        true);

        // Stack together respective outputs from both forward and reverse passes.
        std::vector<Shape> in_shapes_y = {{H_shape[0], 1, X_shape[1], H_shape[2]},
                                          {H_shape[0], 1, X_shape[1], H_shape[2]}};
        std::vector<Shape> in_shapes_h = {{H_shape[0], 1, H_shape[2]}, {H_shape[0], 1, H_shape[2]}};
        Shape output_shape_y{H_shape[0], 2, X_shape[1], H_shape[2]};
        Shape output_shape_h{H_shape[0], 2, H_shape[2]};

        reference::concat({forward_res_y.data(), reverse_res_y.data()}, Y, in_shapes_y, output_shape_y, 1, sizeof(T));
        reference::concat({forward_res_h.data(), reverse_res_h.data()}, Ho, in_shapes_h, output_shape_h, 1, sizeof(T));
    }
}

template <typename T, typename U>
void rnn_sequence(const char* X,
                  const Shape& X_shape,
                  const char* H,
                  const Shape& H_shape,
                  const char* seq_lengths,
                  const Shape& seq_lengths_shape,
                  const char* W,
                  const Shape& W_shape,
                  const char* R,
                  const Shape& R_shape,
                  const char* B,
                  const Shape& B_shape,
                  char* Y,
                  char* Ho,
                  const std::string& activation_f,
                  float clip,
                  const op::RecurrentSequenceDirection direction) {
    OutputVector results;
    if (direction == op::RecurrentSequenceDirection::FORWARD || direction == op::RecurrentSequenceDirection::REVERSE) {
        CellArgs args;
        args.activation_f = activation_f;
        args.clip = clip;
        std::vector<const char*> inputs = {X, seq_lengths, H, W, R, B};
        std::vector<char*> outputs = {Y, Ho};
        std::vector<Shape> shapes = {X_shape, seq_lengths_shape, H_shape, W_shape, R_shape, B_shape};
        cell_pass<T, U>(CellType::RNN,
                        inputs,
                        shapes,
                        outputs,
                        args,
                        direction == op::RecurrentSequenceDirection::REVERSE);
    } else if (direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) {
        // Split bidirectional case to forward + reverse passes.
        // split inputs
        std::vector<std::vector<char>> H_split(2, std::vector<char>(sizeof(T) * shape_size(H_shape) / 2));
        std::vector<std::vector<char>> W_split(2, std::vector<char>(sizeof(T) * shape_size(W_shape) / 2));
        std::vector<std::vector<char>> R_split(2, std::vector<char>(sizeof(T) * shape_size(R_shape) / 2));
        std::vector<std::vector<char>> B_split(2, std::vector<char>(sizeof(T) * shape_size(B_shape) / 2));
        char* h_pointers[2] = {H_split[0].data(), H_split[1].data()};
        char* w_pointers[2] = {W_split[0].data(), W_split[1].data()};
        char* r_pointers[2] = {R_split[0].data(), R_split[1].data()};
        char* b_pointers[2] = {B_split[0].data(), B_split[1].data()};
        reference::split(H, H_shape, sizeof(T), 1, 2, h_pointers);
        reference::split(W, W_shape, sizeof(T), 0, 2, w_pointers);
        reference::split(R, R_shape, sizeof(T), 0, 2, r_pointers);
        reference::split(B, B_shape, sizeof(T), 0, 2, b_pointers);
        std::vector<char> forward_res_y(sizeof(T) * H_shape[0] * H_shape[2] * X_shape[1]);
        std::vector<char> forward_res_h(sizeof(T) * H_shape[0] * H_shape[2]);
        std::vector<char> reverse_res_y(sizeof(T) * H_shape[0] * H_shape[2] * X_shape[1]);
        std::vector<char> reverse_res_h(sizeof(T) * H_shape[0] * H_shape[2]);

        CellArgs args;
        args.activation_f = activation_f;
        args.clip = clip;
        std::vector<Shape> shapes = {X_shape, seq_lengths_shape, H_shape, W_shape, R_shape, B_shape};
        // update H,W,R,B shapes after split
        shapes[2][1] = 1;
        for (size_t i = 3; i < shapes.size(); ++i) {
            shapes[i][0] = 1;
        }
        // forward pass
        cell_pass<T, U>(CellType::RNN,
                        {X, seq_lengths, h_pointers[0], w_pointers[0], r_pointers[0], b_pointers[0]},
                        shapes,
                        {forward_res_y.data(), forward_res_h.data()},
                        args,
                        false);
        // reverse pass
        cell_pass<T, U>(CellType::RNN,
                        {X, seq_lengths, h_pointers[1], w_pointers[1], r_pointers[1], b_pointers[1]},
                        shapes,
                        {reverse_res_y.data(), reverse_res_h.data()},
                        args,
                        true);

        // Stack together respective outputs from both forward and reverse passes.
        std::vector<Shape> in_shapes_y = {{H_shape[0], 1, X_shape[1], H_shape[2]},
                                          {H_shape[0], 1, X_shape[1], H_shape[2]}};
        std::vector<Shape> in_shapes_h = {{H_shape[0], 1, H_shape[2]}, {H_shape[0], 1, H_shape[2]}};
        Shape output_shape_y{H_shape[0], 2, X_shape[1], H_shape[2]};
        Shape output_shape_h{H_shape[0], 2, H_shape[2]};

        reference::concat({forward_res_y.data(), reverse_res_y.data()}, Y, in_shapes_y, output_shape_y, 1, sizeof(T));
        reference::concat({forward_res_h.data(), reverse_res_h.data()}, Ho, in_shapes_h, output_shape_h, 1, sizeof(T));
    }
}
}  // namespace reference
}  // namespace ov
