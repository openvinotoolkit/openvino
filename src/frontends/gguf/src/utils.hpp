// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "node_context.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/topk.hpp"

namespace ov {
class Model;
}  // namespace ov

namespace ov::op {
namespace v0 {
class Parameter;
}  // namespace v0
namespace v3 {
class ShapeOf;
}  // namespace v3
}  // namespace ov::op

namespace ov::frontend::gguf {

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs);

/// \brief Resolve one combined or two explicit GLU inputs and apply the decoder's swapped flag.
std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> get_glu_inputs(const NodeContext& context);

/// \brief Find a Parameter whose friendly name or output tensor names include `name`.
/// Returns nullptr if the model has no such Parameter.
std::shared_ptr<ov::op::v0::Parameter> find_parameter(const std::shared_ptr<ov::Model>& model, const std::string& name);

/// \brief Read an integer string attribute from the model's `gguf_mmproj` rt_info; must be >= `minimum`.
int64_t mmproj_int_metadata(const std::shared_ptr<ov::Model>& model, const std::string& key, int64_t minimum = 1);

int non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb);

template <typename T>
std::vector<T> permute(const std::vector<T>& x, const std::vector<size_t>& perm) {
    std::vector<T> result;
    result.reserve(perm.size());
    for (size_t i : perm) {
        result.push_back(x[i]);
    }
    return result;
}

// Select entries out of an already computed shape vector. Prefer this over repeated get_dimensions
// calls on one tensor, which build a fresh ShapeOf each time.
std::shared_ptr<ov::Node> gather_dims(const ov::Output<ov::Node>& shape, const std::vector<int>& dims);

// Use gather_dims for an existing shape: an Output overload would take its ShapeOf again.
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                         const std::vector<int>& dims) = delete;
// Takes the Output rather than the node so a producer with several outputs keeps the right port.
std::shared_ptr<ov::Node> get_dimensions(const ov::Output<ov::Node>& output, const std::vector<int>& dims);

// x / max(sqrt(sum(x^2, -1)), eps), matching ggml's l2_norm.
ov::Output<ov::Node> make_l2_norm(const ov::Output<ov::Node>& x, float eps);

ov::Output<ov::Node> apply_hadamard(const NodeContext& context, ov::Output<ov::Node> input, bool matrix_form = false);
void decompose_hadamard(ov::Output<ov::Node> input);

/// \brief Give `out`'s producer and its first output tensor the same name.
void name_output(const ov::Output<ov::Node>& out, const std::string& name);

// Take ownership of the temporary output vector assembled by translators, rename its producers,
// then return the same vector without an extra copy.
OutputVector rename_outputs_with_suffix(OutputVector outputs, const std::string& suffix);

bool has_nonnegative_indices(const ov::Output<ov::Node>& output);

// GGML TOP_K and ARGSORT return I32 indices, which are OpenVINO TopK's second output.
ov::Output<ov::Node> make_topk_indices(const ov::Output<ov::Node>& input,
                                       const ov::Output<ov::Node>& k,
                                       int64_t axis,
                                       ov::op::v11::TopK::Mode mode,
                                       bool stable = false);

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(const RopeConfig& rope_config,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight = nullptr,
                                                           bool imrope = false,
                                                           bool stateful = false);

ov::Output<ov::Node> process_view_input(const NodeContext& context, int input_index, int slice_len = 0);

namespace op {
template <typename T>
OutputVector translate_1to1_match_1_input(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto res = std::make_shared<T>(context.get_input(0));
    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

template <typename T>
OutputVector translate_1to1_match_2_inputs(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    const auto output_type = lhs.get_element_type();
    const bool promote =
        output_type != rhs.get_element_type() && output_type.is_real() && rhs.get_element_type().is_real();
    if (promote) {
        const auto compute_type = output_type == ov::element::f64 || rhs.get_element_type() == ov::element::f64
                                      ? ov::element::f64
                                      : ov::element::f32;
        lhs = std::make_shared<ov::op::v0::Convert>(lhs, compute_type);
        rhs = std::make_shared<ov::op::v0::Convert>(rhs, compute_type);
    }
    ov::Output<ov::Node> res = std::make_shared<T>(lhs, rhs);
    if (promote && res.get_element_type().is_real() && res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }
    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}
}  // namespace op

}  // namespace ov::frontend::gguf
