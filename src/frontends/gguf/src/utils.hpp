// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "node_context.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/topk.hpp"

namespace ov {
class Model;
namespace op {
namespace v0 {
class Parameter;
}  // namespace v0
namespace v3 {
class ShapeOf;
}  // namespace v3
}  // namespace op

namespace frontend {
namespace gguf {

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs);

/// \brief Find a Parameter whose friendly name or output tensor names include `name`.
/// Returns nullptr if the model has no such Parameter.
std::shared_ptr<ov::op::v0::Parameter> find_parameter(const std::shared_ptr<ov::Model>& model, const std::string& name);

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

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                         const std::vector<int>& dims);
// Takes the Output rather than the node so a producer with several outputs keeps the right port.
std::shared_ptr<ov::Node> get_dimensions(const ov::Output<ov::Node>& output, const std::vector<int>& dims);

// Take ownership of the temporary output vector assembled by translators, rename its producers,
// then return the same vector without an extra copy.
OutputVector rename_outputs_with_suffix(OutputVector outputs, const std::string& suffix);

/// \brief Build a TopK over `axis` and return its INDICES port.
///
/// Shared by the ARGSORT and TOP_K translators. Both want ggml's "indices that sort/select along
/// ne[0]" semantics, which in OpenVINO is output(1) of a TopK whose index element type follows the
/// decoder's "output_type" attribute. Keeping that contract in one place stops the two call sites
/// from drifting apart.
///
/// \param input       tensor to sort/select over
/// \param k           number of elements to keep along `axis` (may be a dynamic value)
/// \param axis        axis to operate on
/// \param mode        MAX for descending, MIN for ascending
/// \param index_type  element type of the returned indices
/// \param stable      whether ties keep their input order
ov::Output<ov::Node> make_topk_indices(const ov::Output<ov::Node>& input,
                                       const ov::Output<ov::Node>& k,
                                       int64_t axis,
                                       ov::op::v11::TopK::Mode mode,
                                       const ov::element::Type& index_type,
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
    auto res = std::make_shared<T>(context.get_input(0), context.get_input(1));
    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}
}  // namespace op

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
