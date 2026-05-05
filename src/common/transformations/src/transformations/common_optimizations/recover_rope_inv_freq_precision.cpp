// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/recover_rope_inv_freq_precision.hpp"

#include <cmath>
#include <limits>
#include <unordered_set>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

namespace {

/// Check if the downstream path from \p start reaches any Sin or Cos op within \p max_depth hops.
bool reaches_sin_cos(const std::shared_ptr<ov::Node>& start, size_t max_depth) {
    std::unordered_set<ov::Node*> visited;
    std::vector<std::pair<ov::Node*, size_t>> stack;
    stack.push_back({start.get(), 0});

    while (!stack.empty()) {
        auto [node, depth] = stack.back();
        stack.pop_back();

        if (depth > max_depth || visited.count(node))
            continue;
        visited.insert(node);

        if (ov::is_type<ov::op::v0::Sin>(node) || ov::is_type<ov::op::v0::Cos>(node))
            return true;

        for (auto& output : node->outputs()) {
            for (auto& input : output.get_target_inputs()) {
                stack.push_back({input.get_node(), depth + 1});
            }
        }
    }
    return false;
}

/// Mark a node and all its downstream consumers (up to Sin/Cos inclusive) with disable_fp16_compression.
void mark_downstream_chain(const std::shared_ptr<ov::Node>& start, size_t max_depth) {
    std::unordered_set<ov::Node*> visited;
    std::vector<std::pair<ov::Node*, size_t>> stack;
    stack.push_back({start.get(), 0});

    while (!stack.empty()) {
        auto [node, depth] = stack.back();
        stack.pop_back();

        if (depth > max_depth || visited.count(node))
            continue;
        visited.insert(node);

        auto shared = node->shared_from_this();
        ov::disable_fp16_compression(shared);

        // Stop at Sin/Cos — don't mark further
        if (ov::is_type<ov::op::v0::Sin>(node) || ov::is_type<ov::op::v0::Cos>(node))
            continue;

        for (auto& output : node->outputs()) {
            for (auto& input : output.get_target_inputs()) {
                stack.push_back({input.get_node(), depth + 1});
            }
        }
    }
}

/// Check if values form a geometric series: vals[i] ≈ vals[0] * r^i for some constant ratio r.
/// Returns the detected base (1/r^(dim/2)) if it's a geometric series, 0.0 otherwise.
double detect_inv_freq_base(const float* vals, size_t n) {
    if (n < 4)
        return 0.0;

    // inv_freq[0] should be close to 1.0
    if (std::abs(vals[0] - 1.0f) > 0.01f)
        return 0.0;

    // All values should be positive and decreasing
    for (size_t i = 0; i < n; i++) {
        if (vals[i] <= 0.0f)
            return 0.0;
        if (i > 0 && vals[i] >= vals[i - 1])
            return 0.0;
    }

    // Compute ratio between consecutive elements — should be approximately constant
    // Use log-space least-squares for robustness against f16 quantization noise
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (size_t i = 0; i < n; i++) {
        double x = static_cast<double>(i);
        double y = std::log(static_cast<double>(vals[i]));
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

    // slope = log(r) where r = base^(-2/dim), dim = 2*n
    // So base = exp(-slope * dim / 2) = exp(-slope * n)
    double base = std::exp(-slope * static_cast<double>(n));

    // Verify the fit: recompute and check max relative error
    double max_rel_err = 0;
    for (size_t i = 0; i < n; i++) {
        double expected = 1.0 / std::pow(base, 2.0 * i / (2.0 * n));
        double actual = static_cast<double>(vals[i]);
        double rel_err = std::abs(expected - actual) / std::abs(expected);
        if (rel_err > max_rel_err)
            max_rel_err = rel_err;
    }

    // Allow up to 2% relative error (f16 quantization can cause ~1% error)
    if (max_rel_err > 0.02)
        return 0.0;

    // Round base to nearest common value (10000, 500000, 1000000, etc.)
    // This ensures we get exact values matching the original PyTorch computation
    const double common_bases[] = {10000.0, 100000.0, 500000.0, 1000000.0, 10000000.0};
    for (double cb : common_bases) {
        double test_max_err = 0;
        for (size_t i = 0; i < n; i++) {
            double expected = 1.0 / std::pow(cb, 2.0 * i / (2.0 * n));
            // Compare against f16-rounded version of the expected value
            float expected_f16 = static_cast<ov::float16>(static_cast<float>(expected));
            double actual = static_cast<double>(vals[i]);
            double err = std::abs(static_cast<double>(expected_f16) - actual);
            if (err > test_max_err)
                test_max_err = err;
        }
        // If all f16-rounded expected values match the actual f16 values, this is the base
        if (test_max_err < 1e-7) {
            return cb;
        }
    }

    // If no exact match with a common base, return the estimated base
    return base;
}

}  // namespace

namespace ov::pass {

bool RecoverRoPEInvFreqPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(RecoverRoPEInvFreqPrecision);
    bool modified = false;

    for (const auto& op : model->get_ordered_ops()) {
        // Look for: f16 Constant -> Convert(f16->f32)
        auto convert = ov::as_type_ptr<ov::op::v0::Convert>(op);
        if (!convert)
            continue;

        if (convert->get_output_element_type(0) != ov::element::f32)
            continue;

        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(convert->get_input_node_shared_ptr(0));
        if (!constant)
            continue;

        if (constant->get_element_type() != ov::element::f16)
            continue;

        // Must be a small constant (inv_freq is typically <= 256 elements)
        auto total_size = ov::shape_size(constant->get_shape());
        if (total_size < 4 || total_size > 512)
            continue;

        // Check that this constant eventually feeds into Sin/Cos through MatMul or Multiply
        // Walk forward from the Convert: expect Broadcast/Reshape/Unsqueeze -> MatMul/Multiply -> ... -> Sin/Cos
        if (!reaches_sin_cos(convert, 8))
            continue;

        // Read the f16 values as f32 (decompressed)
        auto f16_data = constant->get_data_ptr<ov::float16>();
        std::vector<float> vals(total_size);
        for (size_t i = 0; i < total_size; i++) {
            vals[i] = static_cast<float>(f16_data[i]);
        }

        // Check if values form a geometric series (inv_freq pattern)
        double base = detect_inv_freq_base(vals.data(), total_size);
        if (base <= 0.0)
            continue;

        // Recompute inv_freq in f64 then convert to f32
        size_t dim = 2 * total_size;  // dim = 2 * num_freq
        std::vector<float> recomputed(total_size);
        for (size_t i = 0; i < total_size; i++) {
            double inv_freq = 1.0 / std::pow(base, static_cast<double>(2 * i) / static_cast<double>(dim));
            recomputed[i] = static_cast<float>(inv_freq);
        }

        // Create new f32 constant with the recomputed values
        auto new_constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, constant->get_shape(), recomputed.data());
        new_constant->set_friendly_name(constant->get_friendly_name() + "_f32_recovered");
        ov::copy_runtime_info(constant, new_constant);

        // Mark the new constant to keep its precision (prevent re-compression to f16)
        ov::disable_fp16_compression(new_constant);
        ov::enable_keep_const_precision(new_constant);

        // Replace the f16 Constant -> Convert(f16->f32) pair with the new f32 Constant
        convert->output(0).replace(new_constant->output(0));

        // Mark the downstream computation chain (up to Sin/Cos) with disable_fp16_compression
        mark_downstream_chain(new_constant, 8);

        modified = true;
    }

    return modified;
}

}  // namespace ov::pass
