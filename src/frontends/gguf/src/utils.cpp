// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace gguf {

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs) {
    auto input_size = context.get_input_size();
    FRONT_END_OP_CONVERSION_CHECK(input_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(input_size <= max_inputs, "Got more inputs than expected");
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                         const std::vector<int>& dims) {
    using namespace ov::op;
    const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims) {
    return get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(node), dims);
}

OutputVector rename_outputs_with_suffix(const OutputVector& outputs, const std::string& suffix) {
    for (const auto& output : outputs) {
        auto node = output.get_node_shared_ptr();
        std::string name = node->get_friendly_name();
        name += "_";
        name += suffix;
        node->set_friendly_name(name);
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    return outputs;
}

namespace {
ov::Output<ov::Node> rope_yarn_ramp_mix(int n_dims, const float corr_dims[2], float ext_factor) {
    int half_n_dims = n_dims / 2;
    std::vector<float> dim_ids_vec(half_n_dims);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0);
    auto dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t)half_n_dims}, dim_ids_vec);
    auto corr_low = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    auto corr_high = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    auto denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_high, corr_low),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    auto ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_low), denom);
    auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    auto ext_factor_node = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_node);
    return ramp_mix;
}

float gguf_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}

void gguf_rope_yarn_corr_dims(int n_dims,
                              int n_ctx_orig,
                              float freq_base,
                              float beta_fast,
                              float beta_slow,
                              float dims[2]) {
    float start = floorf(gguf_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end = ceilf(gguf_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<float>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(const RopeConfig& rope_config,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           bool imrope) {
    if (imrope) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, true);
        auto pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } else {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    const float freq_base = rope_config.freq_base;
    const float freq_scale = rope_config.freq_scale;
    const float ext_factor = rope_config.ext_factor;
    const float attn_factor = rope_config.attn_factor;
    const float beta_fast = rope_config.beta_fast;
    const float beta_slow = rope_config.beta_slow;
    const int n_dims = rope_config.n_dims;
    const size_t n_dims_half = n_dims >> 1;
    const int n_ctx_orig = rope_config.n_ctx_orig;

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    std::vector<float> factor(n_dims_half);

    Output<Node> freq_factors;

    Output<Node> theta;
    float mscale = attn_factor;
    if (imrope) {
        std::vector<int64_t> gather_indices(n_dims_half);
        for (size_t j = 0; j < n_dims_half; j++) {
            gather_indices[j] = j % 3;
            factor[j] = std::pow(theta_scale, j);
        }
        auto gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{n_dims_half}, gather_indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        auto factor_const = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_dims_half}, factor);
        theta = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_const);
    } else {
        float corr_dims[2];
        gguf_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        for (size_t i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        freq_factors =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        if (rope_freqs_weight) {
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_freqs_weight);
        }

        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        auto theta_interp =
            std::make_shared<ov::op::v1::Multiply>(theta_extrap,
                                                   ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        if (ext_factor == 0.0f) {
            theta = theta_interp;
        } else {
            auto ramp_mix = rope_yarn_ramp_mix(n_dims, corr_dims, ext_factor);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);

            theta =
                std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                                                  std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    }

    Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    if (!imrope) {
        auto mscale_node = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_node);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_node);
    }

    return std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_input(const NodeContext& context, int input_index, int slice_len) {
    // Only works for VIEW operations that slice at the lowest dimension
    // If the VIEW also reshape the result, `slice_len` should be provided
    auto input = context.get_input(input_index);
    auto src1_stride = context.get_input_stride(input_index);

    int64_t split_addr = context.get_input_view_offset(input_index) / (int64_t)src1_stride[3];
    if (slice_len == 0) {
        slice_len = context.get_input_shape(input_index)[3].get_length();
    }
    int64_t slice_end = split_addr + slice_len;

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {split_addr});
    auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    auto sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, stride, axes);
    return sliced;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
