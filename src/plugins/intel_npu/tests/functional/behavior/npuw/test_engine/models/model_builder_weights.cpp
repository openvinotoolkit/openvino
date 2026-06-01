// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_builder_weights.hpp"

#include <cstdint>
#include <vector>

#include "model_builder_internal.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace test {
namespace npuw {

ov::Output<ov::Node> FloatWeight::operator()(const std::string& name,
                                             const ov::Shape& shape,
                                             ov::element::Type compute_precision) const {
    // Deterministic pseudo-random fill: each element gets a unique value derived
    // from the tensor name via xorshift32.  Same name always produces the same
    // weights, but values look random and span a wide enough range ([-0.5, 0.5))
    // to survive FP16 quantisation through NPUW.  This prevents CSE from merging
    // same-shape projections and produces diverse logits in the LM head.
    uint32_t state = seed_from_name(name);
    size_t total = ov::shape_size(shape);
    std::vector<float> data(total);
    for (size_t i = 0; i < total; ++i) {
        uint32_t r = xorshift32(state);
        data[i] = static_cast<float>(r % 10000u) / 10000.0f - 0.5f;  // [-0.5, 0.5)
    }
    auto weight = ov::opset11::Constant::create(storage_type, shape, data);
    weight->set_friendly_name(name);

    if (storage_type == compute_precision) {
        return weight->output(0);
    }
    auto convert = std::make_shared<ov::opset11::Convert>(weight, compute_precision);
    convert->set_friendly_name(name + "_convert");
    return convert->output(0);
}

ov::Output<ov::Node> CompressedWeight::operator()(const std::string& name,
                                                  const ov::Shape& shape,
                                                  ov::element::Type compute_precision) const {
    OPENVINO_ASSERT(shape.size() == 2 || shape.size() == 3,
                    "CompressedWeight expects 2D or 3D shape, got ",
                    shape.size(),
                    "D");
    // 3D [batch, rows, cols]: batched expert weights (MoE).
    // 2D [rows, cols]: standard linear projection.
    const bool is_3d = (shape.size() == 3);
    const size_t batch = is_3d ? shape[0] : 1;
    const size_t rows = is_3d ? shape[1] : shape[0];
    const size_t cols = is_3d ? shape[2] : shape[1];

    // --- Validate pattern constraints ---
    const bool has_zp =
        (pattern == DCOffPattern::SYMM_ZP || pattern == DCOffPattern::GPTQ || pattern == DCOffPattern::ASYMM_ZP);
    if (has_zp) {
        OPENVINO_ASSERT(storage_type == ov::element::u4,
                        "Zero-point patterns require u4 storage type, got ",
                        storage_type);
    }

    // Decomp element type: f32 for SYMM_NO_ZP_F32 and GPTQ, f16 otherwise.
    const bool decomp_f32 = (pattern == DCOffPattern::SYMM_NO_ZP_F32 || pattern == DCOffPattern::GPTQ);
    const auto decomp_et = decomp_f32 ? ov::element::f32 : ov::element::f16;

    // --- Weight value range ---
    int8_t lo = 0, hi = 0;
    if (storage_type == ov::element::i4) {
        lo = -7;  // Symmetric range [-7, 7] (not [-8, 7]) — no zero point needed.
        hi = 7;
    } else if (storage_type == ov::element::u4) {
        lo = 1;
        hi = 15;
    } else if (storage_type == ov::element::nf4) {
        lo = 0;
        hi = 15;
    } else {
        lo = -100;
        hi = 100;
    }

    // --- Group quantization setup ---
    const bool has_groups = group_size > 0;
    size_t num_groups = 1;
    if (has_groups) {
        OPENVINO_ASSERT(group_size >= 64 && group_size % 64 == 0,
                        "Group size must be >= 64 and a multiple of 64 "
                        "(DCOFF AVX2 unpack constraint), got ",
                        group_size);
        OPENVINO_ASSERT(cols >= group_size && cols % group_size == 0,
                        "Group quantization requires cols (",
                        cols,
                        ") >= group_size (",
                        group_size,
                        ") and evenly divisible");
        num_groups = cols / group_size;
    }

    // GPTQ and ASYMM_ZP only have group-quant DCOFF patterns (Reshape2 / AsymmZP::Reshape).
    // No per-channel (group_size=0) DCOFF pass exists for these pattern types.
    if (pattern == DCOffPattern::GPTQ || pattern == DCOffPattern::ASYMM_ZP) {
        OPENVINO_ASSERT(has_groups,
                        "DCOffPattern::",
                        (pattern == DCOffPattern::GPTQ ? "GPTQ" : "ASYMM_ZP"),
                        " requires group_size > 0 (no per-channel DCOFF pass exists)");
    }

    // Weight shape includes batch dim for 3D (MoE batched experts).
    // Group quant adds an extra group dim inside each [rows, cols] slice.
    ov::Shape weight_shape;
    if (is_3d) {
        weight_shape = has_groups ? ov::Shape{batch, rows, num_groups, group_size} : ov::Shape{batch, rows, cols};
    } else {
        weight_shape = has_groups ? ov::Shape{rows, num_groups, group_size} : ov::Shape{rows, cols};
    }
    uint32_t w_state = seed_from_name(name);
    std::vector<int8_t> w_data(batch * rows * cols);
    for (size_t i = 0; i < w_data.size(); ++i) {
        uint32_t r = xorshift32(w_state);
        int val = static_cast<int>(lo) + static_cast<int>(r % static_cast<uint32_t>(hi - lo + 1));
        w_data[i] = static_cast<int8_t>(val);
    }
    auto weight = ov::opset11::Constant::create(storage_type, weight_shape, w_data);
    weight->set_friendly_name(name);

    ov::Output<ov::Node> decomp_input = weight->output(0);

    // --- Convert weight to decomp element type (skip if already matching) ---
    ov::Output<ov::Node> multiply_input;
    if (storage_type != decomp_et) {
        auto convert = std::make_shared<ov::opset11::Convert>(decomp_input, decomp_et);
        convert->set_friendly_name(name + "_convert");
        multiply_input = convert->output(0);
    } else {
        multiply_input = decomp_input;
    }

    // --- Zero-point subtraction (SYMM_ZP, GPTQ, ASYMM_ZP) ---
    if (has_zp) {
        // ZP shape: always 2D (same reasoning as scale — MoE executor doesn't slice ZP)
        ov::Shape zp_shape;
        zp_shape = has_groups ? ov::Shape{rows, num_groups, 1} : ov::Shape{rows, 1};
        const size_t zp_count = has_groups ? rows * num_groups : rows;
        const int mid = (static_cast<int>(lo) + static_cast<int>(hi)) / 2;

        if (pattern == DCOffPattern::GPTQ) {
            // GPTQ: ZP is f32 Constant fed directly to Subtract (no Convert).
            // Uniform value so it stays Constant after partitioning.
            std::vector<float> zp_f32(zp_count, static_cast<float>(mid));
            auto zp_const = ov::opset11::Constant::create(ov::element::f32, zp_shape, zp_f32);
            zp_const->set_friendly_name(name + "_zp");

            auto subtract = std::make_shared<ov::opset11::Subtract>(multiply_input, zp_const);
            subtract->set_friendly_name(name + "_subtract");
            multiply_input = subtract->output(0);

        } else if (pattern == DCOffPattern::SYMM_ZP) {
            // SymmZP: u4 ZP Constant → Convert(f16) → Subtract.
            // Uniform value across all layers so it stays Constant after partitioning.
            std::vector<int8_t> zp_data(zp_count, static_cast<int8_t>(mid));
            auto zp_const = ov::opset11::Constant::create(storage_type, zp_shape, zp_data);
            zp_const->set_friendly_name(name + "_zp");

            auto zp_convert = std::make_shared<ov::opset11::Convert>(zp_const, ov::element::f16);
            zp_convert->set_friendly_name(name + "_zp_convert");

            auto subtract = std::make_shared<ov::opset11::Subtract>(multiply_input, zp_convert);
            subtract->set_friendly_name(name + "_subtract");
            multiply_input = subtract->output(0);

        } else {
            // AsymmZP: u4 ZP Constant → Convert(f16) → Subtract.
            // Per-layer varying values (seeded from name) so NPUW promotes
            // it to a Parameter after partitioning.
            uint32_t zp_state = seed_from_name(name + "_zp");
            std::vector<int8_t> zp_data(zp_count);
            for (size_t i = 0; i < zp_data.size(); ++i) {
                uint32_t r = xorshift32(zp_state);
                int zp_val = mid + static_cast<int>(r % 3u) - 1;
                zp_data[i] = static_cast<int8_t>(zp_val);
            }
            auto zp_const = ov::opset11::Constant::create(storage_type, zp_shape, zp_data);
            zp_const->set_friendly_name(name + "_zp");

            auto zp_convert = std::make_shared<ov::opset11::Convert>(zp_const, ov::element::f16);
            zp_convert->set_friendly_name(name + "_zp_convert");

            auto subtract = std::make_shared<ov::opset11::Subtract>(multiply_input, zp_convert);
            subtract->set_friendly_name(name + "_subtract");
            multiply_input = subtract->output(0);
        }
    }

    // --- Scale: per-group or per-channel, with optional batch dim ---
    // Magnitude kept small (scale_range ≈ 1/hi) so decompressed values stay moderate
    // (roughly ±1), preventing hidden state overflow.
    // 3D batched weights (MoE) use 3D scale [batch, rows, 1] so DEVICE_ROUTED transform
    // can identify the expert dimension (shape[0] == num_experts).
    ov::Shape scale_shape;
    if (is_3d) {
        scale_shape = has_groups ? ov::Shape{batch, rows, num_groups, 1} : ov::Shape{batch, rows, 1};
    } else {
        scale_shape = has_groups ? ov::Shape{rows, num_groups, 1} : ov::Shape{rows, 1};
    }
    const size_t scale_count = batch * (has_groups ? rows * num_groups : rows);
    const float scale_range = 1.0f / static_cast<float>(hi);
    uint32_t s_state = seed_from_name(name + "_scale");
    std::vector<float> scale_data(scale_count);
    for (size_t i = 0; i < scale_data.size(); ++i) {
        uint32_t r = xorshift32(s_state);
        scale_data[i] = scale_range * (0.1f + static_cast<float>(r % 1000u) / 1000.0f);
    }
    auto scale = ov::opset11::Constant::create(decomp_et, scale_shape, scale_data);
    scale->set_friendly_name(name + "_scale");

    auto scaled = std::make_shared<ov::opset11::Multiply>(multiply_input, scale);
    scaled->set_friendly_name(name + "_decompress");

    ov::Output<ov::Node> decompressed = scaled->output(0);

    // --- Group quant: collapse group dims back to original shape ---
    if (has_groups) {
        std::vector<int64_t> out_dims;
        if (is_3d)
            out_dims = {static_cast<int64_t>(batch), static_cast<int64_t>(rows), static_cast<int64_t>(cols)};
        else
            out_dims = {static_cast<int64_t>(rows), static_cast<int64_t>(cols)};
        auto out_shape = ov::opset11::Constant::create(ov::element::i64, ov::Shape{out_dims.size()}, out_dims);
        auto reshaped = std::make_shared<ov::opset11::Reshape>(decompressed, out_shape, false);
        reshaped->set_friendly_name(name + "_reshape");
        decompressed = reshaped->output(0);
    }

    // --- Convert to compute precision if needed ---
    if (decomp_et != compute_precision) {
        auto to_compute = std::make_shared<ov::opset11::Convert>(decompressed, compute_precision);
        to_compute->set_friendly_name(name + "_to_compute");
        return to_compute->output(0);
    }
    return decompressed;
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
