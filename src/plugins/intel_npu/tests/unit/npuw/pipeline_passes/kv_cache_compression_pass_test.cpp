// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for run_kv_cache_dynamic_quantization_passes().
//
// Each test builds a synthetic SDPA model, runs the compression pass, and
// verifies the structural outcome (param/result counts, absence of internal
// DynamicQuantize ops after decomposition).  Parametrized over:
//   - num_sdpa:     1 (single head) or 3 (multi-head)
//   - quant_type:   Asymmetric (adds scale + zp) or Symmetric (scale only)
//   - quant_dt:     u8 / i8
//
// Model structure per SDPA block N:
//
//   Parameters:
//     past_key_values.N.key   [1,H,Spast,D]  ---+
//     new_key.N               [1,H,1,D]         +-- Concat(axis=2) --> MatMul1(Q, K^T)
//                                                                          |
//     query.N                 [1,H,1,D]       -----------------------------|
//     mask.N                  [1,1,1,Spast+1] --> Add --> Softmax --> MatMul2
//                                                                          |
//     past_key_values.N.value [1,H,D,Spast]   ---+                        |
//     new_value.N             [1,H,D,1]          +-- Concat(axis=3) -------|
//
//   Results:
//     present.N.key   = concat_key   (DynamicQuantize inserted here)
//     present.N.value = concat_value (DynamicQuantize inserted here)
//     attn_out.N      = MatMul2 output

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <unordered_map>

#include "npuw/kv_cache_compressed.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/openvino.hpp"
#include "ov_ops/dynamic_quantize.hpp"

namespace {

using namespace ov;

// ============================================================================
// Model builder
// ============================================================================

std::shared_ptr<Model> build_sdpa_model(size_t num_sdpa) {
    // [batch, heads, seq_past, head_dim]
    const Shape past_shape      = {1, 4, 8, 64};
    const Shape new_token_shape = {1, 4, 1, 64};
    const Shape mask_shape      = {1, 1, 1, 9};   // broadcastable over [1,4,1,9]

    ParameterVector params;
    ResultVector   results;

    for (size_t n = 0; n < num_sdpa; ++n) {
        const std::string idx = std::to_string(n);

        auto make_param = [&](const std::string& name, const Shape& shape) {
            auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
            p->set_friendly_name(name);
            p->output(0).get_tensor().set_names({name});
            params.push_back(p);
            return p;
        };

        auto past_key = make_param("past_key_values." + idx + ".key",   past_shape);
        auto past_val = make_param("past_key_values." + idx + ".value", past_shape);
        auto query    = make_param("query."    + idx, new_token_shape);
        auto new_key  = make_param("new_key."  + idx, new_token_shape);
        auto new_val  = make_param("new_value." + idx, new_token_shape);
        auto mask     = make_param("mask."     + idx, mask_shape);

        // Build SDPA: Q @ K^T -> Add(mask) -> Softmax -> @ V
        auto concat_key = std::make_shared<op::v0::Concat>(OutputVector{past_key, new_key}, 2);
        auto concat_val = std::make_shared<op::v0::Concat>(OutputVector{past_val, new_val}, 2);

        auto qk       = std::make_shared<op::v0::MatMul>(query, concat_key, false, true);
        auto add      = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
        auto softmax  = std::make_shared<op::v8::Softmax>(add->output(0), 3);
        auto attn_out = std::make_shared<op::v0::MatMul>(softmax->output(0), concat_val->output(0));

        auto make_result = [&](Output<Node> out, const std::string& name) {
            auto r = std::make_shared<op::v0::Result>(out);
            r->set_friendly_name(name);
            r->output(0).get_tensor().set_names({name});
            results.push_back(r);
        };

        make_result(concat_key->output(0), "present." + idx + ".key");
        make_result(concat_val->output(0), "present." + idx + ".value");
        make_result(attn_out->output(0),   "attn_out." + idx);
    }

    auto model = std::make_shared<Model>(results, params, "sdpa_test_" + std::to_string(num_sdpa));
    model->validate_nodes_and_infer_types();
    return model;
}

bool has_dynamic_quantize_ops(const std::shared_ptr<Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::internal::DynamicQuantize>(op)) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// Test parameters
// ============================================================================

using QuantizationType = ov::npuw::KVCacheCompressionConfig::QuantizationType;

struct CompressionPassParams {
    std::string    name;
    size_t         num_sdpa;
    QuantizationType quant_type;
    element::Type  quant_dt;
    // Expected newly-added params/results per single cache tensor (key or value):
    //   Asymmetric: scale + zp = 2
    //   Symmetric:  scale      = 1
    size_t added_per_cache;
};

std::ostream& operator<<(std::ostream& os, const CompressionPassParams& p) {
    return os << p.name;
}

class KVCacheCompressionPassTest : public ::testing::TestWithParam<CompressionPassParams> {
protected:
    ov::npuw::KVCacheCompressionParams make_compression_params(const CompressionPassParams& p) const {
        ov::npuw::KVCacheCompressionParams cp;
        cp.key   = {p.quant_type, p.quant_dt};
        cp.value = {p.quant_type, p.quant_dt};
        return cp;
    }
};

// ============================================================================
// Tests
// ============================================================================

// The model must still pass type-inference after the pass.
TEST_P(KVCacheCompressionPassTest, ModelValidAfterPass) {
    const auto& p = GetParam();
    auto model = build_sdpa_model(p.num_sdpa);

    ASSERT_NO_THROW(ov::npuw::run_kv_cache_dynamic_quantization_passes(model, make_compression_params(p)));
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

// The pass must inject exactly (num_sdpa * 2 * added_per_cache) new Parameters
// (scale + optionally zp, for both key and value past caches).
TEST_P(KVCacheCompressionPassTest, ParameterCountIncreasedCorrectly) {
    const auto& p = GetParam();
    auto model = build_sdpa_model(p.num_sdpa);
    const size_t params_before = model->get_parameters().size();
    // 2 caches (key + value) per SDPA, each getting added_per_cache new params
    const size_t expected_new = p.num_sdpa * 2 * p.added_per_cache;

    ASSERT_NO_THROW(ov::npuw::run_kv_cache_dynamic_quantization_passes(model, make_compression_params(p)));
    EXPECT_EQ(model->get_parameters().size(), params_before + expected_new);
}

// The pass must add exactly (num_sdpa * 2 * added_per_cache) new Results
// (scale + optionally zp, for both key and value present outputs).
TEST_P(KVCacheCompressionPassTest, ResultCountIncreasedCorrectly) {
    const auto& p = GetParam();
    auto model = build_sdpa_model(p.num_sdpa);
    const size_t results_before = model->get_results().size();
    const size_t expected_new   = p.num_sdpa * 2 * p.added_per_cache;

    ASSERT_NO_THROW(ov::npuw::run_kv_cache_dynamic_quantization_passes(model, make_compression_params(p)));
    EXPECT_EQ(model->get_results().size(), results_before + expected_new);
}

// All DynamicQuantize internal ops must be decomposed away by the end of the pass.
TEST_P(KVCacheCompressionPassTest, NoDynamicQuantizeOpsInDecomposedModel) {
    const auto& p = GetParam();
    auto model = build_sdpa_model(p.num_sdpa);

    ASSERT_NO_THROW(ov::npuw::run_kv_cache_dynamic_quantization_passes(model, make_compression_params(p)));
    EXPECT_FALSE(has_dynamic_quantize_ops(model));
}

// The present.N.key and present.N.value results must produce a quantized
// (non-floating-point) element type after the pass.
TEST_P(KVCacheCompressionPassTest, PresentCacheOutputIsQuantizedType) {
    const auto& p = GetParam();
    auto model = build_sdpa_model(p.num_sdpa);

    ASSERT_NO_THROW(ov::npuw::run_kv_cache_dynamic_quantization_passes(model, make_compression_params(p)));
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());

    for (const auto& result : model->get_results()) {
        const auto& name = result->get_friendly_name();
        // Only check the main quantized cache outputs; skip scale/zp results
        const bool is_main_cache = (name.find("present.") != std::string::npos) &&
                                   (name.find("/scale") == std::string::npos) &&
                                   (name.find("/zp")    == std::string::npos);
        if (!is_main_cache) continue;

        const auto type = result->get_input_element_type(0);
        EXPECT_TRUE(type.is_integral())
            << "Present cache result '" << name << "' should be an integer (quantized) type, got "
            << type;
    }
}

// With identity dequant parameters (scale=1, zp=0) the transformed model's
// attn_out must match the reference model's attn_out to within floating-point
// rounding tolerance.
TEST_P(KVCacheCompressionPassTest, AttnOutputMatchesWithIdentityDequant) {
    const auto& p = GetParam();

    auto ref_model   = build_sdpa_model(p.num_sdpa);
    auto xform_model = build_sdpa_model(p.num_sdpa);

    ASSERT_NO_THROW(ov::npuw::run_kv_cache_dynamic_quantization_passes(
        xform_model, make_compression_params(p)));
    ASSERT_NO_THROW(xform_model->validate_nodes_and_infer_types());

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::unordered_map<std::string, Tensor> orig_inputs;
    for (const auto& param : ref_model->get_parameters()) {
        Tensor t(element::f32, param->get_shape());
        auto* d = t.data<float>();
        for (size_t i = 0; i < t.get_size(); ++i) d[i] = dist(rng);
        orig_inputs[param->get_friendly_name()] = t;
    }

    TensorVector ref_inputs;
    for (const auto& param : ref_model->get_parameters()) {
        ref_inputs.push_back(orig_inputs.at(param->get_friendly_name()));
    }

    TensorVector xform_inputs;
    for (const auto& param : xform_model->get_parameters()) {
        auto it = orig_inputs.find(param->get_friendly_name());
        if (it != orig_inputs.end()) {
            xform_inputs.push_back(it->second);
        } else {
            // New scale/zp params: scale=1, zp=0 -> identity dequant
            Tensor t(param->get_element_type(), param->get_shape());
            if (param->get_element_type() == element::f32)
                std::fill_n(t.data<float>(), t.get_size(), 1.0f);
            else
                std::fill_n(t.data<uint8_t>(), t.get_size(), uint8_t(0));
            xform_inputs.push_back(t);
        }
    }

    auto make_output_tensors = [](const std::shared_ptr<Model>& m) {
        TensorVector outs;
        for (const auto& res : m->get_results())
            outs.emplace_back(res->get_output_element_type(0), res->get_output_shape(0));
        return outs;
    };

    auto ref_outputs   = make_output_tensors(ref_model);
    auto xform_outputs = make_output_tensors(xform_model);

    ASSERT_TRUE(ref_model->evaluate(ref_outputs, ref_inputs));
    ASSERT_TRUE(xform_model->evaluate(xform_outputs, xform_inputs));

    auto index_outputs = [](const std::shared_ptr<Model>& m, const TensorVector& outs) {
        std::unordered_map<std::string, const Tensor*> map;
        for (size_t i = 0; i < m->get_results().size(); ++i)
            map[m->get_results()[i]->get_friendly_name()] = &outs[i];
        return map;
    };

    auto ref_map   = index_outputs(ref_model,   ref_outputs);
    auto xform_map = index_outputs(xform_model, xform_outputs);

    for (size_t n = 0; n < p.num_sdpa; ++n) {
        const std::string name = "attn_out." + std::to_string(n);
        ASSERT_TRUE(ref_map.count(name))   << "Missing ref result: "   << name;
        ASSERT_TRUE(xform_map.count(name)) << "Missing xform result: " << name;

        const auto& ref_t   = *ref_map.at(name);
        const auto& xform_t = *xform_map.at(name);
        ASSERT_EQ(ref_t.get_shape(), xform_t.get_shape());
        ASSERT_EQ(ref_t.get_element_type(), element::f32);

        const float* ref_data   = ref_t.data<float>();
        const float* xform_data = xform_t.data<float>();
        const size_t count = ref_t.get_size();

        double sum_sq_err = 0.0, sum_sq_orig = 0.0;
        for (size_t i = 0; i < count; ++i) {
            double diff = static_cast<double>(ref_data[i]) - static_cast<double>(xform_data[i]);
            sum_sq_err  += diff * diff;
            sum_sq_orig += static_cast<double>(ref_data[i]) * static_cast<double>(ref_data[i]);
        }
        const double rel_l2 = (sum_sq_orig > 1e-12)
                              ? std::sqrt(sum_sq_err / sum_sq_orig)
                              : std::sqrt(sum_sq_err);
        EXPECT_LT(rel_l2, 1e-5)
            << "attn_out." << n << " diverges with identity dequant (rel L2=" << rel_l2 << ")";
    }
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    SingleSdpa,
    KVCacheCompressionPassTest,
    ::testing::Values(
        CompressionPassParams{"SingleSdpa_Asymmetric_u8",
            1, QuantizationType::Asymmetric, element::u8, 2},
        CompressionPassParams{"SingleSdpa_Symmetric_i8",
            1, QuantizationType::Symmetric,  element::i8, 1},
        CompressionPassParams{"SingleSdpa_Symmetric_i4",
            1, QuantizationType::Symmetric,  element::i4, 1}
    ),
    [](const ::testing::TestParamInfo<CompressionPassParams>& info) { return info.param.name; }
);

INSTANTIATE_TEST_SUITE_P(
    MultiSdpa,
    KVCacheCompressionPassTest,
    ::testing::Values(
        CompressionPassParams{"ThreeSdpa_Asymmetric_u8",
            3, QuantizationType::Asymmetric, element::u8, 2},
        CompressionPassParams{"ThreeSdpa_Symmetric_i8",
            3, QuantizationType::Symmetric,  element::i8, 1},
        CompressionPassParams{"ThreeSdpa_Symmetric_i4",
            3, QuantizationType::Symmetric,  element::i4, 1}
    ),
    [](const ::testing::TestParamInfo<CompressionPassParams>& info) { return info.param.name; }
);

// ============================================================================
// Multi-step decode simulation
// ============================================================================

// Single-head SDPA model (H=1) with a fixed past window of `window` tokens.
// H=1 keeps each token's data contiguous in memory, enabling simple slicing.
//
// Key layout:   {1, 1, seq, D}  -- tokens on axis 2, concat on axis 2
// Value layout: {1, 1, D, seq}  -- tokens on axis 3, concat on axis 3
// This mirrors the real model where V is transposed for MatMul(attn, V^T).
std::shared_ptr<Model> build_decode_step_model(size_t window) {
    constexpr size_t D = 64;
    const Shape past_key_shape  = {1, 1, window, D};
    const Shape past_val_shape  = {1, 1, D, window};
    const Shape new_key_shape   = {1, 1, 1, D};
    const Shape new_value_shape = {1, 1, D, 1};
    const Shape mask_shape      = {1, 1, 1, window + 1};

    ParameterVector params;
    ResultVector   results;

    auto make_param = [&](const std::string& n, const Shape& s) {
        auto p = std::make_shared<op::v0::Parameter>(element::f32, s);
        p->set_friendly_name(n);
        p->output(0).get_tensor().set_names({n});
        params.push_back(p);
        return p;
    };
    auto make_result = [&](Output<Node> out, const std::string& n) {
        auto r = std::make_shared<op::v0::Result>(out);
        r->set_friendly_name(n);
        r->output(0).get_tensor().set_names({n});
        results.push_back(r);
    };

    auto past_key = make_param("past_key_values.0.key",   past_key_shape);
    auto past_val = make_param("past_key_values.0.value", past_val_shape);
    auto query    = make_param("query.0",    new_key_shape);
    auto new_key  = make_param("new_key.0",  new_key_shape);
    auto new_val  = make_param("new_value.0", new_value_shape);
    auto mask     = make_param("mask.0",     mask_shape);

    auto concat_key = std::make_shared<op::v0::Concat>(OutputVector{past_key, new_key}, 2);
    auto concat_val = std::make_shared<op::v0::Concat>(OutputVector{past_val, new_val}, 3);
    auto qk         = std::make_shared<op::v0::MatMul>(query, concat_key, false, true);
    auto add_node   = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
    auto softmax    = std::make_shared<op::v8::Softmax>(add_node->output(0), -1);
    auto attn_out   = std::make_shared<op::v0::MatMul>(softmax->output(0), concat_val->output(0), false, true);

    make_result(concat_key->output(0), "present.0.key");
    make_result(concat_val->output(0), "present.0.value");
    make_result(attn_out->output(0),   "attn_out.0");

    auto m = std::make_shared<Model>(results, params, "decode_step");
    m->validate_nodes_and_infer_types();
    return m;
}

// Cast every element of a quantised tensor to float (integer value preserved).
Tensor cast_quant_to_f32(const Tensor& src) {
    Tensor dst(element::f32, src.get_shape());
    float* d = dst.data<float>();
    const size_t n = src.get_size();
    const auto et  = src.get_element_type();
    if (et == element::u8) {
        const uint8_t* s = src.data<uint8_t>();
        for (size_t i = 0; i < n; ++i) d[i] = float(s[i]);
    } else if (et == element::i8) {
        const int8_t* s = src.data<int8_t>();
        for (size_t i = 0; i < n; ++i) d[i] = float(s[i]);
    } else if (et == element::i4) {
        const uint8_t* s = static_cast<const uint8_t*>(src.data());
        for (size_t i = 0; i < n; ++i) {
            const uint8_t byte = s[i / 2];
            int val = (i % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            if (val > 7) val -= 16;
            d[i] = float(val);
        }
    } else {
        ADD_FAILURE() << "cast_quant_to_f32: unsupported element type " << et;
        throw std::runtime_error("Unsupported element type");
    }
    return dst;
}

// ---------------------------------------------------------------------------
// Key slicing helpers  (key layout: {1, 1, seq, D}, tokens on axis 2)
// ---------------------------------------------------------------------------

// Slice last `keep` tokens (rows) from [1, 1, total, D] -- contiguous tail.
template<typename T>
Tensor slice_last_tokens_h1(const Tensor& src, size_t keep) {
    const size_t total = src.get_shape()[2];
    const size_t D     = src.get_shape()[3];
    const size_t off   = (total - keep) * D;
    Tensor dst(src.get_element_type(), Shape{1, 1, keep, D});
    const T* sd = src.data<T>();
    std::copy(sd + off, sd + off + keep * D, dst.data<T>());
    return dst;
}

// Slice last `keep` per-token scales from [1, 1, total, 1] f32 tensor.
Tensor slice_last_scales_h1(const Tensor& src, size_t keep) {
    const size_t off = src.get_shape()[2] - keep;
    Tensor dst(element::f32, Shape{1, 1, keep, 1});
    const float* sd = src.data<float>();
    std::copy(sd + off, sd + off + keep, dst.data<float>());
    return dst;
}

// Slice last `keep` per-token zero-points from [1, 1, total, 1] u8 tensor.
Tensor slice_last_zp_h1(const Tensor& src, size_t keep) {
    const size_t off = src.get_shape()[2] - keep;
    Tensor dst(src.get_element_type(), Shape{1, 1, keep, 1});
    const uint8_t* sd = src.data<uint8_t>();
    std::copy(sd + off, sd + off + keep, dst.data<uint8_t>());
    return dst;
}

// ---------------------------------------------------------------------------
// Value slicing helpers  (value layout: {1, 1, D, seq}, tokens on axis 3)
// ---------------------------------------------------------------------------

// Slice last `keep` columns from [1, 1, rows, total] -- strided: pick the
// last `keep` elements from each of `rows` rows.
template<typename T>
Tensor slice_last_cols_h1(const Tensor& src, size_t keep) {
    const size_t rows  = src.get_shape()[2];   // D
    const size_t total = src.get_shape()[3];   // seq
    const size_t col_off = total - keep;
    Tensor dst(src.get_element_type(), Shape{1, 1, rows, keep});
    const T* sd = src.data<T>();
    T* dd = dst.data<T>();
    for (size_t r = 0; r < rows; ++r) {
        std::copy(sd + r * total + col_off,
                  sd + r * total + col_off + keep,
                  dd + r * keep);
    }
    return dst;
}

// Slice last `keep` per-channel scales from [1, 1, 1, total] f32 tensor.
Tensor slice_last_scales_col(const Tensor& src, size_t keep) {
    const size_t total = src.get_shape()[3];
    const size_t off   = total - keep;
    Tensor dst(element::f32, Shape{1, 1, 1, keep});
    const float* sd = src.data<float>();
    std::copy(sd + off, sd + off + keep, dst.data<float>());
    return dst;
}

// Slice last `keep` per-channel zero-points from [1, 1, 1, total] u8 tensor.
Tensor slice_last_zp_col(const Tensor& src, size_t keep) {
    const size_t total = src.get_shape()[3];
    const size_t off   = total - keep;
    Tensor dst(src.get_element_type(), Shape{1, 1, 1, keep});
    const uint8_t* sd = src.data<uint8_t>();
    std::copy(sd + off, sd + off + keep, dst.data<uint8_t>());
    return dst;
}

// Independent key/value quantisation configuration for the decode-loop test.
struct DecodeLoopParams {
    std::string      name;
    QuantizationType key_quant_type;
    element::Type    key_dt;
    QuantizationType val_quant_type;
    element::Type    val_dt;

    ov::npuw::KVCacheCompressionParams to_compression_params() const {
        ov::npuw::KVCacheCompressionParams p;
        p.key   = {key_quant_type, key_dt};
        p.value = {val_quant_type, val_dt};
        return p;
    }
};

class KVCacheMultiStepDecodeTest : public ::testing::TestWithParam<DecodeLoopParams> {};

TEST_P(KVCacheMultiStepDecodeTest, DecodeLoopAccuracy) {
    const auto& p = GetParam();

    constexpr size_t WINDOW  = 2;
    constexpr size_t N_STEPS = 3;
    constexpr size_t D       = 64;

    const bool key_asym = (p.key_quant_type == QuantizationType::Asymmetric);
    const bool val_asym = (p.val_quant_type == QuantizationType::Asymmetric);

    auto ref_model   = build_decode_step_model(WINDOW);
    auto xform_model = build_decode_step_model(WINDOW);
    ASSERT_NO_THROW(ov::npuw::run_kv_cache_dynamic_quantization_passes(
        xform_model, p.to_compression_params()));
    ASSERT_NO_THROW(xform_model->validate_nodes_and_infer_types());
    ov::save_model(xform_model, "xform_model_" + p.name + ".xml");
    ov::save_model(ref_model, "ref_model_" + p.name + ".xml");

    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto zero_f32 = [](Shape sh) {
        Tensor t(element::f32, sh); std::fill_n(t.data<float>(), t.get_size(), 0.0f); return t; };
    auto ones_f32 = [](Shape sh) {
        Tensor t(element::f32, sh); std::fill_n(t.data<float>(), t.get_size(), 1.0f); return t; };
    auto zero_u8 = [](Shape sh) {
        Tensor t(element::u8, sh); std::memset(t.data(), 0, t.get_byte_size()); return t; };

    // Key: {1,1,WINDOW,D}, Value: {1,1,D,WINDOW}
    Tensor ref_past_key = zero_f32({1, 1, WINDOW, D});
    Tensor ref_past_val = zero_f32({1, 1, D, WINDOW});

    // Key scale: per-token {1,1,W,1}  (reduce axis 3 = D)
    // Value scale: per-channel {1,1,1,W}  (reduce axis 2 = D in transposed layout)
    Tensor xf_past_key  = zero_f32({1, 1, WINDOW, D});
    Tensor xf_key_scale = ones_f32({1, 1, WINDOW, 1});
    Tensor xf_key_zp    = zero_u8({1, 1, WINDOW, 1});
    Tensor xf_past_val  = zero_f32({1, 1, D, WINDOW});
    Tensor xf_val_scale = ones_f32({1, 1, 1, WINDOW});
    Tensor xf_val_zp    = zero_u8({1, 1, 1, WINDOW});

    // Key: tokens on axis 2 -> slice_last_tokens_h1
    auto slice_and_cast_key = [&](const Tensor& src) -> Tensor {
        const auto et = src.get_element_type();
        if (et == element::u8) return cast_quant_to_f32(slice_last_tokens_h1<uint8_t>(src, WINDOW));
        if (et == element::i8) return cast_quant_to_f32(slice_last_tokens_h1<int8_t> (src, WINDOW));
        return slice_last_tokens_h1<float>(cast_quant_to_f32(src), WINDOW);
    };

    // Value: tokens on axis 3 -> slice_last_cols_h1
    auto slice_and_cast_val = [&](const Tensor& src) -> Tensor {
        const auto et = src.get_element_type();
        if (et == element::u8) return cast_quant_to_f32(slice_last_cols_h1<uint8_t>(src, WINDOW));
        if (et == element::i8) return cast_quant_to_f32(slice_last_cols_h1<int8_t> (src, WINDOW));
        return slice_last_cols_h1<float>(cast_quant_to_f32(src), WINDOW);
    };

    const double tol = (p.key_dt == element::i4 || p.val_dt == element::i4) ? 0.5 : 0.05;

    for (size_t step = 0; step < N_STEPS; ++step) {
        Tensor t_query(element::f32, Shape{1, 1, 1, D});
        Tensor t_new_key(element::f32, Shape{1, 1, 1, D});
        Tensor t_new_val(element::f32, Shape{1, 1, D, 1});
        Tensor t_mask(element::f32, Shape{1, 1, 1, WINDOW + 1});
        for (auto* t : {&t_query, &t_new_key, &t_new_val})
            for (size_t i = 0; i < t->get_size(); ++i) t->data<float>()[i] = dist(rng);
        std::fill_n(t_mask.data<float>(), t_mask.get_size(), 0.0f);

        // -- Reference inference --
        TensorVector ref_inputs;
        for (const auto& param : ref_model->get_parameters()) {
            const std::string& n = param->get_friendly_name();
            if      (n == "past_key_values.0.key")   ref_inputs.push_back(ref_past_key);
            else if (n == "past_key_values.0.value") ref_inputs.push_back(ref_past_val);
            else if (n == "query.0")     ref_inputs.push_back(t_query);
            else if (n == "new_key.0")   ref_inputs.push_back(t_new_key);
            else if (n == "new_value.0") ref_inputs.push_back(t_new_val);
            else if (n == "mask.0")      ref_inputs.push_back(t_mask);
            else FAIL() << "Unexpected ref param: " << n;
        }
        TensorVector ref_outputs;
        for (const auto& res : ref_model->get_results())
            ref_outputs.emplace_back(res->get_output_element_type(0), res->get_output_shape(0));
        ASSERT_TRUE(ref_model->evaluate(ref_outputs, ref_inputs))
            << "step " << step << ": ref evaluate() failed";

        std::unordered_map<std::string, Tensor> rout;
        for (size_t i = 0; i < ref_model->get_results().size(); ++i)
            rout[ref_model->get_results()[i]->get_friendly_name()] = ref_outputs[i];

        // Key present {1,1,W+1,D} -> slice last W rows (axis 2)
        ref_past_key = slice_last_tokens_h1<float>(rout.at("present.0.key"), WINDOW);
        // Value present {1,1,D,W+1} -> slice last W columns (axis 3)
        ref_past_val = slice_last_cols_h1<float>(rout.at("present.0.value"), WINDOW);

        // -- Compressed inference --
        TensorVector xform_inputs;
        for (const auto& param : xform_model->get_parameters()) {
            const std::string& n = param->get_friendly_name();
            if      (n == "past_key_values.0.key")                         xform_inputs.push_back(xf_past_key);
            else if (n == "DynamicQuantize/0/past_key_values/key/scale")   xform_inputs.push_back(xf_key_scale);
            else if (n == "DynamicQuantize/0/past_key_values/key/zp")      xform_inputs.push_back(xf_key_zp);
            else if (n == "past_key_values.0.value")                       xform_inputs.push_back(xf_past_val);
            else if (n == "DynamicQuantize/0/past_key_values/value/scale") xform_inputs.push_back(xf_val_scale);
            else if (n == "DynamicQuantize/0/past_key_values/value/zp")    xform_inputs.push_back(xf_val_zp);
            else if (n == "query.0")     xform_inputs.push_back(t_query);
            else if (n == "new_key.0")   xform_inputs.push_back(t_new_key);
            else if (n == "new_value.0") xform_inputs.push_back(t_new_val);
            else if (n == "mask.0")      xform_inputs.push_back(t_mask);
            else FAIL() << "Unexpected compressed param: " << n;
        }
        TensorVector xform_outputs;
        for (const auto& res : xform_model->get_results())
            xform_outputs.emplace_back(res->get_output_element_type(0), res->get_output_shape(0));
        ASSERT_TRUE(xform_model->evaluate(xform_outputs, xform_inputs))
            << "step " << step << ": compressed evaluate() failed";

        std::unordered_map<std::string, Tensor> xout;
        for (size_t i = 0; i < xform_model->get_results().size(); ++i)
            xout[xform_model->get_results()[i]->get_friendly_name()] = xform_outputs[i];

        // -- Compare attn_out.0 --
        {
            const Tensor& xt = xout.at("attn_out.0");
            const Tensor& rt = rout.at("attn_out.0");
            ASSERT_EQ(xt.get_shape(), rt.get_shape());
            const float* xd = xt.data<float>(), *rd = rt.data<float>();
            const size_t cnt = xt.get_size();
            double ss_err = 0, ss_ref = 0;
            for (size_t i = 0; i < cnt; ++i) {
                double e = double(xd[i]) - double(rd[i]);
                ss_err += e * e;
                ss_ref += double(rd[i]) * double(rd[i]);
            }
            const double rel = (ss_ref > 1e-12) ? std::sqrt(ss_err / ss_ref) : std::sqrt(ss_err);
            EXPECT_LT(rel, tol)
                << "Step " << step << ": attn_out.0 rel-L2 = " << rel;
        }

        // -- Advance compressed past state --
        // Key: {1,1,W+1,D} -> cast + slice rows (axis 2)
        xf_past_key  = slice_and_cast_key(xout.at("present.0.key"));
        // Value: {1,1,D,W+1} -> cast + slice cols (axis 3)
        xf_past_val  = slice_and_cast_val(xout.at("present.0.value"));
        // Key scale {1,1,W+1,1} -> slice rows (axis 2)
        xf_key_scale = slice_last_scales_h1(xout.at("DynamicQuantize/0/present/key/scale"), WINDOW);
        if (key_asym) xf_key_zp = slice_last_zp_h1(xout.at("DynamicQuantize/0/present/key/zp"), WINDOW);
        // Value scale {1,1,1,W+1} -> slice cols (axis 3)
        xf_val_scale = slice_last_scales_col(xout.at("DynamicQuantize/0/present/value/scale"), WINDOW);
        if (val_asym) xf_val_zp = slice_last_zp_col(xout.at("DynamicQuantize/0/present/value/zp"), WINDOW);
    }
}

INSTANTIATE_TEST_SUITE_P(
    DecodeLoop,
    KVCacheMultiStepDecodeTest,
    ::testing::Values(
        DecodeLoopParams{"Key_Asym_u8__Val_Asym_u8",
            QuantizationType::Asymmetric, element::u8,
            QuantizationType::Asymmetric, element::u8},
        DecodeLoopParams{"Key_Sym_i8__Val_Sym_i8",
            QuantizationType::Symmetric, element::i8,
            QuantizationType::Symmetric, element::i8},
        DecodeLoopParams{"Key_Sym_i4__Val_Sym_i4",
            QuantizationType::Symmetric, element::i4,
            QuantizationType::Symmetric, element::i4},
        DecodeLoopParams{"Key_Asym_u8__Val_Sym_i8",
            QuantizationType::Asymmetric, element::u8,
            QuantizationType::Symmetric,  element::i8},
        DecodeLoopParams{"Key_Sym_i4__Val_Asym_u8",
            QuantizationType::Symmetric,  element::i4,
            QuantizationType::Asymmetric, element::u8},
        DecodeLoopParams{"Key_Asym_u8__Val_Sym_i4",
            QuantizationType::Asymmetric, element::u8,
            QuantizationType::Symmetric,  element::i4}
    ),
    [](const ::testing::TestParamInfo<DecodeLoopParams>& info) { return info.param.name; }
);

}  // namespace
