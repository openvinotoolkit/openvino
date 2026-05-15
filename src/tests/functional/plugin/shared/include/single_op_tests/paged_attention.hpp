// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/paged_attention.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace ov {
namespace test {

TEST_P(PagedAttentionLayerTest, Inference) {
    const auto& [inType, inputShapes, extendBlockIndices, enableXattn, sinkInput, slidingWindow, useAlibi, maxContextLen, additional_config] = GetParam();
    (void)inputShapes; (void)enableXattn; (void)sinkInput; (void)slidingWindow; (void)useAlibi; (void)maxContextLen;

#ifdef OPENVINO_ARCH_X86_64
    if (inType == ov::element::f16 && !ov::with_cpu_x86_avx512_core_fp16())
        GTEST_SKIP() << "f16 PA requires AVX512-FP16 hardware";
    if (inType == ov::element::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP() << "bf16 PA requires BF16 hardware";
#endif

    // Use a single ov::Core to avoid re-registration errors
    auto& core_ref = *core;

    // Read per-instantiation comparison thresholds (default: 1e-3 abs, 1e-2 rel)
    float test_abs_threshold = 1e-3f;
    float test_rel_threshold = 1e-2f;
    {
        auto it = additional_config.find("test_abs_threshold");
        if (it != additional_config.end()) test_abs_threshold = it->second.as<float>();
        it = additional_config.find("test_rel_threshold");
        if (it != additional_config.end()) test_rel_threshold = it->second.as<float>();
    }

    // Strip test-only keys from CPU config.
    // inference_precision=f32 keeps ConvertPrecision from converting f32 Constants
    // (scale, alibi) to f16/bf16, which would corrupt PA kernel arithmetic.
    ov::AnyMap cpu_cfg = additional_config;
    cpu_cfg.erase("test_use_rotation");
    cpu_cfg.erase("test_block_size");
    cpu_cfg.erase("test_adaptive_rkv_eviction_size");
    cpu_cfg.erase("test_abs_threshold");
    cpu_cfg.erase("test_rel_threshold");
    cpu_cfg[ov::hint::inference_precision.name()] = ov::element::f32;

    ov::AnyMap tmpl_cfg;
    tmpl_cfg[ov::hint::inference_precision.name()] = ov::element::f32;
    // Run CPU with fresh cache copies (the plugin mutates them in-place)
    ov::Tensor key_cache_cpu(key_cache_init_.get_element_type(), key_cache_init_.get_shape());
    ov::Tensor value_cache_cpu(value_cache_init_.get_element_type(), value_cache_init_.get_shape());
    key_cache_init_.copy_to(key_cache_cpu);
    value_cache_init_.copy_to(value_cache_cpu);

    auto steps_cpu = steps_;
    for (auto& s : steps_cpu) {
        s.tensors[pa_model_->get_parameters()[3]] = key_cache_cpu;
        s.tensors[pa_model_->get_parameters()[4]] = value_cache_cpu;
    }
    auto cpu_out = run_device(core_ref, pa_model_, ov::test::utils::DEVICE_CPU, cpu_cfg, extendBlockIndices, steps_cpu);

    // Run TEMPLATE with an independent fresh cache copy
    ov::Tensor key_cache_tmpl(key_cache_init_.get_element_type(), key_cache_init_.get_shape());
    ov::Tensor value_cache_tmpl(value_cache_init_.get_element_type(), value_cache_init_.get_shape());
    key_cache_init_.copy_to(key_cache_tmpl);
    value_cache_init_.copy_to(value_cache_tmpl);

    auto steps_tmpl = steps_;
    for (auto& s : steps_tmpl) {
        s.tensors[pa_model_->get_parameters()[3]] = key_cache_tmpl;
        s.tensors[pa_model_->get_parameters()[4]] = value_cache_tmpl;
    }
    auto tmpl_out = run_device(core_ref, pa_model_, ov::test::utils::DEVICE_TEMPLATE, tmpl_cfg, extendBlockIndices, steps_tmpl);

    OPENVINO_ASSERT(cpu_out.size() == tmpl_out.size(), "PA verify: step count mismatch");
    for (size_t i = 0; i < cpu_out.size(); ++i) {
        const size_t n_outs = cpu_out[i].size();
        OPENVINO_ASSERT(n_outs == tmpl_out[i].size(), "PA verify: output count mismatch at step ", i);
        for (size_t oi = 0; oi < n_outs; ++oi) {
            const auto& ct = cpu_out[i][oi];
            const auto& tt = tmpl_out[i][oi];
            // Guard against empty tensors
            if (ct.get_size() == 0 || tt.get_size() == 0) {
                continue;
            }
            // Output 2 (diversity): only computed by TEMPLATE, skip comparison
            if (oi > 1) {
                continue;
            }
            ov::test::utils::compare(tt, ct, test_abs_threshold, test_rel_threshold);
        }
    }
}

// Verify that score_aggregation_window=0 produces all-zero output 1.
TEST_P(PagedAttentionLayerTest, ScoreWindowZeroZerosOutput1) {
    const auto& [inType, inputShapes, extendBlockIndices, enableXattn, sinkInput, slidingWindow, useAlibi, maxContextLen, additional_config] = GetParam();
    (void)inputShapes; (void)enableXattn; (void)sinkInput; (void)slidingWindow; (void)useAlibi; (void)maxContextLen;

#ifdef OPENVINO_ARCH_X86_64
    if (inType == ov::element::f16 && !ov::with_cpu_x86_avx512_core_fp16())
        GTEST_SKIP() << "f16 PA requires AVX512-FP16 hardware";
    if (inType == ov::element::bf16 && !ov::with_cpu_x86_bfloat16())
        GTEST_SKIP() << "bf16 PA requires BF16 hardware";
#endif

    int64_t cfg_block_size = 32;
    {
        auto it = additional_config.find("test_block_size");
        if (it != additional_config.end())
            cfg_block_size = static_cast<int64_t>(it->second.as<int>());
    }

    OPENVINO_ASSERT(!targetStaticShapes.empty() && targetStaticShapes[0][0].size() == 4,
                    "PagedAttention test expects [L,B,H,S] shapes");
    const int64_t head_num  = static_cast<int64_t>(targetStaticShapes[0][0][2]);
    const int64_t head_size = static_cast<int64_t>(targetStaticShapes[0][0][3]);

    // Build model with score_aggregation_window = 0 and output 1 wired
    auto model_zero = make_paged_attn_model(inType, /*enable_xattn=*/false, head_size, head_num,
                                            /*use_sink=*/false, /*sliding_window=*/0, /*use_alibi=*/false,
                                            /*max_ctx_len=*/maxContextLen, /*use_rotation=*/false,
                                            cfg_block_size, /*arkv=*/0, /*score_window_val=*/0);

    // Allocate fresh zero-initialized caches for this model
    ov::Tensor kc0(inType, key_cache_init_.get_shape());
    ov::Tensor vc0(inType, value_cache_init_.get_shape());
    std::memset(kc0.data(), 0, kc0.get_byte_size());
    std::memset(vc0.data(), 0, vc0.get_byte_size());

    // Build step inputs against the new model's parameters
    int32_t past = 0;
    std::vector<StepInputs> steps_zero;
    steps_zero.reserve(targetStaticShapes.size());
    for (size_t i = 0; i < targetStaticShapes.size(); ++i) {
        steps_zero.push_back(make_step_inputs(model_zero, targetStaticShapes[i][0],
                                              static_cast<int>(i), extendBlockIndices,
                                              kc0, vc0, past));
    }

    ov::AnyMap tmpl_cfg;
    tmpl_cfg[ov::hint::inference_precision.name()] = ov::element::f32;

    // Strip test-only keys; inference_precision=f32 keeps f32 Constants (scale, alibi) unconverted.
    ov::AnyMap cpu_cfg = additional_config;
    cpu_cfg.erase("test_use_rotation");
    cpu_cfg.erase("test_block_size");
    cpu_cfg.erase("test_adaptive_rkv_eviction_size");
    cpu_cfg.erase("test_abs_threshold");
    cpu_cfg.erase("test_rel_threshold");
    cpu_cfg[ov::hint::inference_precision.name()] = ov::element::f32;

    ov::Tensor kc_cpu(inType, key_cache_init_.get_shape());
    ov::Tensor vc_cpu(inType, value_cache_init_.get_shape());
    std::memset(kc_cpu.data(), 0, kc_cpu.get_byte_size());
    std::memset(vc_cpu.data(), 0, vc_cpu.get_byte_size());
    auto steps_cpu = steps_zero;
    for (auto& s : steps_cpu) {
        s.tensors[model_zero->get_parameters()[3]] = kc_cpu;
        s.tensors[model_zero->get_parameters()[4]] = vc_cpu;
    }

    auto tmpl_out = run_device(*core, model_zero, ov::test::utils::DEVICE_TEMPLATE,
                               tmpl_cfg, extendBlockIndices, steps_zero);
    auto cpu_out  = run_device(*core, model_zero, ov::test::utils::DEVICE_CPU,
                               cpu_cfg, extendBlockIndices, steps_cpu);

    for (size_t step = 0; step < tmpl_out.size(); ++step) {
        for (const auto* outs : {&tmpl_out[step], &cpu_out[step]}) {
            ASSERT_GE(outs->size(), 2u) << "Expected at least 2 outputs at step " << step;
            const auto& scores = (*outs)[1];
            ASSERT_GT(scores.get_size(), 0u) << "score output is empty at step " << step;
            // Read scores as f32 regardless of storage type
            const auto et = scores.get_element_type();
            for (size_t j = 0; j < scores.get_size(); ++j) {
                float v = 0.f;
                if (et == ov::element::f32)
                    v = static_cast<const float*>(scores.data())[j];
                else if (et == ov::element::f16)
                    v = static_cast<float>(static_cast<const ov::float16*>(scores.data())[j]);
                else if (et == ov::element::bf16)
                    v = static_cast<float>(static_cast<const ov::bfloat16*>(scores.data())[j]);
                EXPECT_EQ(v, 0.f)
                    << "output 1 not zero at step=" << step << " index=" << j
                    << " (score_aggregation_window=0)";
            }
        }
    }
}

}  // namespace test
}  // namespace ov
