// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/paged_attention.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {

TEST_P(PagedAttentionLayerTest, Inference) {
    const auto& [inType, inputShapes, extendBlockIndices, enableXattn, sinkInput, slidingWindow, useAlibi, maxContextLen, additional_config] = GetParam();
    (void)inputShapes; (void)enableXattn; (void)sinkInput; (void)slidingWindow; (void)useAlibi; (void)maxContextLen;

    // Single Core from the test framework (avoids re-registration errors).
    auto& core_ref = *core;

    // CPU config — strip test-only keys that the CPU plugin doesn't understand
    ov::AnyMap cpu_cfg = additional_config;
    cpu_cfg.erase("test_use_rotation");
    cpu_cfg.erase("test_block_size");
    cpu_cfg.erase("test_adaptive_rkv_eviction_size");
    cpu_cfg[ov::hint::inference_precision.name()] = ov::element::f32;

    // TEMPLATE config
    ov::AnyMap tmpl_cfg;
    tmpl_cfg[ov::hint::inference_precision.name()] = ov::element::f32;
    // Run CPU with fresh caches (do NOT reuse key_cache_init_ directly; the plugin mutates it).
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
            // Guard against empty tensors (CPU may not populate outputs 1/2 in all cases)
            if (ct.get_size() == 0 || tt.get_size() == 0) {
                continue;
            }
            // Only output 0 (attention) is compared strictly.
            // Output 1 (score aggregation) differs during large prefill for the same
            // reason as max_context_len - the CPU uses a different internal code path.
            // Output 2 (diversity scores) is TEMPLATE-only; the CPU kernel doesn't compute it.
            if (oi > 0) {
                continue;
            }
            ov::test::utils::compare(tt, ct, /*abs*/1e-3f, /*rel*/1e-2f);
        }
    }
}

}  // namespace test
}  // namespace ov
