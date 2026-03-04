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

    for (size_t i = 0; i < cpu_out.size(); ++i) {
        std::cerr << "[PA_VERIFY] step " << i
                  << " CPU  shape=[" << cpu_out[i].get_shape()[0] << "," << cpu_out[i].get_shape()[1] << "]"
                  << " TMPL shape=[" << tmpl_out[i].get_shape()[0] << "," << tmpl_out[i].get_shape()[1] << "]\n";
        const float* cpu_ptr  = cpu_out[i].data<float>();
        const float* tmpl_ptr = tmpl_out[i].data<float>();
        const size_t n = cpu_out[i].get_size();
        const size_t dump = std::min<size_t>(n, 16u);
        std::cerr << "[PA_VERIFY]   CPU  first " << dump << " vals:";
        for (size_t j = 0; j < dump; ++j) std::cerr << " " << cpu_ptr[j];
        std::cerr << "\n";
        std::cerr << "[PA_VERIFY]   TMPL first " << dump << " vals:";
        for (size_t j = 0; j < dump; ++j) std::cerr << " " << tmpl_ptr[j];
        std::cerr << "\n";
        float max_diff = 0;
        for (size_t j = 0; j < n; ++j) max_diff = std::max(max_diff, std::abs(cpu_ptr[j] - tmpl_ptr[j]));
        std::cerr << "[PA_VERIFY]   max_abs_diff=" << max_diff << "\n";
        ov::test::utils::compare(tmpl_out[i], cpu_out[i], /*abs*/1e-3f, /*rel*/1e-2f);
    }
}

}  // namespace test
}  // namespace ov
