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
    const auto& [inType, inputShapes, extendBlockIndices, enableXattn, sinkInput, slidingWindow, additional_config] = GetParam();
    (void)inputShapes; (void)enableXattn; (void)sinkInput; (void)slidingWindow;

    // CPU config
    ov::AnyMap cpu_cfg = additional_config;
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
    auto cpu_out = run_device(pa_model_, ov::test::utils::DEVICE_CPU, cpu_cfg, extendBlockIndices, steps_cpu);

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
    auto tmpl_out = run_device(pa_model_, ov::test::utils::DEVICE_TEMPLATE, tmpl_cfg, extendBlockIndices, steps_tmpl);

    for (size_t i = 0; i < cpu_out.size(); ++i)
        ov::test::utils::compare(tmpl_out[i], cpu_out[i], /*abs*/1e-3f, /*rel*/1e-2f);
}

}  // namespace test
}  // namespace ov
