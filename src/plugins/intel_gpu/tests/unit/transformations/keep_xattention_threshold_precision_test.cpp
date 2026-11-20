// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/keep_xattention_threshold_precision.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "transformations/paged_attention/convert_pagedattn_inputs.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "plugin/transformations/paged_attention_bf16_fallback.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

using PAExt = ov::op::PagedAttentionExtension;

std::shared_ptr<ov::Model> make_pa_model_with_threshold_et(const ov::element::Type& thr_et) {
    // Build a minimal model containing PagedAttentionExtension with valid 28 inputs.
    // We keep shapes mostly dynamic but MUST satisfy rank/type checks in
    // ov::op::PagedAttentionExtension::validate_and_infer_types().

    const size_t thr_idx = cldnn::paged_attention::PagedAttentionInputIdx::XATTENTION_THRESHOLD;

    ov::ParameterVector params(28);
    // 0..4: query/key/value/key_cache/value_cache
    params[0] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(2));
    params[1] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(2));
    params[2] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(2));
    params[3] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic());
    params[4] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic());

    // 5..8: i32 vectors
    for (size_t i = 5; i <= 8; i++)
        params[i] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(1));

    // 9: scale scalar real
    params[9] = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(0));
    // 10: sliding_window scalar i32
    params[10] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(0));
    // 11: alibi_slopes rank1 real
    params[11] = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(1));
    // 12: max_context_len scalar i32
    params[12] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(0));
    // 13: score_aggregation_window rank0/1 i32
    params[13] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(0));

    // 14: rotated_block_indices rank1 i32
    params[14] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(1));
    // 15: rotation_deltas rank1/2 i32
    params[15] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(1));
    // 16: rotation_trig_lut rank1/2 f16/f32
    params[16] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(1));

    // 17: xattention_threshold rank1 f16/f32 (or other type in negative test)
    params[thr_idx] = std::make_shared<ov::op::v0::Parameter>(thr_et, ov::PartialShape::dynamic(1));

    // 18..19: scalars i32
    params[18] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(0));
    params[19] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(0));

    // 20: sinks rank1/4 any type (use f16)
    params[20] = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(1));

    // 21: scalar i32
    params[21] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(0));
    // 22..25: vectors/matrices i32
    params[22] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(1));
    params[23] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(1));
    params[24] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(1));
    params[25] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(1));
    // 26..27: qq_bias inputs
    params[26] = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape::dynamic(1));
    params[27] = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape::dynamic(1));

    ov::OutputVector inputs;
    inputs.reserve(params.size());
    for (const auto& p : params)
        inputs.emplace_back(p);

    auto pa = std::make_shared<PAExt>(inputs);
    auto res = std::make_shared<ov::op::v0::Result>(pa);
    return std::make_shared<ov::Model>(ov::ResultVector{res}, params);
}

std::shared_ptr<PAExt> get_pa_ext(const std::shared_ptr<ov::Model>& model) {
    return ov::as_type_ptr<PAExt>(model->get_results().at(0)->input_value(0).get_node_shared_ptr());
}

std::shared_ptr<ov::Model> make_ref_pa_model_with_precision_sensitive_threshold(const ov::element::Type& thr_et) {
    auto model = make_pa_model_with_threshold_et(thr_et);
    const size_t thr_idx = cldnn::paged_attention::PagedAttentionInputIdx::XATTENTION_THRESHOLD;
    auto pa = get_pa_ext(model);
    OPENVINO_ASSERT(pa != nullptr);
    ov::mark_as_precision_sensitive(pa->input(thr_idx));
    return model;
}

}  // namespace

TEST_F(TransformationTestsF, KeepXAttentionThresholdPrecisionMarksF32Threshold) {
    model = make_pa_model_with_threshold_et(ov::element::f32);
    model_ref = make_ref_pa_model_with_precision_sensitive_threshold(ov::element::f32);

    manager.register_pass<KeepXAttentionThresholdPrecision>();
}

TEST_F(TransformationTestsF, KeepXAttentionThresholdPrecisionMarksF16Threshold) {
    // PagedAttentionExtension validation allows only f16/f32 on xattention_threshold port,
    // so only the valid real-type cases can be covered here.
    model = make_pa_model_with_threshold_et(ov::element::f16);
    model_ref = make_ref_pa_model_with_precision_sensitive_threshold(ov::element::f16);

    manager.register_pass<KeepXAttentionThresholdPrecision>();
}

TEST_F(TransformationTestsF, KeepXAttentionThresholdPrecisionPreservesPagedAttentionInputsInBF16) {
    using InputIdx = cldnn::paged_attention::PagedAttentionInputIdx;
    constexpr auto lut_idx = InputIdx::ROTATION_TRIG_LUT;
    constexpr auto threshold_idx = InputIdx::XATTENTION_THRESHOLD;
    model = make_pa_model_with_threshold_et(ov::element::f16);
    auto pa = get_pa_ext(model);
    model->remove_parameter(ov::as_type_ptr<ov::op::v0::Parameter>(pa->get_input_node_shared_ptr(lut_idx)));
    auto empty_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, {});
    pa->input(lut_idx).replace_source_output(empty_lut);
    pa->validate_and_infer_types();

    model_ref = model->clone();
    auto ref_pa = get_pa_ext(model_ref);
    auto threshold = ov::as_type_ptr<ov::op::v0::Parameter>(ref_pa->get_input_node_shared_ptr(threshold_idx));
    for (const auto& param : model_ref->get_parameters()) {
        if (param != threshold && param->get_element_type().is_real()) {
            param->set_element_type(ov::element::bf16);
        }
    }
    ov::mark_as_precision_sensitive(ref_pa->input(threshold_idx));
    ov::disable_conversion(threshold, ov::element::f16, ov::element::bf16);
    ov::disable_conversion(ref_pa->get_input_node_shared_ptr(lut_idx), ov::element::f32, ov::element::bf16);
    model_ref->validate_nodes_and_infer_types();

    manager.register_pass<KeepXAttentionThresholdPrecision>(true);
    manager.register_pass<ov::pass::ConvertPrecision>(
        precisions_map{{ov::element::f16, ov::element::bf16}, {ov::element::f32, ov::element::bf16}},
        type_to_fuse_map{},
        true,
        true);
}

TEST_F(TransformationTestsF, PagedAttentionBF16FallbackUsesFP16AndKeepsBF16ModelOutput) {
    using Idx = cldnn::paged_attention::PagedAttentionInputIdx;
    model = make_pa_model_with_threshold_et(ov::element::f32);
    auto pa = get_pa_ext(model);
    for (const auto idx : {Idx::QUERY, Idx::KEY, Idx::VALUE, Idx::KEY_CACHE, Idx::VALUE_CACHE}) {
        auto param = ov::as_type_ptr<ov::op::v0::Parameter>(pa->get_input_node_shared_ptr(idx));
        param->set_element_type(idx == Idx::KEY_CACHE || idx == Idx::VALUE_CACHE ? ov::element::f32 : ov::element::bf16);
    }
    model->validate_nodes_and_infer_types();

    model_ref = model->clone();
    auto ref_pa = get_pa_ext(model_ref);
    for (const auto idx : {Idx::KEY_CACHE, Idx::VALUE_CACHE}) {
        auto param = ov::as_type_ptr<ov::op::v0::Parameter>(ref_pa->get_input_node_shared_ptr(idx));
        param->set_element_type(ov::element::f16);
        ov::enable_keep_const_precision(param);
    }
    for (const auto idx : {Idx::QUERY, Idx::KEY, Idx::VALUE}) {
        ref_pa->input(idx).replace_source_output(
            std::make_shared<ov::op::v0::Convert>(ref_pa->input_value(idx), ov::element::f16));
    }
    ref_pa->validate_and_infer_types();
    model_ref->get_results().front()->input(0).replace_source_output(
        std::make_shared<ov::op::v0::Convert>(ref_pa, ov::element::bf16));
    model_ref->validate_nodes_and_infer_types();

    ov::pass::ConvertPagedAttnInputs::KVCacheConfig cache_config;
    cache_config.inferencePrecision = ov::element::f16;
    cache_config.keyCachePrecision = ov::element::f16;
    cache_config.valueCachePrecision = ov::element::f16;
    manager.register_pass<ov::pass::ConvertPagedAttnInputs>(
        cache_config,
        [](ov::element::Type, bool, size_t, int64_t&, int64_t&) {});
    manager.register_pass<PagedAttentionBF16Fallback>();
}

TEST_F(TransformationTestsF, PagedAttentionBF16FallbackConvertsAuxInputsAndConnectedOutputs) {
    using Idx = cldnn::paged_attention::PagedAttentionInputIdx;
    model = make_pa_model_with_threshold_et(ov::element::f32);
    auto pa = get_pa_ext(model);
    for (const auto idx : {Idx::QUERY, Idx::KEY, Idx::VALUE, Idx::SCALE, Idx::ALIBI, Idx::SINKS}) {
        auto param = ov::as_type_ptr<ov::op::v0::Parameter>(pa->get_input_node_shared_ptr(idx));
        param->set_element_type(ov::element::bf16);
    }
    auto res1 = std::make_shared<ov::op::v0::Result>(pa->output(1));
    auto res2 = std::make_shared<ov::op::v0::Result>(pa->output(2));
    model->add_results({res1, res2});
    model->validate_nodes_and_infer_types();

    model_ref = model->clone();
    auto ref_pa = get_pa_ext(model_ref);
    for (const auto idx : {Idx::QUERY, Idx::KEY, Idx::VALUE, Idx::SCALE, Idx::ALIBI, Idx::SINKS}) {
        ref_pa->input(idx).replace_source_output(
            std::make_shared<ov::op::v0::Convert>(ref_pa->input_value(idx), ov::element::f16));
    }
    for (int i = 0; i < 3; ++i) {
        ref_pa->set_out_type(i, ov::element::f16);
    }
    ref_pa->validate_and_infer_types();
    for (size_t i = 0; i < 3; ++i) {
        model_ref->get_results().at(i)->input(0).replace_source_output(
            std::make_shared<ov::op::v0::Convert>(ref_pa->output(i), ov::element::bf16));
    }
    model_ref->validate_nodes_and_infer_types();

    manager.register_pass<PagedAttentionBF16Fallback>();
}

TEST_F(TransformationTestsF, PagedAttentionBF16FallbackSkipsNativeBF16AndQuantizedCache) {
    using Idx = cldnn::paged_attention::PagedAttentionInputIdx;
    for (const auto cache_et : {ov::element::bf16, ov::element::i8}) {
        model = make_pa_model_with_threshold_et(ov::element::f32);
        auto pa = get_pa_ext(model);
        for (const auto idx : {Idx::QUERY, Idx::KEY, Idx::VALUE}) {
            auto param = ov::as_type_ptr<ov::op::v0::Parameter>(pa->get_input_node_shared_ptr(idx));
            param->set_element_type(ov::element::bf16);
        }
        for (const auto idx : {Idx::KEY_CACHE, Idx::VALUE_CACHE}) {
            auto param = ov::as_type_ptr<ov::op::v0::Parameter>(pa->get_input_node_shared_ptr(idx));
            param->set_element_type(cache_et);
        }
        model->validate_nodes_and_infer_types();
        model_ref = model->clone();

        ov::pass::Manager pass_manager;
        pass_manager.register_pass<PagedAttentionBF16Fallback>();
        pass_manager.run_passes(model);

        auto res = compare_functions(model, model_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}
