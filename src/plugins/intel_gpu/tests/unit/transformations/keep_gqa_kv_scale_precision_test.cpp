// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/keep_gqa_kv_scale_precision.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

using ov::op::internal::GroupQueryAttention;
using ov::op::internal::GroupQueryAttentionQuantType;

// k_scale / v_scale are inputs 12 / 13 of GroupQueryAttention.
constexpr size_t k_scale_idx = 12;
constexpr size_t v_scale_idx = 13;

// Builds a 14-input GroupQueryAttention model. When quantized, the past KV cache is i8 and the op
// carries kv_cache_bit_width=8 + a per-channel quant scheme, so is_kv_quantized() is true; otherwise
// the cache is f16 with the default (unquantized) attributes. k_scale/v_scale (inputs 12/13) are f32
// in both cases so the no-op test can assert on the exact ports the pass targets.
std::shared_ptr<ov::Model> make_gqa_model(bool quantized) {
    const auto kv_et = quantized ? ov::element::i8 : ov::element::f16;

    ov::ParameterVector params;
    ov::OutputVector args(14);

    const auto add_param = [&](size_t idx, const ov::element::Type& et, const ov::PartialShape& pshape) {
        auto p = std::make_shared<ov::op::v0::Parameter>(et, pshape);
        params.push_back(p);
        args[idx] = p;
    };

    add_param(0, ov::element::f16, ov::PartialShape{1, 2, 1, 16});  // query
    add_param(1, ov::element::f16, ov::PartialShape{1, 1, 1, 16});  // key
    add_param(2, ov::element::f16, ov::PartialShape{1, 1, 1, 16});  // value
    add_param(3, kv_et, ov::PartialShape{1, 1, 8, 16});             // past_key
    add_param(4, kv_et, ov::PartialShape{1, 1, 8, 16});             // past_value
    add_param(5, ov::element::i32, ov::PartialShape{1});            // seqlens_k
    add_param(6, ov::element::i32, ov::PartialShape{1});            // total_sequence_length
    for (size_t i = 7; i <= 11; ++i)                                // empty Constants are treated as absent optional inputs by has_input()
        args[i] = ov::op::v0::Constant::create(ov::element::dynamic, ov::Shape{0}, {});
    add_param(k_scale_idx, ov::element::f32, ov::PartialShape{1});  // k_scale
    add_param(v_scale_idx, ov::element::f32, ov::PartialShape{1});  // v_scale

    // Keep this test compatible with both string-based and enum-based quant type APIs.
    const auto quant_type = quantized ? GroupQueryAttentionQuantType::PER_CHANNEL : GroupQueryAttentionQuantType::NONE;

    auto gqa = std::make_shared<GroupQueryAttention>(args,
                                                     /*num_heads*/ 2,
                                                     /*kv_num_heads*/ 1,
                                                     /*scale*/ 0.0f,
                                                     /*do_rotary*/ false,
                                                     /*rotary_interleaved*/ false,
                                                     /*kv_cache_bit_width*/ quantized ? 8 : 0,
                                                     /*k_quant_type*/ quant_type,
                                                     /*v_quant_type*/ quant_type);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gqa->output(0)),
                             std::make_shared<ov::op::v0::Result>(gqa->output(1)),
                             std::make_shared<ov::op::v0::Result>(gqa->output(2))};
    return std::make_shared<ov::Model>(results, params, "gqa_model");
}

std::shared_ptr<GroupQueryAttention> get_gqa(const std::shared_ptr<ov::Model>& model) {
    return ov::as_type_ptr<GroupQueryAttention>(model->get_results().at(0)->input_value(0).get_node_shared_ptr());
}

}  // namespace

// Quantized KV cache: the pass must mark k_scale (12) and v_scale (13) as precision-sensitive so that
// ConvertPrecision keeps them fp32. The model_ref comparison (RUNTIME_KEYS) reliably catches a missing mark.
TEST_F(TransformationTestsF, KeepGQAKVScalePrecisionMarksQuantizedKVScales) {
    model = make_gqa_model(/*quantized*/ true);

    model_ref = make_gqa_model(/*quantized*/ true);
    auto gqa_ref = get_gqa(model_ref);
    ov::mark_as_precision_sensitive(gqa_ref->input(k_scale_idx));
    ov::mark_as_precision_sensitive(gqa_ref->input(v_scale_idx));

    manager.register_pass<KeepGQAKVScalePrecision>();
}

// Explicit positive check, independent of the fixture comparator.
TEST(KeepGQAKVScalePrecisionTest, QuantizedKVScalesMarked) {
    auto model = make_gqa_model(/*quantized*/ true);
    auto gqa = get_gqa(model);

    ov::pass::Manager manager;
    manager.register_pass<KeepGQAKVScalePrecision>();
    manager.run_passes(model);

    EXPECT_TRUE(ov::is_precision_sensitive(gqa->input(k_scale_idx)));
    EXPECT_TRUE(ov::is_precision_sensitive(gqa->input(v_scale_idx)));
}

// Non-quantized KV cache: is_kv_quantized() is false, so the pass must be a no-op. The directional
// comparator cannot catch an erroneous mark, so assert the absence explicitly.
TEST(KeepGQAKVScalePrecisionTest, NonQuantizedKVIsNoOp) {
    auto model = make_gqa_model(/*quantized*/ false);
    auto gqa = get_gqa(model);

    ov::pass::Manager manager;
    manager.register_pass<KeepGQAKVScalePrecision>();
    manager.run_passes(model);

    EXPECT_FALSE(ov::is_precision_sensitive(gqa->input(k_scale_idx)));
    EXPECT_FALSE(ov::is_precision_sensitive(gqa->input(v_scale_idx)));
}
