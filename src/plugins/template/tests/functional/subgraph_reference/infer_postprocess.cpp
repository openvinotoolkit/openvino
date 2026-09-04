// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"

namespace ov {
namespace test {
namespace {

// infer_postprocess() regression test: ICompiledModel freezes its public output shape before
// CommonOptimizations runs, so an output resolved dynamic->static only by a decomposition pass
// inside CommonOptimizations must still get copied, not skipped as "already up to date".
// GroupQueryAttention triggers this directly: present_key/value start dynamic (tied to query's
// dynamic batch) and the decomposition's ScatterUpdate branch makes them static (past_key's shape).
// Verified via OV_ENABLE_SERIALIZE_TRACING IR dumps: [-1,1,4,16] before, [1,1,4,16] after.
TEST(TemplateInferPostprocess, PresentKeyValueCopiedAfterShapeResolvesPostDecomposition) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 1, 1, 16});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 1, 1, 16});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 1, 1, 16});
    const auto past_key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 1, 4, 16});
    const auto past_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 1, 4, 16});
    const auto seqlens_k = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});

    const auto gqa = std::make_shared<op::internal::GroupQueryAttention>(
        OutputVector{query, key, value, past_key, past_value, seqlens_k},
        /*num_heads=*/1,
        /*kv_num_heads=*/1,
        /*scale=*/1.0f,
        /*do_rotary=*/false,
        /*rotary_interleaved=*/false);

    const auto output = std::make_shared<op::v0::Result>(gqa->output(0));
    const auto present_key = std::make_shared<op::v0::Result>(gqa->output(1));
    const auto present_value = std::make_shared<op::v0::Result>(gqa->output(2));
    const auto model = std::make_shared<Model>(ResultVector{output, present_key, present_value},
                                               ParameterVector{query, key, value, past_key, past_value, seqlens_k});

    ASSERT_TRUE(model->output(1).get_partial_shape().is_dynamic())
        << "Precondition: present_key must be declared dynamic before decomposition runs, "
           "otherwise this test no longer exercises the bug it targets.";

    auto core = ov::test::utils::PluginCache::get().core("TEMPLATE");
    auto compiled_model = core->compile_model(model, "TEMPLATE");
    auto infer_request = compiled_model.create_infer_request();

    Tensor query_tensor(element::f32, Shape{1, 1, 1, 16});
    std::fill_n(query_tensor.data<float>(), query_tensor.get_size(), 0.f);

    Tensor key_tensor(element::f32, Shape{1, 1, 1, 16});
    std::fill_n(key_tensor.data<float>(), key_tensor.get_size(), 9.f);
    Tensor value_tensor(element::f32, Shape{1, 1, 1, 16});
    std::fill_n(value_tensor.data<float>(), value_tensor.get_size(), 90.f);

    // Rows 0-3 filled with 1/2/3/4 (past_key) and 10/20/30/40 (past_value), 16 elements per row.
    Tensor past_key_tensor(element::f32, Shape{1, 1, 4, 16});
    Tensor past_value_tensor(element::f32, Shape{1, 1, 4, 16});
    for (size_t row = 0; row < 4; ++row) {
        std::fill_n(past_key_tensor.data<float>() + row * 16, 16, static_cast<float>(row + 1));
        std::fill_n(past_value_tensor.data<float>() + row * 16, 16, static_cast<float>((row + 1) * 10));
    }

    Tensor seqlens_tensor(element::i32, Shape{1});
    seqlens_tensor.data<int32_t>()[0] = 3;  // past_seqlen(3) + current_seqlen(1) - 1

    infer_request.set_tensor(compiled_model.input(0), query_tensor);
    infer_request.set_tensor(compiled_model.input(1), key_tensor);
    infer_request.set_tensor(compiled_model.input(2), value_tensor);
    infer_request.set_tensor(compiled_model.input(3), past_key_tensor);
    infer_request.set_tensor(compiled_model.input(4), past_value_tensor);
    infer_request.set_tensor(compiled_model.input(5), seqlens_tensor);
    infer_request.infer();

    // scatter_idx = [past_seqlen] = [3]: only row 3 of the capacity-4 buffer is overwritten by the new
    // token; rows 0-2 keep past_key/past_value's values unchanged.
    std::vector<float> expected_present_key(64);
    std::vector<float> expected_present_value(64);
    for (size_t row = 0; row < 4; ++row) {
        const float key_val = row < 3 ? static_cast<float>(row + 1) : 9.f;
        const float value_val = row < 3 ? static_cast<float>((row + 1) * 10) : 90.f;
        std::fill_n(expected_present_key.begin() + row * 16, 16, key_val);
        std::fill_n(expected_present_value.begin() + row * 16, 16, value_val);
    }

    const auto actual_present_key = infer_request.get_tensor(compiled_model.output(1));
    const auto actual_present_value = infer_request.get_tensor(compiled_model.output(2));

    ASSERT_EQ(actual_present_key.get_shape(), Shape({1, 1, 4, 16}));
    ASSERT_EQ(actual_present_value.get_shape(), Shape({1, 1, 4, 16}));

    const auto* key_ptr = actual_present_key.data<float>();
    const auto* value_ptr = actual_present_value.data<float>();
    for (size_t i = 0; i < expected_present_key.size(); ++i) {
        EXPECT_FLOAT_EQ(key_ptr[i], expected_present_key[i]) << "present_key mismatch at index " << i;
        EXPECT_FLOAT_EQ(value_ptr[i], expected_present_value[i]) << "present_value mismatch at index " << i;
    }
}

}  // namespace
}  // namespace test
}  // namespace ov
