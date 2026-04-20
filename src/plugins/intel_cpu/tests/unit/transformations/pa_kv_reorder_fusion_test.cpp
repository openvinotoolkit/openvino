// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/cpu_opset/common/op/pa_kv_reorder.hpp"
#include "transformations/cpu_opset/common/pass/pa_kv_reorder_fusion.hpp"

// Test that PaKVReorderFusion pass is registered and can be instantiated
TEST(PaKVReorderFusionTest, PassInstantiation) {
    auto pass = std::make_shared<ov::intel_cpu::PaKVReorderFusion>();
    ASSERT_NE(pass, nullptr);
}

// Test that the pass doesn't crash on empty model
TEST(PaKVReorderFusionTest, EmptyModel) {
    using namespace ov;
    auto model = std::make_shared<Model>(ResultVector{}, ParameterVector{});

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::PaKVReorderFusion>();
    manager.run_passes(model);

    ASSERT_EQ(model->get_results().size(), 0);
}

// Test that the pass doesn't modify model without matching pattern
TEST(PaKVReorderFusionTest, NoPatternToFuse) {
    using namespace ov;
    // Simple model: Input -> Result
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
    auto result = std::make_shared<op::v0::Result>(input);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input});

    size_t ops_before = model->get_ops().size();

    ov::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::PaKVReorderFusion>();
    manager.run_passes(model);

    size_t ops_after = model->get_ops().size();

    // Model should remain unchanged
    ASSERT_EQ(ops_before, ops_after);
}

// Test that Gather->ScatterUpdate pattern is fused into PaKVReorder
TEST(PaKVReorderFusionTest, FusionPattern) {
    using namespace ov;

    // Create model with key and value cache Gather->ScatterUpdate pattern
    // Key cache: cache -> Gather -> ScatterUpdate
    auto key_cache = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    key_cache->set_friendly_name("key_cache.0_clone_for_k_update");

    auto value_cache = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    value_cache->set_friendly_name("value_cache.0_clone_for_v_update");

    // Indices for Gather (which blocks to copy from)
    auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    block_update_indices->set_friendly_name("block_update_indices");

    // Indices for ScatterUpdate (which blocks to copy to)
    auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    block_indices->set_friendly_name("block_indices");

    // Required parameters for fusion
    auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    block_indices_begins->set_friendly_name("block_indices_begins");

    auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    block_update_indices_begins->set_friendly_name("block_update_indices_begins");

    // Axis = 0 (gather/scatter along block dimension)
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});

    // Key path: key_cache -> Gather -> ScatterUpdate
    auto key_gather = std::make_shared<op::v8::Gather>(key_cache, block_update_indices, axis);
    auto key_scatter = std::make_shared<op::v3::ScatterUpdate>(key_cache, block_indices, key_gather, axis);
    key_scatter->set_friendly_name("updated_key_cache_0");

    // Value path: value_cache -> Gather -> ScatterUpdate
    auto value_gather = std::make_shared<op::v8::Gather>(value_cache, block_update_indices, axis);
    auto value_scatter = std::make_shared<op::v3::ScatterUpdate>(value_cache, block_indices, value_gather, axis);
    value_scatter->set_friendly_name("updated_value_cache_0");

    // Concat key and value outputs
    auto concat = std::make_shared<op::v0::Concat>(OutputVector{key_scatter, value_scatter}, 0);

    auto result = std::make_shared<op::v0::Result>(concat);

    auto model = std::make_shared<Model>(
        ResultVector{result},
        ParameterVector{key_cache, value_cache, block_indices, block_indices_begins,
                       block_update_indices, block_update_indices_begins});

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::PaKVReorderFusion>();
    manager.run_passes(model);

    // Verify that PaKVReorder op was created
    bool found_pa_kv_reorder = false;
    int gather_count = 0;
    int scatter_count = 0;

    for (const auto& op : model->get_ops()) {
        if (std::dynamic_pointer_cast<ov::intel_cpu::op::PaKVReorder>(op)) {
            found_pa_kv_reorder = true;
        }
        if (std::dynamic_pointer_cast<op::v8::Gather>(op)) {
            gather_count++;
        }
        if (std::dynamic_pointer_cast<op::v3::ScatterUpdate>(op)) {
            scatter_count++;
        }
    }

    // After fusion, there should be PaKVReorder and no Gather/ScatterUpdate ops
    ASSERT_TRUE(found_pa_kv_reorder) << "PaKVReorder op should be created after fusion";
    ASSERT_EQ(gather_count, 0) << "Gather ops should be removed after fusion";
    ASSERT_EQ(scatter_count, 0) << "ScatterUpdate ops should be removed after fusion";
}

// Test that PaKVReorder op can be created
TEST(PaKVReorderOpTest, OpCreation) {
    using namespace ov;
    auto key_cache = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    auto value_cache = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{8});
    auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    auto pa_kv_reorder = std::make_shared<ov::intel_cpu::op::PaKVReorder>(
        key_cache, value_cache, block_indices, block_indices_begins,
        block_update_indices, block_update_indices_begins);

    ASSERT_NE(pa_kv_reorder, nullptr);
    ASSERT_EQ(pa_kv_reorder->get_input_size(), 6);
    ASSERT_EQ(pa_kv_reorder->get_output_size(), 1);
}

// Note: Input validation is done at compile-time via constructor signature
// The PaKVReorder constructor requires all 6 inputs, so passing fewer will fail to compile

// Test model with PaKVReorder op can be created
TEST(PaKVReorderOpTest, ModelWithOp) {
    using namespace ov;
    auto key_cache = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    key_cache->set_friendly_name("key_cache");

    auto value_cache = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    value_cache->set_friendly_name("value_cache");

    auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{8});
    block_indices->set_friendly_name("block_indices");

    auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    block_indices_begins->set_friendly_name("block_indices_begins");

    auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    block_update_indices->set_friendly_name("block_update_indices");

    auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    block_update_indices_begins->set_friendly_name("block_update_indices_begins");

    auto pa_kv_reorder = std::make_shared<ov::intel_cpu::op::PaKVReorder>(
        key_cache, value_cache, block_indices, block_indices_begins,
        block_update_indices, block_update_indices_begins);

    auto result = std::make_shared<op::v0::Result>(pa_kv_reorder->output(0));

    auto model = std::make_shared<Model>(
        ResultVector{result},
        ParameterVector{key_cache, value_cache, block_indices, block_indices_begins,
                       block_update_indices, block_update_indices_begins});

    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->get_parameters().size(), 6);
    ASSERT_EQ(model->get_results().size(), 1);

    // Verify that PaKVReorder op exists in the model
    bool found_pa_kv_reorder = false;
    for (const auto& op : model->get_ops()) {
        if (std::dynamic_pointer_cast<ov::intel_cpu::op::PaKVReorder>(op)) {
            found_pa_kv_reorder = true;
            break;
        }
    }
    ASSERT_TRUE(found_pa_kv_reorder) << "PaKVReorder op not found in model";
}

// Test that PaKVReorder op has correct type info
TEST(PaKVReorderOpTest, TypeInfo) {
    using namespace ov;
    auto key_cache = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    auto value_cache = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{8});
    auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    auto pa_kv_reorder = std::make_shared<ov::intel_cpu::op::PaKVReorder>(
        key_cache, value_cache, block_indices, block_indices_begins,
        block_update_indices, block_update_indices_begins);

    auto type_info = pa_kv_reorder->get_type_info();
    ASSERT_STREQ(type_info.name, "PaKVReorder");
}
