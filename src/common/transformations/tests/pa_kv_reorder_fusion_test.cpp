// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pa_kv_reorder_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/pa_kv_reorder.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;

TEST(PaKVReorderFusionTest, PassInstantiation) {
    auto pass = std::make_shared<ov::pass::PaKVReorderFusion>();
    ASSERT_NE(pass, nullptr);
}

TEST(PaKVReorderFusionTest, EmptyModel) {
    using namespace ov;
    auto model = std::make_shared<Model>(ResultVector{}, ParameterVector{});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PaKVReorderFusion>();
    manager.run_passes(model);

    ASSERT_EQ(model->get_results().size(), 0);
}

TEST(PaKVReorderFusionTest, NoPatternToFuse) {
    using namespace ov;
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
    auto result = std::make_shared<ov::op::v0::Result>(input);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input});

    size_t ops_before = model->get_ops().size();

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PaKVReorderFusion>();
    manager.run_passes(model);

    size_t ops_after = model->get_ops().size();
    ASSERT_EQ(ops_before, ops_after);
}

TEST(PaKVReorderFusionTest, FusionPattern) {
    using namespace ov;

    auto key_cache = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    key_cache->set_friendly_name("key_cache.0_clone_for_k_update");

    auto value_cache = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    value_cache->set_friendly_name("value_cache.0_clone_for_v_update");

    auto block_update_indices = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});
    block_update_indices->set_friendly_name("block_update_indices");

    auto block_indices = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});
    block_indices->set_friendly_name("block_indices");

    auto block_indices_begins = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    block_indices_begins->set_friendly_name("block_indices_begins");

    auto block_update_indices_begins = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    block_update_indices_begins->set_friendly_name("block_update_indices_begins");

    auto axis = ov::op::v0::Constant::create(element::i32, Shape{}, {0});

    auto key_gather = std::make_shared<ov::op::v8::Gather>(key_cache, block_update_indices, axis);
    auto key_scatter = std::make_shared<ov::op::v3::ScatterUpdate>(key_cache, block_indices, key_gather, axis);
    key_scatter->set_friendly_name("updated_key_cache_0");

    auto value_gather = std::make_shared<ov::op::v8::Gather>(value_cache, block_update_indices, axis);
    auto value_scatter = std::make_shared<ov::op::v3::ScatterUpdate>(value_cache, block_indices, value_gather, axis);
    value_scatter->set_friendly_name("updated_value_cache_0");

    auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{key_scatter, value_scatter}, 0);
    auto result = std::make_shared<ov::op::v0::Result>(concat);

    auto model = std::make_shared<Model>(ResultVector{result},
                                         ParameterVector{key_cache,
                                                         value_cache,
                                                         block_indices,
                                                         block_indices_begins,
                                                         block_update_indices,
                                                         block_update_indices_begins});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PaKVReorderFusion>();
    manager.run_passes(model);

    bool found_pa_kv_reorder = false;
    int gather_count = 0;
    int scatter_count = 0;

    for (const auto& op : model->get_ops()) {
        if (std::dynamic_pointer_cast<ov::op::internal::PaKVReorder>(op)) {
            found_pa_kv_reorder = true;
        }
        if (std::dynamic_pointer_cast<ov::op::v8::Gather>(op)) {
            gather_count++;
        }
        if (std::dynamic_pointer_cast<ov::op::v3::ScatterUpdate>(op)) {
            scatter_count++;
        }
    }

    ASSERT_TRUE(found_pa_kv_reorder) << "PaKVReorder op should be created after fusion";
    ASSERT_EQ(gather_count, 0) << "Gather ops should be removed after fusion";
    ASSERT_EQ(scatter_count, 0) << "ScatterUpdate ops should be removed after fusion";
}

TEST_F(TransformationTestsF, PaKVReorderFusion_basic) {
    disable_result_friendly_names_check();
    {
        auto key_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        key_cache->set_friendly_name("key_cache.0_clone_for_k_update");
        auto value_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        value_cache->set_friendly_name("value_cache.0_clone_for_v_update");

        auto block_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        block_indices->set_friendly_name("block_indices");
        auto block_indices_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        block_indices_begins->set_friendly_name("block_indices_begins");

        auto block_update_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        block_update_indices->set_friendly_name("block_update_indices");
        auto block_update_indices_begins =
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        block_update_indices_begins->set_friendly_name("block_update_indices_begins");

        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto scatter_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});

        auto key_gather = std::make_shared<ov::op::v8::Gather>(key_cache, block_update_indices, gather_axis);
        auto value_gather = std::make_shared<ov::op::v8::Gather>(value_cache, block_update_indices, gather_axis);

        auto key_scatter =
            std::make_shared<ov::op::v3::ScatterUpdate>(key_cache, block_indices, key_gather, scatter_axis);
        auto value_scatter =
            std::make_shared<ov::op::v3::ScatterUpdate>(value_cache, block_indices, value_gather, scatter_axis);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{key_scatter, value_scatter}, 0);
        auto result = std::make_shared<ov::op::v0::Result>(concat);

        model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                            ov::ParameterVector{key_cache,
                                                                value_cache,
                                                                block_indices,
                                                                block_indices_begins,
                                                                block_update_indices,
                                                                block_update_indices_begins});

        manager.register_pass<ov::pass::PaKVReorderFusion>();
    }

    {
        auto key_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        key_cache->set_friendly_name("key_cache.0_clone_for_k_update");
        auto value_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        value_cache->set_friendly_name("value_cache.0_clone_for_v_update");

        auto block_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        block_indices->set_friendly_name("block_indices");
        auto block_indices_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        block_indices_begins->set_friendly_name("block_indices_begins");

        auto block_update_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        block_update_indices->set_friendly_name("block_update_indices");
        auto block_update_indices_begins =
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        block_update_indices_begins->set_friendly_name("block_update_indices_begins");

        auto pa_kv_reorder = std::make_shared<ov::op::internal::PaKVReorder>(key_cache,
                                                                             value_cache,
                                                                             block_indices,
                                                                             block_indices_begins,
                                                                             block_update_indices,
                                                                             block_update_indices_begins);
        pa_kv_reorder->set_friendly_name("pa_kv_reorder_0");
        auto result = std::make_shared<ov::op::v0::Result>(pa_kv_reorder);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result},
                                                ov::ParameterVector{key_cache,
                                                                    value_cache,
                                                                    block_indices,
                                                                    block_indices_begins,
                                                                    block_update_indices,
                                                                    block_update_indices_begins});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, PaKVReorderFusion_skip_on_mismatched_block_indices) {
    disable_result_friendly_names_check();
    auto key_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
    key_cache->set_friendly_name("key_cache.0_clone_for_k_update");
    auto value_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
    value_cache->set_friendly_name("value_cache.0_clone_for_v_update");

    auto block_indices_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
    block_indices_k->set_friendly_name("block_indices_k");
    auto block_indices_v = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
    block_indices_v->set_friendly_name("block_indices_v");

    auto block_indices_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
    block_indices_begins->set_friendly_name("block_indices_begins");
    auto block_update_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
    block_update_indices->set_friendly_name("block_update_indices");
    auto block_update_indices_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
    block_update_indices_begins->set_friendly_name("block_update_indices_begins");

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto scatter_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});

    auto key_gather = std::make_shared<ov::op::v8::Gather>(key_cache, block_update_indices, gather_axis);
    auto value_gather = std::make_shared<ov::op::v8::Gather>(value_cache, block_update_indices, gather_axis);

    auto key_scatter =
        std::make_shared<ov::op::v3::ScatterUpdate>(key_cache, block_indices_k, key_gather, scatter_axis);
    auto value_scatter =
        std::make_shared<ov::op::v3::ScatterUpdate>(value_cache, block_indices_v, value_gather, scatter_axis);

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{key_scatter, value_scatter}, 0);
    auto result = std::make_shared<ov::op::v0::Result>(concat);

    model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                        ov::ParameterVector{key_cache,
                                                            value_cache,
                                                            block_indices_k,
                                                            block_indices_v,
                                                            block_indices_begins,
                                                            block_update_indices,
                                                            block_update_indices_begins});

    manager.register_pass<ov::pass::PaKVReorderFusion>();

    model_ref = model->clone();
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

TEST(PaKVReorderOpTest, OpCreation) {
    using namespace ov;
    auto key_cache = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    auto value_cache = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    auto block_indices = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{8});
    auto block_indices_begins = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    auto block_update_indices = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});
    auto block_update_indices_begins = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});

    auto pa_kv_reorder = std::make_shared<ov::op::internal::PaKVReorder>(key_cache,
                                                                         value_cache,
                                                                         block_indices,
                                                                         block_indices_begins,
                                                                         block_update_indices,
                                                                         block_update_indices_begins);

    ASSERT_NE(pa_kv_reorder, nullptr);
    ASSERT_EQ(pa_kv_reorder->get_input_size(), 6);
    ASSERT_EQ(pa_kv_reorder->get_output_size(), 1);
}

TEST(PaKVReorderOpTest, ModelWithOp) {
    using namespace ov;
    auto key_cache = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    key_cache->set_friendly_name("key_cache");

    auto value_cache = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    value_cache->set_friendly_name("value_cache");

    auto block_indices = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{8});
    block_indices->set_friendly_name("block_indices");

    auto block_indices_begins = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    block_indices_begins->set_friendly_name("block_indices_begins");

    auto block_update_indices = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});
    block_update_indices->set_friendly_name("block_update_indices");

    auto block_update_indices_begins = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    block_update_indices_begins->set_friendly_name("block_update_indices_begins");

    auto pa_kv_reorder = std::make_shared<ov::op::internal::PaKVReorder>(key_cache,
                                                                         value_cache,
                                                                         block_indices,
                                                                         block_indices_begins,
                                                                         block_update_indices,
                                                                         block_update_indices_begins);

    auto result = std::make_shared<ov::op::v0::Result>(pa_kv_reorder->output(0));

    auto model = std::make_shared<Model>(ResultVector{result},
                                         ParameterVector{key_cache,
                                                         value_cache,
                                                         block_indices,
                                                         block_indices_begins,
                                                         block_update_indices,
                                                         block_update_indices_begins});

    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->get_parameters().size(), 6);
    ASSERT_EQ(model->get_results().size(), 1);

    bool found_pa_kv_reorder = false;
    for (const auto& op : model->get_ops()) {
        if (std::dynamic_pointer_cast<ov::op::internal::PaKVReorder>(op)) {
            found_pa_kv_reorder = true;
            break;
        }
    }
    ASSERT_TRUE(found_pa_kv_reorder) << "PaKVReorder op not found in model";
}

TEST(PaKVReorderOpTest, TypeInfo) {
    using namespace ov;
    auto key_cache = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    auto value_cache = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 8, 32, 64});
    auto block_indices = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{8});
    auto block_indices_begins = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    auto block_update_indices = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});
    auto block_update_indices_begins = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});

    auto pa_kv_reorder = std::make_shared<ov::op::internal::PaKVReorder>(key_cache,
                                                                         value_cache,
                                                                         block_indices,
                                                                         block_indices_begins,
                                                                         block_update_indices,
                                                                         block_update_indices_begins);

    auto type_info = pa_kv_reorder->get_type_info();
    ASSERT_STREQ(type_info.name, "PaKVReorder");
}
