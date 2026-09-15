// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "config.h"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/pa_kv_reorder.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/pass/pa_kv_reorder_fusion.hpp"
#include "transformations/transformation_pipeline.h"

namespace ov::intel_cpu {

class PaKVReorderPrecisionTest : public testing::TestWithParam<element::Type> {};

TEST_P(PaKVReorderPrecisionTest, KeepsCacheInputsInStoragePrecision) {
    const auto cache_precision = GetParam();
    auto key_cache = std::make_shared<op::v0::Parameter>(cache_precision, Shape{4, 8, 32, 64});
    key_cache->set_friendly_name("key_cache.0_clone_for_k_update");
    auto value_cache = std::make_shared<op::v0::Parameter>(cache_precision, Shape{4, 8, 32, 64});
    value_cache->set_friendly_name("value_cache.0_clone_for_v_update");
    auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    block_indices->set_friendly_name("block_indices");
    auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    block_indices_begins->set_friendly_name("block_indices_begins");
    auto block_update_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{4});
    block_update_indices->set_friendly_name("block_update_indices");
    auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    block_update_indices_begins->set_friendly_name("block_update_indices_begins");

    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto key_gather = std::make_shared<op::v8::Gather>(key_cache, block_update_indices, axis);
    auto value_gather = std::make_shared<op::v8::Gather>(value_cache, block_update_indices, axis);
    auto key_scatter = std::make_shared<op::v3::ScatterUpdate>(key_cache, block_indices, key_gather, axis);
    auto value_scatter = std::make_shared<op::v3::ScatterUpdate>(value_cache, block_indices, value_gather, axis);
    auto concat = std::make_shared<op::v0::Concat>(OutputVector{key_scatter, value_scatter}, 0);
    auto model = std::make_shared<Model>(OutputVector{concat},
                                         ParameterVector{key_cache,
                                                         value_cache,
                                                         block_indices,
                                                         block_indices_begins,
                                                         block_update_indices,
                                                         block_update_indices_begins});

    ASSERT_TRUE(pass::PaKVReorderFusion(cache_precision).run_on_model(model));

    Config config;
    Transformations transformations(model, config);
    transformations.UpToLpt();

    std::shared_ptr<op::internal::PaKVReorder> pa_kv_reorder;
    for (const auto& node : model->get_ops()) {
        pa_kv_reorder = as_type_ptr<op::internal::PaKVReorder>(node);
        if (pa_kv_reorder) {
            break;
        }
    }

    ASSERT_NE(pa_kv_reorder, nullptr);
    EXPECT_EQ(pa_kv_reorder->get_input_element_type(0), cache_precision);
    EXPECT_EQ(pa_kv_reorder->get_input_element_type(1), cache_precision);
    EXPECT_EQ(pa_kv_reorder->input_value(0).get_node_shared_ptr(), key_cache);
    EXPECT_EQ(pa_kv_reorder->input_value(1).get_node_shared_ptr(), value_cache);
}

INSTANTIATE_TEST_SUITE_P(F16AndBF16, PaKVReorderPrecisionTest, testing::Values(element::f16, element::bf16));

}  // namespace ov::intel_cpu