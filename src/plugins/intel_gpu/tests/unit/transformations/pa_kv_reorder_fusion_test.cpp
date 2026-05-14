// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"

#include "intel_gpu/op/pa_kv_reorder.hpp"
#include "plugin/transformations/pa_kv_reorder_fusion.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, PaKVReorderFusion_basic) {
    disable_result_friendly_names_check();
    {
        auto key_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{4, 2});
        key_cache->set_friendly_name("key_cache.0_clone_for_k_update");
        auto value_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{4, 2});
        value_cache->set_friendly_name("value_cache.0_clone_for_v_update");

        auto block_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        block_indices->set_friendly_name("block_indices");
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

        auto key_scatter = std::make_shared<ov::op::v3::ScatterUpdate>(key_cache, block_indices, key_gather, scatter_axis);
        auto value_scatter = std::make_shared<ov::op::v3::ScatterUpdate>(value_cache, block_indices, value_gather, scatter_axis);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{key_scatter, value_scatter}, 0);
        auto result = std::make_shared<ov::op::v0::Result>(concat);

        model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                            ov::ParameterVector{key_cache,
                                                                value_cache,
                                                                block_indices,
                                                                block_indices_begins,
                                                                block_update_indices,
                                                                block_update_indices_begins});

        manager.register_pass<PaKVReorderFusion>(false,
                                                 std::vector<size_t>{0, 1, 3, 2},
                                                 std::vector<size_t>{0, 1, 2, 3},
                                                 ov::element::f16,
                                                 ov::element::f16,
                                                 ov::element::f16);
    }

    {
        auto key_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{4, 2});
        key_cache->set_friendly_name("key_cache.0_clone_for_k_update");
        auto value_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{4, 2});
        value_cache->set_friendly_name("value_cache.0_clone_for_v_update");

        auto block_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        block_indices->set_friendly_name("block_indices");
        auto block_indices_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        block_indices_begins->set_friendly_name("block_indices_begins");

        auto block_update_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        block_update_indices->set_friendly_name("block_update_indices");
        auto block_update_indices_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        block_update_indices_begins->set_friendly_name("block_update_indices_begins");

        auto pa_kv_reorder = std::make_shared<ov::intel_gpu::op::PA_KV_Reorder>(key_cache,
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
    auto key_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{4, 2});
    key_cache->set_friendly_name("key_cache.0_clone_for_k_update");
    auto value_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{4, 2});
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

    auto key_scatter = std::make_shared<ov::op::v3::ScatterUpdate>(key_cache, block_indices_k, key_gather, scatter_axis);
    auto value_scatter = std::make_shared<ov::op::v3::ScatterUpdate>(value_cache, block_indices_v, value_gather, scatter_axis);

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

    manager.register_pass<PaKVReorderFusion>(false,
                                             std::vector<size_t>{0, 1, 3, 2},
                                             std::vector<size_t>{0, 1, 2, 3},
                                             ov::element::f16,
                                             ov::element::f16,
                                             ov::element::f16);

    model_ref = model->clone();
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
