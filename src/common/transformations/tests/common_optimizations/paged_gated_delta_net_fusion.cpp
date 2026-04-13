// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_gated_delta_net_fusion.hpp"

#include <gtest/gtest.h>

#include <cstdlib>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/runtime/core.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace internal = ov::op::internal;

std::shared_ptr<v0::Parameter> make_f32_param(const std::string& name, const Shape& shape) {
    auto p = std::make_shared<v0::Parameter>(element::f32, shape);
    p->set_friendly_name(name);
    p->get_output_tensor(0).set_names({name});
    return p;
}

std::shared_ptr<v0::Parameter> make_i32_param(const std::string& name, const Shape& shape) {
    auto p = std::make_shared<v0::Parameter>(element::i32, shape);
    p->set_friendly_name(name);
    p->get_output_tensor(0).set_names({name});
    return p;
}

std::shared_ptr<ov::Model> build_fusable_model() {
    auto query = make_f32_param("query", Shape{2, 3, 4, 8});
    auto key = make_f32_param("key", Shape{2, 3, 4, 8});
    auto value = make_f32_param("value", Shape{2, 3, 4, 6});

    auto recurrent_state = make_f32_param("past_recurrent_state", Shape{2, 4, 8, 6});
    recurrent_state->get_output_tensor(0).set_names({"cache_params.past.recurrent_state.0"});

    auto gate = make_f32_param("gate", Shape{2, 3, 4});
    auto beta = make_f32_param("beta", Shape{2, 3, 4});

    auto gdn = std::make_shared<internal::GatedDeltaNet>(query, key, value, recurrent_state, gate, beta);

    auto out = std::make_shared<v0::Result>(gdn->output(0));
    auto present_state = std::make_shared<v0::Result>(gdn->output(1));
    present_state->get_output_tensor(0).set_names({"cache_params.present.recurrent_state.0"});

    auto subsequence_begins = make_i32_param("subsequence_begins", Shape{3});
    auto block_indices = make_i32_param("paged_gdn.block_indices", Shape{5});
    auto block_indices_begins = make_i32_param("paged_gdn.block_indices_begins", Shape{3});
    auto past_lens = make_i32_param("paged_gdn.past_lens", Shape{2});
    auto cache_interval = make_i32_param("paged_gdn.cache_interval", Shape{2});

    ParameterVector params{
        query,
        key,
        value,
        recurrent_state,
        gate,
        beta,
        subsequence_begins,
        block_indices,
        block_indices_begins,
        past_lens,
        cache_interval,
    };

    return std::make_shared<ov::Model>(ResultVector{out, present_state}, params);
}

std::shared_ptr<ov::Model> build_non_fusable_model() {
    auto query = make_f32_param("query", Shape{2, 3, 4, 8});
    auto key = make_f32_param("key", Shape{2, 3, 4, 8});
    auto value = make_f32_param("value", Shape{2, 3, 4, 6});

    auto recurrent_state = make_f32_param("past_recurrent_state", Shape{2, 4, 8, 6});
    recurrent_state->get_output_tensor(0).set_names({"cache_params.past.recurrent_state.0"});

    auto gate = make_f32_param("gate", Shape{2, 3, 4});
    auto beta = make_f32_param("beta", Shape{2, 3, 4});

    auto gdn = std::make_shared<internal::GatedDeltaNet>(query, key, value, recurrent_state, gate, beta);
    auto add_rhs = make_f32_param("state_add_rhs", Shape{2, 4, 8, 6});
    auto state_add = std::make_shared<v1::Add>(gdn->output(1), add_rhs);

    auto out = std::make_shared<v0::Result>(gdn->output(0));
    auto present_state = std::make_shared<v0::Result>(state_add);
    present_state->get_output_tensor(0).set_names({"cache_params.present.recurrent_state.0"});

    ParameterVector params{query, key, value, recurrent_state, gate, beta, add_rhs};
    return std::make_shared<ov::Model>(ResultVector{out, present_state}, params);
}

}  // namespace

class PagedGatedDeltaNetFusionTest : public ::TransformationTestsF {};

TEST_F(PagedGatedDeltaNetFusionTest, FusesWhenStateOutputIsOnlyResultConsumer) {
    model = build_fusable_model();

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PagedGatedDeltaNetFusion>();
    manager.run_passes(model);

    size_t paged_gdn_count = 0;
    size_t gdn_count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (std::string(op->get_type_name()) == "PagedGatedDeltaNet") {
            ++paged_gdn_count;
        }
        if (std::string(op->get_type_name()) == "GatedDeltaNet") {
            ++gdn_count;
        }
    }

    EXPECT_EQ(paged_gdn_count, 1u);
    EXPECT_EQ(gdn_count, 0u);
}

TEST_F(PagedGatedDeltaNetFusionTest, DoesNotFuseWhenStateOutputHasNonResultConsumer) {
    model = build_non_fusable_model();
    model_ref = build_non_fusable_model();

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PagedGatedDeltaNetFusion>();
    manager.run_passes(model);

    compare_functions(model, model_ref);
}

TEST(PagedGatedDeltaNetRealModel, RealModelAfterPATransformation) {
    const char* model_path = std::getenv("OV_GDN_REAL_MODEL_PATH");
    if (!model_path || std::string(model_path).empty()) {
        GTEST_SKIP() << "OV_GDN_REAL_MODEL_PATH is not set";
    }

    ov::Core core;
    auto real_model = core.read_model(model_path);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::SDPAToPagedAttention>(false, false, false, false, false, false);
    manager.run_passes(real_model);

    size_t paged_gdn_count = 0;
    size_t gdn_count = 0;
    for (const auto& op : real_model->get_ordered_ops()) {
        if (std::string(op->get_type_name()) == "PagedGatedDeltaNet") {
            ++paged_gdn_count;
        }
        if (std::string(op->get_type_name()) == "GatedDeltaNet") {
            ++gdn_count;
        }
    }

    EXPECT_GE(paged_gdn_count, 1u);
    EXPECT_EQ(gdn_count, 0u);
}
