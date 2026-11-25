// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/prune_shared_kv_rows.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov;
using namespace ov::op;

namespace {

std::shared_ptr<v0::Parameter> parameter(const char* name, element::Type type, const PartialShape& shape) {
    auto node = std::make_shared<v0::Parameter>(type, shape);
    node->set_friendly_name(name);
    node->output(0).set_names({name});
    return node;
}

std::shared_ptr<PagedAttentionExtension> attention(const Output<Node>& q,
                                                   const Output<Node>& k,
                                                   const Output<Node>& v,
                                                   const Output<Node>& key_cache,
                                                   const Output<Node>& value_cache,
                                                   const Output<Node>& past_lens,
                                                   const Output<Node>& subsequence_begins,
                                                   bool write_cache) {
    auto empty_i32 = v0::Constant::create(element::i32, Shape{0}, {});
    auto empty_f32 = v0::Constant::create(element::f32, Shape{0}, {});
    auto scalar_i32 = v0::Constant::create(element::i32, Shape{}, {0});
    OutputVector args{q,
                      k,
                      v,
                      key_cache,
                      value_cache,
                      past_lens,
                      subsequence_begins,
                      empty_i32,
                      empty_i32,
                      v0::Constant::create(element::f32, Shape{}, {1}),
                      scalar_i32,
                      empty_f32,
                      scalar_i32,
                      empty_i32,
                      empty_i32,
                      empty_i32,
                      empty_f32,
                      empty_f32,
                      scalar_i32,
                      scalar_i32,
                      empty_f32,
                      scalar_i32,
                      empty_i32,
                      empty_i32,
                      empty_i32,
                      empty_i32,
                      v0::Constant::create(element::u8, Shape{0}, {}),
                      empty_i32};
    return std::make_shared<PagedAttentionExtension>(args, write_cache);
}

struct Inputs {
    std::shared_ptr<v0::Parameter> data = parameter("data", element::f32, {-1, 4});
    std::shared_ptr<v0::Parameter> rows = parameter("rows", element::i64, {-1});
    std::shared_ptr<v0::Parameter> key = parameter("key_cache", element::f32, {-1, 1, 16, 4});
    std::shared_ptr<v0::Parameter> value = parameter("value_cache", element::f32, {-1, 1, 16, 4});
    std::shared_ptr<v0::Parameter> past = parameter("past_lens", element::i32, {-1});
    std::shared_ptr<v0::Parameter> begins = parameter("subsequence_begins", element::i32, {-1});
    std::shared_ptr<v0::Parameter> sampled = parameter("sampled_tokens_indices", element::i64, {-1});
    ParameterVector params{data, rows, key, value, past, begins};
};

Output<Node> selected_query(const Output<Node>& data, const Output<Node>& indices) {
    auto shape = v0::Constant::create(element::i64, Shape{2}, {-1, 1});
    auto rows = std::make_shared<v1::Reshape>(indices, shape, false);
    auto axis = v0::Constant::create(element::i64, Shape{1}, {0});
    auto gather = std::make_shared<v8::Gather>(data, rows, axis);
    auto unsqueeze = std::make_shared<v0::Unsqueeze>(gather, axis);
    return std::make_shared<v1::Reshape>(unsqueeze, v0::Constant::create(element::i64, Shape{2}, {-1, 4}), false);
}

std::shared_ptr<Model> model_with_head(const Output<Node>& hidden, Inputs& i, bool with_sampled = true) {
    if (!with_sampled)
        return std::make_shared<Model>(OutputVector{hidden}, i.params);
    i.params.push_back(i.sampled);
    auto head = std::make_shared<v8::Gather>(hidden, i.sampled, v0::Constant::create(element::i64, Shape{}, {0}));
    return std::make_shared<Model>(OutputVector{head}, i.params);
}

std::shared_ptr<Model> getModel(bool shared,
                                bool mark,
                                bool with_token_types = false,
                                bool ambiguous = false,
                                bool with_sampled = true) {
    Inputs i;
    auto owner = attention(i.data, i.data, i.data, i.key, i.value, i.past, i.begins, true);
    auto query = selected_query(owner->output(0), i.rows);
    if (ambiguous) {
        auto other_rows = parameter("other_rows", element::i64, {-1});
        i.params.push_back(other_rows);
        query = std::make_shared<v1::Add>(query, selected_query(owner->output(0), other_rows));
    }
    auto second = attention(query, i.data, i.data, i.key, i.value, i.past, i.begins, !shared);
    if (with_token_types)
        i.params.push_back(parameter("token_type_ids", element::i32, {-1}));
    auto model = model_with_head(second->output(0), i, with_sampled);
    if (mark)
        model->get_rt_info()[pass::shared_kv_row_selection_key] = true;
    return model;
}

std::shared_ptr<PagedAttentionExtension> shared_attention(const std::shared_ptr<Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        const auto pa = as_type_ptr<PagedAttentionExtension>(node);
        if (pa && !pa->get_write_kv_cache())
            return pa;
    }
    return nullptr;
}

TEST(PruneSharedKVRows, DerivesKeptRowsFromSampledTokens) {
    const auto model = getModel(true, true);
    const auto parameters = model->get_parameters();
    const auto find = [&](const std::string& name) {
        for (const auto& parameter : parameters) {
            if (parameter->get_friendly_name() == name)
                return parameter;
        }
        return std::shared_ptr<v0::Parameter>{};
    };
    ASSERT_TRUE(pass::PruneSharedKVRows().run_on_model(model));

    EXPECT_EQ(model->get_parameters(), parameters);
    const auto pa = shared_attention(model);
    ASSERT_NE(pa, nullptr);
    for (const auto index : {1, 2})
        EXPECT_TRUE(is_type<v8::Gather>(pa->get_input_node_ptr(index))) << index;
    EXPECT_NE(pa->get_input_node_shared_ptr(5), find("past_lens"));
    EXPECT_NE(pa->get_input_node_shared_ptr(6), find("subsequence_begins"));
    EXPECT_TRUE(find("rows")->get_output_target_inputs(0).empty());
    const auto head = model->get_results().front()->get_input_node_shared_ptr(0);
    EXPECT_NE(head->get_input_node_shared_ptr(1), find("sampled_tokens_indices"));
    EXPECT_FALSE(pass::PruneSharedKVRows().run_on_model(model));
}

class PruneSharedKVRowsTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<pass::PruneSharedKVRows>();
    }
};

TEST_F(PruneSharedKVRowsTest, KeepsUnmarkedModelsUnchanged) {
    model = getModel(true, false);
}

TEST_F(PruneSharedKVRowsTest, KeepsModelsWithoutSampledTokensUnchanged) {
    model = getModel(true, true, false, false, false);
}

TEST_F(PruneSharedKVRowsTest, KeepsIndependentCachesUnchanged) {
    model = getModel(false, true);
}

TEST_F(PruneSharedKVRowsTest, KeepsTokenTypeModelsUnchanged) {
    model = getModel(true, true, true);
}

TEST_F(PruneSharedKVRowsTest, KeepsQueryBiasModelsUnchanged) {
    model = getModel(true, true);
    model->add_parameters({parameter("qq_bias", element::u8, {-1})});
}

TEST_F(PruneSharedKVRowsTest, KeepsSharedAttentionWithQueryBiasUnchanged) {
    model = getModel(true, true);
    shared_attention(model)->input(26).replace_source_output(v0::Constant::create(element::u8, Shape{1}, {1}));
}

TEST_F(PruneSharedKVRowsTest, KeepsSharedRowsOutsideSampledGatherUnchanged) {
    model = getModel(true, true);
    model->add_results({std::make_shared<v0::Result>(shared_attention(model)->output(0))});
}

TEST_F(PruneSharedKVRowsTest, KeepsUsedSharedScoresUnchanged) {
    model = getModel(true, true);
    model->add_results({std::make_shared<v0::Result>(shared_attention(model)->output(1))});
}

TEST_F(PruneSharedKVRowsTest, KeepsSharedLayersWithDifferentLengthsUnchanged) {
    Inputs i;
    auto other_past = parameter("other_past_lens", element::i32, {-1});
    i.params.push_back(other_past);
    auto owner = attention(i.data, i.data, i.data, i.key, i.value, i.past, i.begins, true);
    auto first =
        attention(selected_query(owner->output(0), i.rows), i.data, i.data, i.key, i.value, i.past, i.begins, false);
    auto second = attention(first->output(0), i.data, i.data, i.key, i.value, other_past, i.begins, false);
    model = model_with_head(second->output(0), i);
    model->get_rt_info()[pass::shared_kv_row_selection_key] = true;
}

TEST_F(PruneSharedKVRowsTest, KeepsAmbiguousSelectionsUnchanged) {
    model = getModel(true, true, false, true);
}

TEST_F(PruneSharedKVRowsTest, KeepsSelectionsBeforeAttentionUnchanged) {
    Inputs i;
    auto query = selected_query(i.data, i.rows);
    auto owner = attention(query, i.data, i.data, i.key, i.value, i.past, i.begins, true);
    auto shared = attention(owner->output(0), i.data, i.data, i.key, i.value, i.past, i.begins, false);
    model = model_with_head(shared->output(0), i);
    model->get_rt_info()[pass::shared_kv_row_selection_key] = true;
}

}  // namespace
