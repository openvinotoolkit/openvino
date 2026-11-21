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

std::shared_ptr<Model> getModel(bool shared, bool mark, bool with_token_types = false, bool ambiguous = false) {
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
    auto model = std::make_shared<Model>(OutputVector{second->output(0)}, i.params);
    if (mark)
        model->get_rt_info()[pass::shared_kv_row_selection_key] = true;
    return model;
}

std::shared_ptr<Model> getModelRef() {
    Inputs i;
    auto tokens = parameter("shared_kv_tokens_indices", element::i64, {-1});
    auto past = parameter("shared_kv_past_lens", element::i32, {-1});
    auto begins = parameter("shared_kv_subsequence_begins", element::i32, {-1});
    i.params.insert(i.params.end(), {tokens, past, begins});
    auto owner = attention(i.data, i.data, i.data, i.key, i.value, i.past, i.begins, true);
    auto ids = std::make_shared<v1::Reshape>(tokens, v0::Constant::create(element::i64, Shape{3}, {1, -1, 1}), false);
    auto query = selected_query(owner->output(0), ids);
    auto axis = v0::Constant::create(element::i64, Shape{}, {0});
    auto key = std::make_shared<v8::Gather>(i.data, tokens, axis);
    auto value = std::make_shared<v8::Gather>(i.data, tokens, axis);
    auto second = attention(query, key, value, i.key, i.value, past, begins, false);
    return std::make_shared<Model>(OutputVector{second->output(0)}, i.params);
}

class PruneSharedKVRowsTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<pass::PruneSharedKVRows>();
    }
};

TEST_F(PruneSharedKVRowsTest, CompactsOnlySharedAttentionInputs) {
    model = getModel(true, true);
    model_ref = getModelRef();
    comparator.enable(FunctionsComparator::CONST_VALUES);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

TEST_F(PruneSharedKVRowsTest, KeepsUnmarkedModelsUnchanged) {
    model = getModel(true, false);
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
    for (const auto& node : model->get_ops()) {
        const auto pa = as_type_ptr<PagedAttentionExtension>(node);
        if (pa && !pa->get_write_kv_cache())
            pa->input(26).replace_source_output(v0::Constant::create(element::u8, Shape{1}, {1}));
    }
}

TEST_F(PruneSharedKVRowsTest, KeepsAmbiguousSelectionsUnchanged) {
    model = getModel(true, true, false, true);
}

TEST_F(PruneSharedKVRowsTest, KeepsSelectionsBeforeAttentionUnchanged) {
    Inputs i;
    auto query = selected_query(i.data, i.rows);
    auto owner = attention(query, i.data, i.data, i.key, i.value, i.past, i.begins, true);
    auto shared = attention(owner->output(0), i.data, i.data, i.key, i.value, i.past, i.begins, false);
    model = std::make_shared<Model>(OutputVector{shared->output(0)}, i.params);
    model->get_rt_info()[pass::shared_kv_row_selection_key] = true;
}

TEST(PruneSharedKVRowsIdempotency, RepeatedRunLeavesModelUnchanged) {
    const auto model = getModel(true, true);
    pass::PruneSharedKVRows pass;
    EXPECT_TRUE(pass.run_on_model(model));
    EXPECT_FALSE(pass.run_on_model(model));
}

}  // namespace
