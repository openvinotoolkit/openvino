// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/prune_shared_kv_rows.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
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
    std::shared_ptr<v0::Parameter> data = parameter("inputs_embeds", element::f32, {-1, 4});
    std::shared_ptr<v0::Parameter> key = parameter("key_cache", element::f32, {-1, 1, 16, 4});
    std::shared_ptr<v0::Parameter> value = parameter("value_cache", element::f32, {-1, 1, 16, 4});
    std::shared_ptr<v0::Parameter> past = parameter("past_lens", element::i32, {-1});
    std::shared_ptr<v0::Parameter> begins = parameter("subsequence_begins", element::i32, {-1});
    std::shared_ptr<v0::Parameter> sampled = parameter("sampled_tokens_indices", element::i64, {-1});
    ParameterVector params{data, key, value, past, begins};
};

std::shared_ptr<PagedAttentionExtension> shared_layer(Inputs& i, const Output<Node>& hidden, bool shared = true) {
    auto query = std::make_shared<v1::Multiply>(hidden, v0::Constant::create(element::f32, Shape{1, 4}, {2}));
    return attention(query, i.data, i.data, i.key, i.value, i.past, i.begins, !shared);
}

// Writes the cache, then a shared layer with a residual, then selects the sampled rows.
std::shared_ptr<Model> getModel(bool shared = true, bool with_sampled = true) {
    Inputs i;
    auto owner = attention(i.data, i.data, i.data, i.key, i.value, i.past, i.begins, true);
    auto layer = shared_layer(i, owner->output(0), shared);
    Output<Node> hidden = std::make_shared<v1::Add>(layer->output(0), owner->output(0));
    if (with_sampled) {
        i.params.push_back(i.sampled);
        hidden = std::make_shared<v8::Gather>(hidden, i.sampled, v0::Constant::create(element::i64, Shape{}, {0}));
    }
    return std::make_shared<Model>(OutputVector{hidden}, i.params);
}

std::shared_ptr<PagedAttentionExtension> shared_attention(const std::shared_ptr<Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        const auto pa = as_type_ptr<PagedAttentionExtension>(node);
        if (pa && !pa->get_write_kv_cache())
            return pa;
    }
    return nullptr;
}

TEST(PruneSharedKVRows, GathersSampledRowsIntoSharedLayers) {
    const auto model = getModel();
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
    // The residual from the owner layer is gathered too.
    const auto residual = model->get_results().front()->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0);
    EXPECT_TRUE(is_type<v8::Gather>(residual->get_input_node_ptr(1)));
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

TEST_F(PruneSharedKVRowsTest, KeepsModelsWithoutRowSelectionUnchanged) {
    model = getModel(true, false);
}

TEST_F(PruneSharedKVRowsTest, KeepsModelsWithoutSharedLayersUnchanged) {
    model = getModel(false);
}

TEST_F(PruneSharedKVRowsTest, KeepsSharedAttentionWithTokenTypesUnchanged) {
    model = getModel();
    shared_attention(model)->input(25).replace_source_output(v0::Constant::create(element::i32, Shape{1}, {1}));
}

TEST_F(PruneSharedKVRowsTest, KeepsSharedAttentionWithQueryBiasUnchanged) {
    model = getModel();
    shared_attention(model)->input(26).replace_source_output(v0::Constant::create(element::u8, Shape{1}, {1}));
}

TEST_F(PruneSharedKVRowsTest, KeepsSharedRowsOutsideSelectionUnchanged) {
    model = getModel();
    model->add_results({std::make_shared<v0::Result>(shared_attention(model)->output(0))});
}

TEST_F(PruneSharedKVRowsTest, KeepsUsedSharedScoresUnchanged) {
    model = getModel();
    model->add_results({std::make_shared<v0::Result>(shared_attention(model)->output(1))});
}

TEST_F(PruneSharedKVRowsTest, KeepsSharedLayersWithDifferentLengthsUnchanged) {
    Inputs i;
    auto other_past = parameter("other_past_lens", element::i32, {-1});
    i.params.insert(i.params.end(), {other_past, i.sampled});
    auto owner = attention(i.data, i.data, i.data, i.key, i.value, i.past, i.begins, true);
    auto first = shared_layer(i, owner->output(0));
    auto query = std::make_shared<v1::Multiply>(first->output(0), v0::Constant::create(element::f32, Shape{1, 4}, {2}));
    auto second = attention(query, i.data, i.data, i.key, i.value, other_past, i.begins, false);
    auto head = std::make_shared<v8::Gather>(second, i.sampled, v0::Constant::create(element::i64, Shape{}, {0}));
    model = std::make_shared<Model>(OutputVector{head}, i.params);
}

TEST_F(PruneSharedKVRowsTest, KeepsRowMixingTailUnchanged) {
    Inputs i;
    i.params.push_back(i.sampled);
    auto owner = attention(i.data, i.data, i.data, i.key, i.value, i.past, i.begins, true);
    auto layer = shared_layer(i, owner->output(0));
    auto total = std::make_shared<v1::ReduceSum>(layer, v0::Constant::create(element::i64, Shape{1}, {0}), true);
    auto hidden = std::make_shared<v1::Add>(owner, total);
    auto head = std::make_shared<v8::Gather>(hidden, i.sampled, v0::Constant::create(element::i64, Shape{}, {0}));
    model = std::make_shared<Model>(OutputVector{head}, i.params);
}

// The rows also end up on the second axis.
std::shared_ptr<Model> row_broadcast_model() {
    Inputs i;
    i.params.push_back(i.sampled);
    auto owner = attention(i.data, i.data, i.data, i.key, i.value, i.past, i.begins, true);
    auto layer = shared_layer(i, owner->output(0));
    auto column = std::make_shared<v1::ReduceSum>(layer, v0::Constant::create(element::i64, Shape{1}, {1}), true);
    auto row = std::make_shared<v1::ReduceSum>(layer, v0::Constant::create(element::i64, Shape{1}, {1}), false);
    auto hidden = std::make_shared<v1::Add>(column, row);
    auto head = std::make_shared<v8::Gather>(hidden, i.sampled, v0::Constant::create(element::i64, Shape{}, {0}));
    return std::make_shared<Model>(OutputVector{head}, i.params);
}

TEST_F(PruneSharedKVRowsTest, KeepsRowBroadcastUnchanged) {
    model = row_broadcast_model();
}

TEST_F(PruneSharedKVRowsTest, KeepsRowCountValuesUnchanged) {
    Inputs i;
    i.params.push_back(i.sampled);
    auto owner = attention(i.data, i.data, i.data, i.key, i.value, i.past, i.begins, true);
    auto layer = shared_layer(i, owner->output(0));
    auto count =
        std::make_shared<v0::Convert>(std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(layer),
                                                                   v0::Constant::create(element::i64, Shape{1}, {0}),
                                                                   v0::Constant::create(element::i64, Shape{}, {0})),
                                      element::f32);
    auto hidden = std::make_shared<v1::Divide>(layer, count);
    auto head = std::make_shared<v8::Gather>(hidden, i.sampled, v0::Constant::create(element::i64, Shape{}, {0}));
    model = std::make_shared<Model>(OutputVector{head}, i.params);
}

TEST(PruneSharedKVRows, KeepsParameterSymbols) {
    const auto model = row_broadcast_model();
    const auto& data = model->get_parameters().front();
    auto shape = data->get_partial_shape();
    const auto symbol = std::make_shared<Symbol>();
    shape[0].set_symbol(symbol);
    data->set_partial_shape(shape);
    model->validate_nodes_and_infer_types();
    EXPECT_FALSE(pass::PruneSharedKVRows().run_on_model(model));
    EXPECT_EQ(data->get_output_partial_shape(0)[0].get_symbol(), symbol);
    EXPECT_EQ(data->get_output_partial_shape(0)[1].get_symbol(), nullptr);
}

TEST_F(PruneSharedKVRowsTest, KeepsUnknownRowsUnchanged) {
    model = getModel();
    auto other = parameter("other", element::f32, {-1, 4});
    const auto pa = shared_attention(model);
    auto query = pa->input_value(0);
    pa->input(0).replace_source_output(std::make_shared<v1::Add>(query, other));
    model->add_parameters({other});
}

}  // namespace
