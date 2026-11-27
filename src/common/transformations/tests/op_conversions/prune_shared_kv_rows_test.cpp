// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/prune_shared_kv_rows.hpp"

#include <limits>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bucketize.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
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
std::shared_ptr<Model> getModel(bool shared = true, bool with_sampled = true, element::Type index_type = element::i64) {
    Inputs i;
    i.sampled = parameter("sampled_tokens_indices", index_type, {-1});
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

TEST_F(PruneSharedKVRowsTest, GathersSampledRowsIntoSharedLayersReference) {
    model = getModel();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);

    Inputs i;
    i.params.push_back(i.sampled);
    const auto scalar = [](int64_t value) {
        return v0::Constant::create(element::i64, Shape{}, {value});
    };
    const auto vector = [](int64_t value) {
        return v0::Constant::create(element::i64, Shape{1}, {value});
    };
    const auto rows_of = [&](const Output<Node>& data, const Output<Node>& indices) {
        return std::make_shared<v8::Gather>(data, indices, scalar(0));
    };
    const auto max = std::numeric_limits<int64_t>::max();
    // begins = [0, 3, 5], sampled = [2, 4]: kept rows [2, 4], past_lens + [2, 1], begins [0, 1, 2], sampled [0, 1].
    const auto begins = std::make_shared<v0::Convert>(i.begins, element::i64);
    const auto starts = std::make_shared<v8::Slice>(begins, vector(0), vector(-1), vector(1));
    const auto ends = std::make_shared<v8::Slice>(begins, vector(1), vector(max), vector(1));
    const auto total_rows = std::make_shared<v8::Slice>(begins, vector(-1), vector(max), vector(1));
    const auto normalized = std::make_shared<v1::Select>(std::make_shared<v1::Less>(i.sampled, scalar(0)),
                                                         std::make_shared<v1::Add>(i.sampled, total_rows),
                                                         i.sampled);
    const auto in_range = std::make_shared<v1::LogicalAnd>(std::make_shared<v1::GreaterEqual>(normalized, scalar(0)),
                                                           std::make_shared<v1::Less>(normalized, total_rows));
    const auto sampled = std::make_shared<v1::Select>(in_range, normalized, total_rows);
    const auto sampled_sequence = std::make_shared<v3::Bucketize>(sampled, ends, element::i64, false);
    const auto sequence_count = std::make_shared<v0::Squeeze>(std::make_shared<v3::ShapeOf>(ends, element::i64));
    const auto sequence_ids = std::make_shared<v4::Range>(scalar(0), sequence_count, scalar(1), element::i64);
    const auto matches =
        std::make_shared<v1::Equal>(std::make_shared<v0::Unsqueeze>(sequence_ids, scalar(1)), sampled_sequence);
    const auto end_column = std::make_shared<v0::Unsqueeze>(ends, scalar(1));
    const auto first_sampled = std::make_shared<v1::ReduceMin>(
        std::make_shared<v0::Concat>(
            OutputVector{std::make_shared<v1::Select>(matches, sampled, end_column), end_column},
            1),
        scalar(1));
    const auto lengths = std::make_shared<v1::Subtract>(ends, starts);
    const auto counts = std::make_shared<v1::Minimum>(
        std::make_shared<v1::Maximum>(std::make_shared<v1::Subtract>(ends, first_sampled), scalar(1)),
        lengths);
    const auto kept_ends = std::make_shared<v0::CumSum>(counts, scalar(0));
    const auto kept_count = std::make_shared<v1::ReduceSum>(counts, scalar(0));
    const auto kept_ids = std::make_shared<v4::Range>(scalar(0), kept_count, scalar(1), element::i64);
    const auto kept_sequence = std::make_shared<v3::Bucketize>(kept_ids, kept_ends, element::i64, false);
    const auto kept_rows =
        std::make_shared<v1::Subtract>(rows_of(ends, kept_sequence),
                                       std::make_shared<v1::Subtract>(rows_of(kept_ends, kept_sequence), kept_ids));
    const auto sampled_lookup =
        std::make_shared<v1::Minimum>(sampled_sequence, std::make_shared<v1::Subtract>(sequence_count, scalar(1)));
    const auto sampled_rows = std::make_shared<v1::Select>(
        in_range,
        std::make_shared<v1::Subtract>(rows_of(kept_ends, sampled_lookup),
                                       std::make_shared<v1::Subtract>(rows_of(ends, sampled_lookup), sampled)),
        kept_count);
    const auto past = std::make_shared<v1::Add>(std::make_shared<v0::Convert>(i.past, element::i64),
                                                std::make_shared<v1::Subtract>(lengths, counts));
    const auto kept_past = std::make_shared<v0::Convert>(past, element::i32);
    const auto kept_begins =
        std::make_shared<v0::Convert>(std::make_shared<v0::Concat>(OutputVector{vector(0), kept_ends}, 0),
                                      element::i32);

    auto owner = attention(i.data, i.data, i.data, i.key, i.value, i.past, i.begins, true);
    const auto owner_rows = rows_of(owner->output(0), kept_rows);
    const auto data_rows = rows_of(i.data, kept_rows);
    auto query = std::make_shared<v1::Multiply>(owner_rows, v0::Constant::create(element::f32, Shape{1, 4}, {2}));
    auto layer = attention(query, data_rows, data_rows, i.key, i.value, kept_past, kept_begins, false);
    auto hidden = std::make_shared<v1::Add>(layer->output(0), owner_rows);
    auto head = rows_of(hidden, sampled_rows);
    model_ref = std::make_shared<Model>(OutputVector{head}, i.params);
}

TEST(PruneSharedKVRows, WidensNarrowSelectionIndices) {
    const auto model = getModel(true, true, element::i8);
    ASSERT_TRUE(pass::PruneSharedKVRows().run_on_model(model));
    // Remapped rows may exceed the range of the original index type, such as -1 over 300 rows.
    const auto head = model->get_results().front()->get_input_node_shared_ptr(0);
    EXPECT_EQ(head->get_input_element_type(1), element::i64);
    EXPECT_EQ(model->get_parameters().back()->get_element_type(), element::i8);
}

TEST_F(PruneSharedKVRowsTest, KeepsSharedAttentionWithXAttentionUnchanged) {
    model = getModel();
    shared_attention(model)->input(17).replace_source_output(v0::Constant::create(element::f32, Shape{1}, {0.9f}));
}

TEST_F(PruneSharedKVRowsTest, KeepsRowsBroadcastFromLowerRankUnchanged) {
    Inputs i;
    i.params.push_back(i.sampled);
    auto owner = attention(i.data, i.data, i.data, i.key, i.value, i.past, i.begins, true);
    auto layer = shared_layer(i, owner->output(0));
    // [T, 4, 1] + [T, 4] aligns the residual's rows with the middle axis.
    auto column = std::make_shared<v0::Unsqueeze>(layer, v0::Constant::create(element::i64, Shape{1}, {2}));
    auto hidden = std::make_shared<v1::Add>(column, owner);
    auto head = std::make_shared<v8::Gather>(hidden, i.sampled, v0::Constant::create(element::i64, Shape{}, {0}));
    model = std::make_shared<Model>(OutputVector{head}, i.params);
}

TEST_F(PruneSharedKVRowsTest, KeepsSelectionsOfUnprunedRowsUnchanged) {
    Inputs i;
    i.params.push_back(i.sampled);
    auto owner = attention(i.data, i.data, i.data, i.key, i.value, i.past, i.begins, true);
    auto first = shared_layer(i, owner->output(0));
    auto second = shared_layer(i, first->output(0));
    const auto axis = v0::Constant::create(element::i64, Shape{}, {0});
    auto first_head = std::make_shared<v8::Gather>(first->output(0), i.sampled, axis);
    auto second_head = std::make_shared<v8::Gather>(second->output(0), i.sampled, axis);
    model = std::make_shared<Model>(OutputVector{first_head, second_head, first->output(0)}, i.params);
}

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
