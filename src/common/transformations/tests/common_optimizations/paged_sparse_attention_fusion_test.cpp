// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_sparse_attention_fusion.hpp"

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/paged_qsa_indexer.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v6 = ov::op::v6;
namespace v8 = ov::op::v8;
namespace v17 = ov::op::v17;
namespace op_internal = ov::op::internal;
using IndexerOutputIdx = op_internal::PagedQSAIndexer::OutputIdx;

constexpr int64_t query_heads = 4;
constexpr int64_t kv_heads = 2;
constexpr int64_t head_dim = 16;
constexpr int64_t value_head_dim = 12;
constexpr int64_t rotary_dim = 8;
constexpr int64_t key_block_size = 4;
constexpr int64_t topk = 2;

struct GraphOptions {
    std::string name;
    bool gathered = true;
    bool initialized = true;
    bool dynamic_shapes = false;
    size_t input_count = 6;
    bool scalar_scale = false;
    bool reinterpreted_selections = false;
    bool existing_caches = false;
    bool causal = true;
    bool different_indexers = false;
    bool different_current_length = false;
    int64_t query_block_size = 1;
    int64_t attention_key_block_size = key_block_size;
    element::Type precision = element::f32;
};

Output<Node> i64(const std::vector<int64_t>& values) {
    return v0::Constant::create(element::i64, Shape{values.size()}, values);
}

std::shared_ptr<Model> make_model(const GraphOptions& o, bool paged) {
    ParameterVector params;
    const auto param = [&](const std::string& name, element::Type type, const PartialShape& shape) {
        auto p = std::make_shared<v0::Parameter>(type, shape);
        p->set_friendly_name(name);
        p->get_output_tensor(0).set_names({name});
        params.push_back(p);
        return p;
    };
    const Dimension batch = o.dynamic_shapes ? Dimension::dynamic() : Dimension(2);
    const Dimension length = o.dynamic_shapes ? Dimension::dynamic() : Dimension(3);
    const Dimension current_length = o.different_current_length ? Dimension(4) : length;
    const auto query = param("query", o.precision, {batch, query_heads, length, head_dim});
    const auto key = param("key", o.precision, {batch, kv_heads, current_length, head_dim});
    const auto value = param("value", o.precision, {batch, kv_heads, current_length, value_head_dim});
    const auto beam_idx = param("beam_idx", element::i32, {batch});
    const auto key_length = param("key_length", element::i64, {batch});
    const auto past_lens = param("past_lens", element::i32, {batch});
    const auto subsequence_begins = param("subsequence_begins", element::i32, {batch + 1});
    const auto block_indices = param("block_indices", element::i32, {-1});
    const auto block_indices_begins = param("block_indices_begins", element::i32, {batch + 1});
    const Dimension tokens = o.dynamic_shapes ? Dimension::dynamic() : Dimension(6);
    const int64_t indexer_heads = o.reinterpreted_selections ? 2 : 1;
    const auto index_query = param("index_query", o.precision, {tokens, query_heads * head_dim});
    const auto index_keys = param("index_keys", o.precision, {tokens, indexer_heads * head_dim});
    const auto cos = param("rope_cos", o.precision, {tokens, rotary_dim});
    const auto sin = param("rope_sin", o.precision, {tokens, rotary_dim});
    const auto index_key_cache = param("index_key_cache.0", o.precision, {-1, indexer_heads, -1, head_dim});
    const auto rope_cache = param("index_block_rope_cache.0", o.precision, {-1, -1, 2 * rotary_dim});
    const auto index_block_cache = param("block_index_keys_cache.0", o.precision, {-1, indexer_heads, -1, head_dim});
    const auto weight = v0::Constant::create(o.precision, Shape{head_dim}, {1.0f});
    const auto indexer = [&] {
        op_internal::PagedQSAIndexer::Config config;
        config.key_block_size = key_block_size;
        config.block_topk = topk;
        config.indexer_head_dim = head_dim;
        config.indexer_rotary_dim = rotary_dim;
        config.num_selection_heads = indexer_heads;
        return std::make_shared<op_internal::PagedQSAIndexer>(OutputVector{index_query,
                                                                           index_keys,
                                                                           cos,
                                                                           sin,
                                                                           weight,
                                                                           weight,
                                                                           past_lens,
                                                                           subsequence_begins,
                                                                           block_indices,
                                                                           block_indices_begins,
                                                                           index_key_cache,
                                                                           rope_cache,
                                                                           index_block_cache},
                                                              config);
    };
    const auto selection = indexer();
    const auto counts = o.different_indexers ? indexer() : selection;
    SinkVector sinks;
    const auto full_kv = [&](const Output<Node>& current, const std::string& id, int64_t depth) {
        const auto variable =
            std::make_shared<op::util::Variable>(op::util::VariableInfo{{batch, kv_heads, -1, depth}, o.precision, id});
        Output<Node> read;
        if (o.initialized) {
            read = std::make_shared<v6::ReadValue>(
                v0::Constant::create(o.precision, Shape{2, kv_heads, 0, static_cast<size_t>(depth)}, {}),
                variable);
        } else {
            read = std::make_shared<v6::ReadValue>(variable);
        }
        if (o.gathered) {
            read = std::make_shared<v8::Gather>(read, beam_idx, i64({0}));
        }
        const auto concat = std::make_shared<v0::Concat>(OutputVector{read, current}, 2);
        sinks.push_back(std::make_shared<v6::Assign>(concat, variable));
        return concat;
    };
    const auto full_key = full_kv(key, "past_key", head_dim);
    const auto full_value = full_kv(value, "past_value", value_head_dim);
    const auto batch_and_length = op::util::node_to_get_shape_value_of_indices_from_shape_source(query, {0, 2});
    const auto restore = [&](const Output<Node>& data, const std::vector<int64_t>& tail) {
        const auto shape = std::make_shared<v0::Concat>(OutputVector{batch_and_length, i64(tail)}, 0);
        return std::make_shared<v1::Reshape>(data, shape, false);
    };
    const auto indices =
        std::make_shared<v1::Transpose>(restore(selection->output(IndexerOutputIdx::SELECTED_BLOCK_INDICES), {1, topk}),
                                        i64({0, 2, 1, 3}));
    const auto count =
        std::make_shared<v1::Transpose>(restore(counts->output(IndexerOutputIdx::SELECTED_BLOCK_COUNT), {1}),
                                        i64({0, 2, 1}));
    const auto scale = v0::Constant::create(o.precision, o.scalar_scale ? Shape{} : Shape{1}, {0.5f});
    const auto attention_sinks = v0::Constant::create(element::f32, Shape{query_heads}, {0.1f});
    if (o.existing_caches) {
        param("key_cache.0", element::f32, PartialShape::dynamic(4));
        param("value_cache.0", element::f32, PartialShape::dynamic(4));
    }

    Output<Node> output;
    if (paged) {
        const auto suffix = o.existing_caches ? ".1" : ".0";
        const auto key_cache = param(std::string("key_cache") + suffix, o.precision, PartialShape::dynamic(4));
        const auto value_cache = param(std::string("value_cache") + suffix, o.precision, PartialShape::dynamic(4));
        enable_keep_const_precision(key_cache);
        enable_keep_const_precision(value_cache);
        const auto flatten = [&](const Output<Node>& data, int64_t width) {
            return std::make_shared<v1::Reshape>(std::make_shared<v1::Transpose>(data, i64({0, 2, 1, 3})),
                                                 i64({-1, width}),
                                                 false);
        };
        Output<Node> paged_scale = v0::Constant::create(o.precision, Shape{}, {0.25f});
        Output<Node> paged_sinks = v0::Constant::create(o.precision, Shape{0}, {});
        if (o.input_count > 6) {
            paged_scale = scale;
            if (!o.scalar_scale) {
                paged_scale = std::make_shared<v0::Squeeze>(paged_scale, i64({0}));
            }
        }
        if (o.input_count > 7) {
            paged_sinks = attention_sinks;
        }
        const auto attention = std::make_shared<op_internal::PagedSparseAttention>(
            OutputVector{flatten(query, query_heads * head_dim),
                         flatten(key, kv_heads * head_dim),
                         flatten(value, kv_heads * value_head_dim),
                         key_cache,
                         value_cache,
                         past_lens,
                         subsequence_begins,
                         block_indices,
                         block_indices_begins,
                         selection->output(IndexerOutputIdx::SELECTED_BLOCK_INDICES),
                         selection->output(IndexerOutputIdx::SELECTED_BLOCK_COUNT),
                         paged_scale,
                         paged_sinks},
            key_block_size);
        attention->get_rt_info()["num_k_heads"] = kv_heads;
        attention->get_rt_info()["k_head_size"] = head_dim;
        attention->get_rt_info()["num_v_heads"] = kv_heads;
        attention->get_rt_info()["v_head_size"] = value_head_dim;
        output = std::make_shared<v1::Transpose>(restore(attention, {query_heads, value_head_dim}), i64({0, 2, 1, 3}));
    } else {
        OutputVector args{query, full_key, full_value, indices, count};
        if (o.input_count > 5) {
            args.push_back(key_length);
        }
        if (o.input_count > 6) {
            args.push_back(scale);
        }
        if (o.input_count > 7) {
            args.push_back(attention_sinks);
        }
        output = std::make_shared<v17::SparseAttention>(args, o.attention_key_block_size, o.query_block_size, o.causal);
    }
    output.get_node()->set_friendly_name("attention");
    return std::make_shared<Model>(OutputVector{output}, sinks, params);
}

class PagedSparseAttentionFusionFixture : public TransformationTestsF {
protected:
    pass::paged_attention::PaParams pa_params;
    std::unordered_set<std::string> retired;
    bool should_convert = true;

    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.set_per_pass_validation(false);
        manager.register_pass<pass::InjectionPass>([this](const std::shared_ptr<Model>& m) {
            pa_params = pass::paged_attention::PaParams{m->get_parameters()};
        });
        manager.register_pass<pass::PagedSparseAttentionFusion>(pa_params, retired);
        manager.register_pass<pass::InjectionPass>([this](const std::shared_ptr<Model>& m) {
            m->add_parameters(pa_params.items());
            m->validate_nodes_and_infer_types();
            const std::unordered_set<std::string> expected =
                should_convert ? std::unordered_set<std::string>{"past_key", "past_value"}
                               : std::unordered_set<std::string>{};
            EXPECT_EQ(retired, expected);
        });
    }
};

class PagedSparseAttentionFusionTest : public PagedSparseAttentionFusionFixture,
                                       public testing::WithParamInterface<GraphOptions> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GraphOptions>& info) {
        return info.param.name;
    }
};

TEST_P(PagedSparseAttentionFusionTest, ConvertsSparseAttention) {
    model = make_model(GetParam(), false);
    model_ref = make_model(GetParam(), true);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

std::vector<GraphOptions> positive_cases() {
    GraphOptions direct{"direct_uninitialized_states"};
    direct.gathered = false;
    direct.initialized = false;
    GraphOptions gathered{"gathered_uninitialized_states"};
    gathered.initialized = false;
    GraphOptions dynamic{"dynamic_f16"};
    dynamic.dynamic_shapes = true;
    dynamic.precision = element::f16;
    GraphOptions scale_sinks{"scale_and_sinks"};
    scale_sinks.input_count = 8;
    scale_sinks.precision = element::f16;
    GraphOptions scalar_scale{"scalar_scale"};
    scalar_scale.input_count = 7;
    scalar_scale.scalar_scale = true;
    scalar_scale.precision = element::bf16;
    GraphOptions no_key_length{"without_key_length"};
    no_key_length.input_count = 5;
    GraphOptions caches{"existing_caches"};
    caches.existing_caches = true;
    return {GraphOptions{"gathered_states"},
            direct,
            gathered,
            dynamic,
            scale_sinks,
            scalar_scale,
            no_key_length,
            caches};
}

INSTANTIATE_TEST_SUITE_P(PagedSparseAttentionFusion,
                         PagedSparseAttentionFusionTest,
                         testing::ValuesIn(positive_cases()),
                         PagedSparseAttentionFusionTest::getTestCaseName);

class PagedSparseAttentionFusionNotConvertedTest : public PagedSparseAttentionFusionTest {};

TEST_P(PagedSparseAttentionFusionNotConvertedTest, KeepsUnsupportedAttention) {
    model = make_model(GetParam(), false);
    should_convert = false;
}

std::vector<GraphOptions> negative_cases() {
    GraphOptions noncausal{"noncausal"};
    noncausal.causal = false;
    GraphOptions query_block{"query_block_size"};
    query_block.dynamic_shapes = true;
    query_block.query_block_size = 3;
    GraphOptions key_block{"different_key_block_size"};
    key_block.attention_key_block_size = 8;
    GraphOptions selections{"different_indexers"};
    selections.different_indexers = true;
    GraphOptions length{"different_current_length"};
    length.different_current_length = true;
    GraphOptions reinterpreted{"reinterpreted_selections"};
    reinterpreted.dynamic_shapes = true;
    reinterpreted.reinterpreted_selections = true;
    return {noncausal, query_block, key_block, selections, length, reinterpreted};
}

INSTANTIATE_TEST_SUITE_P(PagedSparseAttentionFusion,
                         PagedSparseAttentionFusionNotConvertedTest,
                         testing::ValuesIn(negative_cases()),
                         PagedSparseAttentionFusionTest::getTestCaseName);

TEST_F(PagedSparseAttentionFusionFixture, HonorsCallback) {
    model = make_model(GraphOptions{}, false);
    should_convert = false;
    manager.get_pass_config()->set_callback<pass::PagedSparseAttentionFusion>([](const std::shared_ptr<const Node>&) {
        return true;
    });
}

}  // namespace
