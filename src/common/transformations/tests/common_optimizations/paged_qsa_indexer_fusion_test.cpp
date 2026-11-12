// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_qsa_indexer_fusion.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "transformations/utils/utils.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v6 = ov::op::v6;
namespace v8 = ov::op::v8;
namespace op_internal = ov::op::internal;
using QSAIndexerInputIdx = op_internal::QSAIndexer::InputIdx;
using QSAIndexerOutputIdx = op_internal::QSAIndexer::OutputIdx;
using PagedQSAIndexerInputIdx = op_internal::PagedQSAIndexer::InputIdx;
using PagedQSAIndexerOutputIdx = op_internal::PagedQSAIndexer::OutputIdx;

constexpr int64_t query_heads = 4;
constexpr int64_t key_heads = 1;
constexpr int64_t head_size = 16;
constexpr int64_t rotary_dim = 8;
constexpr int64_t key_block_size = 4;
constexpr int64_t block_topk = 2;
constexpr int64_t selection_heads = 1;
constexpr float epsilon = 1e-6f;

Output<Node> i64(const std::vector<int64_t>& values) {
    return v0::Constant::create(element::i64, Shape{values.size()}, values);
}

std::shared_ptr<v0::Parameter> make_named_param(const std::string& name,
                                                element::Type type,
                                                const PartialShape& shape) {
    const auto param = std::make_shared<v0::Parameter>(type, shape);
    param->set_friendly_name(name);
    param->get_output_tensor(0).set_names({name});
    return param;
}

struct GraphOptions {
    bool gathered_states = false;
    bool variable_states = true;
    int64_t rope_length = -1;
};

// The dense QSAIndexer as created by FuseQSAIndexer, with its states kept in variables.
std::shared_ptr<Model> make_model(const GraphOptions& o, bool paged) {
    ParameterVector params;
    const auto param = [&params](element::Type type, const PartialShape& shape) {
        params.push_back(std::make_shared<v0::Parameter>(type, shape));
        return params.back();
    };
    const auto query = param(element::f32, PartialShape{-1, -1, query_heads, head_size});
    const auto raw_keys = param(element::f32, PartialShape{-1, -1, key_heads, head_size});
    const auto rope_cos = param(element::f32, PartialShape{-1, o.rope_length, rotary_dim});
    const auto rope_sin = param(element::f32, PartialShape{-1, o.rope_length, rotary_dim});
    const auto key_length = param(element::i32, PartialShape{-1});
    const auto cache_padding = param(element::f32, PartialShape{-1, key_heads, -1, head_size});
    const auto beam_idx = o.gathered_states ? param(element::i32, PartialShape{-1}) : nullptr;

    SinkVector sinks;
    std::vector<std::shared_ptr<op::util::Variable>> variables;
    const auto read_state = [&](const std::string& id, const PartialShape& shape) -> Output<Node> {
        if (!o.variable_states) {
            return param(element::f32, shape);
        }
        variables.push_back(std::make_shared<op::util::Variable>(op::util::VariableInfo{shape, element::f32, id}));
        Output<Node> state = std::make_shared<v6::ReadValue>(variables.back());
        if (o.gathered_states) {
            state = std::make_shared<v8::Gather>(state, beam_idx, v0::Constant::create(element::i64, Shape{}, {0}));
        }
        return state;
    };
    const auto past_block_keys = read_state("block_keys", PartialShape{-1, -1, head_size});
    const auto incomplete_raw_keys = read_state("incomplete_raw_keys", PartialShape{-1, key_block_size - 1, head_size});
    const auto incomplete_rope = read_state("incomplete_rope", PartialShape{-1, 2 * rotary_dim});
    const auto block_keys_cache = std::make_shared<v0::Concat>(
        OutputVector{std::make_shared<v0::Unsqueeze>(past_block_keys, i64({1})), cache_padding},
        2);
    const auto incomplete_raw_keys_with_head = std::make_shared<v0::Unsqueeze>(incomplete_raw_keys, i64({1}));
    const auto q_norm_weight = v0::Constant::create(element::f32, Shape{head_size}, {1.0f});
    const auto k_norm_weight = v0::Constant::create(element::f32, Shape{head_size}, {0.5f});

    op_internal::QSAIndexer::Config config;
    config.key_block_size = key_block_size;
    config.block_topk = block_topk;
    config.num_selection_heads = selection_heads;
    config.indexer_rotary_dim = rotary_dim;
    config.indexer_head_dim = head_size;
    config.epsilon = epsilon;

    Output<Node> indices, count, block_keys_state, raw_keys_state, rope_state;
    if (paged) {
        const auto past_lens = make_named_param("past_lens", element::i32, PartialShape{-1});
        const auto subsequence_begins = make_named_param("subsequence_begins", element::i32, PartialShape{-1});
        const auto block_indices = make_named_param("block_indices", element::i32, PartialShape{-1});
        const auto block_indices_begins = make_named_param("block_indices_begins", element::i32, PartialShape{-1});
        const auto index_key_cache =
            make_named_param("index_key_cache.0", element::f32, PartialShape{-1, key_heads, -1, head_size});
        const auto index_block_rope_cache =
            make_named_param("index_block_rope_cache.0", element::f32, PartialShape{-1, -1, 2 * rotary_dim});
        const auto block_index_keys_cache =
            make_named_param("block_index_keys_cache.0", element::f32, PartialShape{-1, key_heads, -1, head_size});
        params.insert(params.end(),
                      {past_lens,
                       subsequence_begins,
                       block_indices,
                       block_indices_begins,
                       index_key_cache,
                       index_block_rope_cache,
                       block_index_keys_cache});

        const auto flatten = [](const Output<Node>& data, int64_t width) {
            return std::make_shared<v1::Reshape>(data, i64({-1, width}), false);
        };
        OutputVector inputs(PagedQSAIndexerInputIdx::COUNT);
        inputs[PagedQSAIndexerInputIdx::INDEX_QUERY] = flatten(query, query_heads * head_size);
        inputs[PagedQSAIndexerInputIdx::RAW_INDEX_KEYS] = flatten(raw_keys, key_heads * head_size);
        inputs[PagedQSAIndexerInputIdx::ROPE_COS] = flatten(rope_cos, rotary_dim);
        inputs[PagedQSAIndexerInputIdx::ROPE_SIN] = flatten(rope_sin, rotary_dim);
        inputs[PagedQSAIndexerInputIdx::Q_NORM_WEIGHT] = q_norm_weight;
        inputs[PagedQSAIndexerInputIdx::K_NORM_WEIGHT] = k_norm_weight;
        inputs[PagedQSAIndexerInputIdx::PAST_LENS] = past_lens;
        inputs[PagedQSAIndexerInputIdx::SUBSEQUENCE_BEGINS] = subsequence_begins;
        inputs[PagedQSAIndexerInputIdx::BLOCK_INDICES] = block_indices;
        inputs[PagedQSAIndexerInputIdx::BLOCK_INDICES_BEGINS] = block_indices_begins;
        inputs[PagedQSAIndexerInputIdx::INDEX_KEY_CACHE] = index_key_cache;
        inputs[PagedQSAIndexerInputIdx::INDEX_BLOCK_ROPE_CACHE] = index_block_rope_cache;
        inputs[PagedQSAIndexerInputIdx::BLOCK_INDEX_KEYS_CACHE] = block_index_keys_cache;
        op_internal::PagedQSAIndexer::Config paged_config;
        paged_config.key_block_size = config.key_block_size;
        paged_config.block_topk = config.block_topk;
        paged_config.num_selection_heads = config.num_selection_heads;
        paged_config.indexer_rotary_dim = config.indexer_rotary_dim;
        paged_config.indexer_head_dim = config.indexer_head_dim;
        paged_config.epsilon = config.epsilon;
        const auto paged_indexer = std::make_shared<op_internal::PagedQSAIndexer>(inputs, paged_config);

        const auto batch_and_length = op::util::node_to_get_shape_value_of_indices_from_shape_source(query, {0, 1});
        const auto restore = [&](size_t output, const std::vector<int64_t>& tail, const std::vector<int64_t>& order) {
            const auto shape = std::make_shared<v0::Concat>(OutputVector{batch_and_length, i64(tail)}, 0);
            const auto token_major = std::make_shared<v1::Reshape>(paged_indexer->output(output), shape, false);
            return std::make_shared<v1::Transpose>(token_major, i64(order));
        };
        indices =
            restore(PagedQSAIndexerOutputIdx::SELECTED_BLOCK_INDICES, {selection_heads, block_topk}, {0, 2, 1, 3});
        count = restore(PagedQSAIndexerOutputIdx::SELECTED_BLOCK_COUNT, {selection_heads}, {0, 2, 1});
        block_keys_state = block_keys_cache;
        raw_keys_state = incomplete_raw_keys_with_head;
        rope_state = incomplete_rope;
    } else {
        OutputVector inputs(QSAIndexerInputIdx::COUNT);
        inputs[QSAIndexerInputIdx::INDEX_QUERY] = query;
        inputs[QSAIndexerInputIdx::RAW_INDEX_KEYS] = raw_keys;
        inputs[QSAIndexerInputIdx::ROPE_COS] = rope_cos;
        inputs[QSAIndexerInputIdx::ROPE_SIN] = rope_sin;
        inputs[QSAIndexerInputIdx::KEY_LENGTH] = key_length;
        inputs[QSAIndexerInputIdx::Q_NORM_WEIGHT] = q_norm_weight;
        inputs[QSAIndexerInputIdx::K_NORM_WEIGHT] = k_norm_weight;
        inputs[QSAIndexerInputIdx::BLOCK_INDEX_KEYS_CACHE] = block_keys_cache;
        inputs[QSAIndexerInputIdx::PAST_INCOMPLETE_RAW_K] = incomplete_raw_keys_with_head;
        inputs[QSAIndexerInputIdx::PAST_INCOMPLETE_ROPE] = incomplete_rope;
        const auto indexer = std::make_shared<op_internal::QSAIndexer>(inputs, config);
        indices = indexer->output(QSAIndexerOutputIdx::SELECTED_BLOCK_INDICES);
        count = indexer->output(QSAIndexerOutputIdx::SELECTED_BLOCK_COUNT);
        block_keys_state = indexer->output(QSAIndexerOutputIdx::BLOCK_INDEX_KEYS_CACHE);
        raw_keys_state = indexer->output(QSAIndexerOutputIdx::PAST_INCOMPLETE_RAW_K);
        rope_state = indexer->output(QSAIndexerOutputIdx::PAST_INCOMPLETE_ROPE);
    }

    ResultVector results{std::make_shared<v0::Result>(indices), std::make_shared<v0::Result>(count)};
    const OutputVector states{std::make_shared<v0::Squeeze>(block_keys_state, i64({1})),
                              std::make_shared<v0::Squeeze>(raw_keys_state, i64({1})),
                              rope_state};
    for (size_t i = 0; i < states.size(); ++i) {
        if (o.variable_states) {
            sinks.push_back(std::make_shared<v6::Assign>(states[i], variables[i]));
        } else {
            results.push_back(std::make_shared<v0::Result>(states[i]));
        }
    }
    return std::make_shared<Model>(results, sinks, params);
}

class QSAIndexerOperatorTest : public testing::Test, public testing::WithParamInterface<bool> {};

TEST_P(QSAIndexerOperatorTest, SelectionShapesAndCloneSupportFusion) {
    const auto model = make_model(GraphOptions{}, GetParam());
    const auto parameters = model->get_parameters();
    parameters.at(0)->set_partial_shape({2, 3, query_heads, head_size});
    parameters.at(1)->set_partial_shape({2, 3, key_heads, head_size});
    model->validate_nodes_and_infer_types();

    std::shared_ptr<Node> indexer;
    for (const auto& node : model->get_ops()) {
        if (GetParam() ? ov::is_type<op_internal::PagedQSAIndexer>(node) : ov::is_type<op_internal::QSAIndexer>(node)) {
            indexer = node;
            break;
        }
    }
    ASSERT_NE(indexer, nullptr);
    const auto expected_indices =
        GetParam() ? PartialShape{6, selection_heads, block_topk} : PartialShape{2, selection_heads, 3, block_topk};
    const auto expected_count = GetParam() ? PartialShape{6, selection_heads} : PartialShape{2, selection_heads, 3};
    EXPECT_EQ(indexer->get_output_partial_shape(0), expected_indices);
    EXPECT_EQ(indexer->get_output_partial_shape(1), expected_count);
    EXPECT_EQ(indexer->get_output_element_type(0), element::i32);
    EXPECT_EQ(indexer->get_output_element_type(1), element::i32);

    const auto clone = indexer->clone_with_new_inputs(indexer->input_values());
    EXPECT_EQ(clone->get_type_info(), indexer->get_type_info());
    EXPECT_EQ(clone->get_output_partial_shape(0), expected_indices);
    EXPECT_EQ(clone->get_output_partial_shape(1), expected_count);
    if (!GetParam()) {
        for (const auto& [output, input] :
             {std::pair{QSAIndexerOutputIdx::BLOCK_INDEX_KEYS_CACHE, QSAIndexerInputIdx::BLOCK_INDEX_KEYS_CACHE},
              std::pair{QSAIndexerOutputIdx::PAST_INCOMPLETE_RAW_K, QSAIndexerInputIdx::PAST_INCOMPLETE_RAW_K},
              std::pair{QSAIndexerOutputIdx::PAST_INCOMPLETE_ROPE, QSAIndexerInputIdx::PAST_INCOMPLETE_ROPE}}) {
            EXPECT_EQ(clone->get_output_partial_shape(output), clone->get_input_partial_shape(input));
            EXPECT_EQ(clone->get_output_element_type(output), clone->get_input_element_type(input));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(PagedQSAIndexerFusion,
                         QSAIndexerOperatorTest,
                         testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                             return info.param ? "Paged" : "Dense";
                         });

std::unordered_set<std::string> run_paged_qsa_indexer_fusion(const std::shared_ptr<Model>& model) {
    pass::paged_attention::PaParams pa_params{model->get_parameters()};
    std::unordered_set<std::string> var_ids_to_remove;
    pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<pass::PagedQSAIndexerFusion>(pa_params, var_ids_to_remove);
    manager.run_passes(model);
    model->add_parameters(pa_params.items());
    model->validate_nodes_and_infer_types();
    return var_ids_to_remove;
}

class PagedQSAIndexerFusionTest : public TransformationTestsF, public testing::WithParamInterface<GraphOptions> {};

class PagedQSAIndexerFusionConvertedTest : public PagedQSAIndexerFusionTest {};

TEST_P(PagedQSAIndexerFusionConvertedTest, ConvertsToPagedQSAIndexer) {
    model = make_model(GetParam(), false);
    model_ref = make_model(GetParam(), true);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    disable_rt_info_check();
    const auto var_ids_to_remove = run_paged_qsa_indexer_fusion(model);
    EXPECT_EQ(var_ids_to_remove,
              (std::unordered_set<std::string>{"block_keys", "incomplete_raw_keys", "incomplete_rope"}));
}

INSTANTIATE_TEST_SUITE_P(PagedQSAIndexerFusion,
                         PagedQSAIndexerFusionConvertedTest,
                         testing::Values(GraphOptions{}, GraphOptions{true}));

class PagedQSAIndexerFusionNotConvertedTest : public PagedQSAIndexerFusionTest {};

TEST_P(PagedQSAIndexerFusionNotConvertedTest, KeepsQSAIndexer) {
    model = make_model(GetParam(), false);
    model_ref = make_model(GetParam(), false);
    EXPECT_TRUE(run_paged_qsa_indexer_fusion(model).empty());
}

// States not read from variables cannot be converted to paged caches.
INSTANTIATE_TEST_SUITE_P(PagedQSAIndexerFusion,
                         PagedQSAIndexerFusionNotConvertedTest,
                         testing::Values(GraphOptions{false, false}));

std::shared_ptr<Model> make_model_with_modified_indexer(const std::string& name) {
    const auto model = make_model(GraphOptions{}, false);
    std::shared_ptr<op_internal::QSAIndexer> indexer;
    for (const auto& node : model->get_ops()) {
        indexer = ov::as_type_ptr<op_internal::QSAIndexer>(node);
        if (indexer) {
            break;
        }
    }
    OPENVINO_ASSERT(indexer, "Expected a dense QSAIndexer in the test model");
    auto inputs = indexer->input_values();
    auto config = indexer->get_config();
    if (name == "zero_block_size") {
        config.key_block_size = 0;
    } else if (name == "selection_heads") {
        config.num_selection_heads = 2;
    } else if (name == "nonfinite_scale") {
        config.scale = std::numeric_limits<float>::infinity();
    } else if (name == "head_dimension") {
        config.indexer_head_dim = head_size / 2;
    } else if (name == "head_count") {
        inputs[QSAIndexerInputIdx::RAW_INDEX_KEYS] =
            v0::Constant::create(element::f32, Shape{1, 1, 3, head_size}, {0.0f});
    } else if (name == "weight_shape") {
        inputs[QSAIndexerInputIdx::K_NORM_WEIGHT] = v0::Constant::create(element::f32, Shape{head_size - 1}, {1.0f});
    } else if (name == "weight_type") {
        inputs[QSAIndexerInputIdx::Q_NORM_WEIGHT] = v0::Constant::create(element::f16, Shape{head_size}, {1.0f});
    } else if (name == "rotary_dimension") {
        config.indexer_rotary_dim = rotary_dim - 1;
    } else if (name == "zero_rotary_dimension") {
        config.indexer_rotary_dim = 0;
        inputs[QSAIndexerInputIdx::ROPE_COS] = v0::Constant::create(element::f32, Shape{1, 1, 0}, std::vector<float>{});
        inputs[QSAIndexerInputIdx::ROPE_SIN] = v0::Constant::create(element::f32, Shape{1, 1, 0}, std::vector<float>{});
        const auto rope_state =
            ov::as_type_ptr<v6::ReadValue>(inputs[QSAIndexerInputIdx::PAST_INCOMPLETE_ROPE].get_node_shared_ptr());
        OPENVINO_ASSERT(rope_state);
        rope_state->get_variable()->update_data_shape(PartialShape{-1, 0});
        rope_state->validate_and_infer_types();
    } else if (name == "incompatible_key_rope_cos" || name == "incompatible_key_rope_sin") {
        model->get_parameters()[0]->set_partial_shape({1, Dimension(1, 3), query_heads, head_size});
        model->get_parameters()[1]->set_partial_shape({1, 1, key_heads, head_size});
        const auto idx =
            name == "incompatible_key_rope_cos" ? QSAIndexerInputIdx::ROPE_COS : QSAIndexerInputIdx::ROPE_SIN;
        inputs[idx] = v0::Constant::create(element::f32, Shape{1, 3, rotary_dim}, {1.0f});
    } else {
        model->get_parameters()[0]->set_partial_shape({1, 3, query_heads, head_size});
        model->get_parameters()[1]->set_partial_shape({1, 3, key_heads, head_size});
        inputs[QSAIndexerInputIdx::ROPE_COS] = v0::Constant::create(element::f32, Shape{1, 5, rotary_dim}, {1.0f});
    }
    const auto replacement = std::make_shared<op_internal::QSAIndexer>(inputs, config);
    ov::replace_node(indexer, replacement);
    model->validate_nodes_and_infer_types();
    return model;
}

class QSAIndexerInvalidInputsTest : public testing::TestWithParam<std::string> {};

TEST_P(QSAIndexerInvalidInputsTest, RejectsInvalidInputsAtConstruction) {
    EXPECT_THROW(make_model_with_modified_indexer(GetParam()), ov::NodeValidationFailure);
}

INSTANTIATE_TEST_SUITE_P(QSAIndexer,
                         QSAIndexerInvalidInputsTest,
                         testing::Values("zero_block_size",
                                         "selection_heads",
                                         "nonfinite_scale",
                                         "head_dimension",
                                         "head_count",
                                         "weight_shape",
                                         "weight_type",
                                         "rotary_dimension",
                                         "rope_token_count"),
                         [](const testing::TestParamInfo<std::string>& info) {
                             return info.param;
                         });

class PagedQSAIndexerInvalidProductionInputsTest : public TransformationTestsF,
                                                   public testing::WithParamInterface<std::string> {
public:
    void SetUp() override {
        TransformationTestsF::SetUp();
        model = make_model_with_modified_indexer(GetParam());
        pa_params = std::make_unique<pass::paged_attention::PaParams>(model->get_parameters());
        manager.register_pass<pass::PagedQSAIndexerFusion>(*pa_params, removed_ids);
    }

    void TearDown() override {
        TransformationTestsF::TearDown();
        ASSERT_NE(pa_params, nullptr);
        EXPECT_TRUE(removed_ids.empty());
        EXPECT_TRUE(pa_params->items().empty());
    }

private:
    std::unique_ptr<pass::paged_attention::PaParams> pa_params;
    std::unordered_set<std::string> removed_ids;
};

TEST_P(PagedQSAIndexerInvalidProductionInputsTest, DoesNotFuseOrAddCaches) {}

INSTANTIATE_TEST_SUITE_P(PagedQSAIndexerFusion,
                         PagedQSAIndexerInvalidProductionInputsTest,
                         testing::Values("zero_rotary_dimension",
                                         "incompatible_key_rope_cos",
                                         "incompatible_key_rope_sin"),
                         [](const testing::TestParamInfo<std::string>& info) {
                             return info.param;
                         });

}  // namespace
