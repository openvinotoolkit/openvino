// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_qsa_indexer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/rms_norm.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

struct GraphOptions {
    element::Type precision = element::f32;
    int64_t query_heads = 4;
    int64_t head_size = 128;
    int64_t key_block_size = 4;
    int64_t token_budget = 16;
    int64_t num_selection_heads = 1;
    float query_epsilon = 1e-6f;
    float key_epsilon = 1e-6f;
    bool decomposed_norm = false;
    bool reverse_norm_operands = false;
    bool dynamic_query_heads = false;
    bool dynamic_head_size = false;
    bool disable_key_norm = false;
    bool disable_rope = false;
    bool use_distinct_key_length = false;
    bool add_intervening_op = false;
    bool add_index_weights = false;
    bool extra_block_key_consumer = false;
    bool extra_state_consumer = false;
    bool dynamic_shapes = false;
};

struct QsaGraph {
    std::shared_ptr<Model> model;
    OutputVector fused_inputs;
    std::shared_ptr<op::v17::QSAPrepare> prepare;
    std::shared_ptr<op::v17::SparseAttentionIndexer> indexer;
};

std::shared_ptr<Model> make_model_from_outputs(const OutputVector& outputs, const ParameterVector& parameters) {
    ResultVector results;
    for (const auto& output : outputs) {
        results.push_back(std::make_shared<v0::Result>(output));
    }
    return std::make_shared<Model>(results, parameters);
}

QsaGraph build_graph(const GraphOptions& options = {}) {
    const auto batch = options.dynamic_shapes ? Dimension::dynamic() : Dimension(2);
    const auto sequence = options.dynamic_shapes ? Dimension::dynamic() : Dimension(3);
    const auto depth = options.dynamic_head_size ? Dimension::dynamic() : Dimension(options.head_size);
    const auto query_heads = options.dynamic_query_heads ? Dimension::dynamic() : Dimension(options.query_heads);
    const auto query_shape = PartialShape{batch, query_heads, sequence, depth};
    const auto keys_shape = PartialShape{batch, sequence, depth};
    const auto cache_shape = PartialShape{batch, Dimension::dynamic(), depth};
    const auto incomplete_keys_shape = PartialShape{batch, Dimension(options.key_block_size - 1), depth};
    const auto rope_shape = PartialShape{batch, sequence, Dimension(16)};
    const auto incomplete_rope_shape = PartialShape{batch, Dimension(32)};

    auto query = std::make_shared<v0::Parameter>(options.precision, query_shape);
    auto raw_keys = std::make_shared<v0::Parameter>(options.precision, keys_shape);
    auto cache = std::make_shared<v0::Parameter>(options.precision, cache_shape);
    auto key_length = std::make_shared<v0::Parameter>(element::i32, PartialShape{batch});
    auto query_weight = std::make_shared<v0::Parameter>(options.precision, PartialShape{depth});
    auto incomplete_keys = std::make_shared<v0::Parameter>(options.precision, incomplete_keys_shape);
    Output<Node> rope_cos;
    Output<Node> rope_sin;
    Output<Node> incomplete_rope;
    ParameterVector parameters{query, raw_keys, cache, key_length, query_weight, incomplete_keys};
    if (options.disable_rope) {
        const auto empty = v0::Constant::create(options.precision, Shape{0}, std::vector<float>{});
        rope_cos = empty;
        rope_sin = empty;
        incomplete_rope = empty;
    } else {
        const auto rope_cos_parameter = std::make_shared<v0::Parameter>(options.precision, rope_shape);
        const auto rope_sin_parameter = std::make_shared<v0::Parameter>(options.precision, rope_shape);
        const auto incomplete_rope_parameter =
            std::make_shared<v0::Parameter>(options.precision, incomplete_rope_shape);
        rope_cos = rope_cos_parameter;
        rope_sin = rope_sin_parameter;
        incomplete_rope = incomplete_rope_parameter;
        parameters.push_back(rope_cos_parameter);
        parameters.push_back(rope_sin_parameter);
        parameters.push_back(incomplete_rope_parameter);
    }
    Output<Node> key_weight;
    if (options.disable_key_norm) {
        key_weight = v0::Constant::create(options.precision, Shape{0}, std::vector<float>{});
    } else {
        const auto key_weight_parameter = std::make_shared<v0::Parameter>(options.precision, PartialShape{depth});
        key_weight = key_weight_parameter;
        parameters.push_back(key_weight_parameter);
    }

    Output<Node> normalized_query;
    if (options.decomposed_norm) {
        const auto axes = v0::Constant::create(element::i64, Shape{1}, {-1});
        const auto exponent = v0::Constant::create(options.precision, Shape{}, {2.0f});
        const auto epsilon = v0::Constant::create(options.precision, Shape{}, {options.query_epsilon});
        const auto square = std::make_shared<v1::Power>(query, exponent);
        const auto mean = std::make_shared<v1::ReduceMean>(square, axes, true);
        const auto variance = options.reverse_norm_operands ? std::make_shared<v1::Add>(epsilon, mean)
                                                            : std::make_shared<v1::Add>(mean, epsilon);
        const auto root = std::make_shared<v0::Sqrt>(variance);
        const auto divided = std::make_shared<v1::Divide>(query, root);
        normalized_query = options.reverse_norm_operands
                               ? Output<Node>(std::make_shared<v1::Multiply>(query_weight, divided))
                               : Output<Node>(std::make_shared<v1::Multiply>(divided, query_weight));
    } else {
        const auto axes = v0::Constant::create(element::i64, Shape{1}, {-1});
        normalized_query = std::make_shared<op::internal::RMSNorm>(query, axes, query_weight, options.query_epsilon);
    }
    if (options.add_intervening_op) {
        normalized_query = std::make_shared<op::v1::Multiply>(normalized_query,
                                                              v0::Constant::create(options.precision, Shape{}, {1.0f}));
    }

    auto prepare = std::make_shared<op::v17::QSAPrepare>(raw_keys,
                                                         cache,
                                                         key_length,
                                                         rope_cos,
                                                         rope_sin,
                                                         key_weight,
                                                         incomplete_keys,
                                                         incomplete_rope,
                                                         options.key_block_size,
                                                         options.key_epsilon);
    Output<Node> indexer_key_length = key_length;
    if (options.use_distinct_key_length) {
        const auto distinct_key_length = std::make_shared<v0::Parameter>(element::i32, PartialShape{batch});
        indexer_key_length = distinct_key_length;
        parameters.push_back(distinct_key_length);
    }
    std::shared_ptr<op::v17::SparseAttentionIndexer> indexer;
    if (options.add_index_weights) {
        auto index_weights =
            std::make_shared<v0::Parameter>(options.precision,
                                            PartialShape{batch, sequence, Dimension(options.query_heads)});
        parameters.push_back(index_weights);
        indexer = std::make_shared<op::v17::SparseAttentionIndexer>(normalized_query,
                                                                    prepare->output(0),
                                                                    indexer_key_length,
                                                                    index_weights,
                                                                    options.key_block_size,
                                                                    options.token_budget,
                                                                    options.num_selection_heads);
    } else {
        indexer = std::make_shared<op::v17::SparseAttentionIndexer>(normalized_query,
                                                                    prepare->output(0),
                                                                    indexer_key_length,
                                                                    options.key_block_size,
                                                                    options.token_budget,
                                                                    options.num_selection_heads);
    }

    indexer->output(0).get_tensor().set_names({"selected_indices"});
    indexer->output(1).get_tensor().set_names({"selected_count"});
    prepare->output(0).get_tensor().set_names({"block_index_keys"});
    prepare->output(1).get_tensor().set_names({"updated_incomplete_raw_keys"});
    prepare->output(2).get_tensor().set_names({"updated_incomplete_rope"});

    OutputVector outputs{indexer->output(0), indexer->output(1), prepare->output(1), prepare->output(2)};
    if (options.extra_block_key_consumer) {
        outputs.push_back(prepare->output(0));
    }
    if (options.extra_state_consumer) {
        outputs.push_back(prepare->output(1));
    }
    prepare->get_rt_info()["qsa_test"] = std::string("runtime-info");

    OutputVector fused_inputs{query,
                              raw_keys,
                              rope_cos,
                              rope_sin,
                              key_length,
                              query_weight,
                              key_weight,
                              cache,
                              incomplete_keys,
                              incomplete_rope};
    return {make_model_from_outputs(outputs, parameters), fused_inputs, prepare, indexer};
}

size_t count_ops(const std::shared_ptr<Model>& model, const std::string& type_name) {
    const auto ordered_ops = model->get_ordered_ops();
    return std::count_if(ordered_ops.begin(), ordered_ops.end(), [&](const auto& node) {
        return node->get_type_name() == type_name;
    });
}

void run_fusion(const std::shared_ptr<Model>& model, const pass::FuseQSAIndexer::Config& limits = {}) {
    pass::Manager manager;
    manager.register_pass<pass::FuseQSAIndexer>(limits);
    manager.run_passes(model);
}

// Kernel limits that the GPU plugin passes to FuseQSAIndexer.
pass::FuseQSAIndexer::Config gpu_limits() {
    pass::FuseQSAIndexer::Config limits;
    limits.supported_num_selection_heads = {1};
    limits.max_epsilon_relative_difference = 1e-4f;
    limits.require_static_head_dims = true;
    limits.require_rope = true;
    return limits;
}

std::shared_ptr<op::internal::QSAIndexer> find_fused(const std::shared_ptr<Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto fused = as_type_ptr<op::internal::QSAIndexer>(node)) {
            return fused;
        }
    }
    return nullptr;
}

TEST(FuseQSAIndexer, FusesSupportedGraphAndMapsAllInputsAndOutputs) {
    auto graph = build_graph();
    const auto indexer_name = graph.indexer->get_friendly_name();
    const auto original_results = graph.model->get_results();
    std::vector<std::string> result_names;
    for (const auto& result : original_results) {
        result_names.push_back(result->get_friendly_name());
    }
    run_fusion(graph.model);

    ASSERT_EQ(count_ops(graph.model, "QSAIndexer"), 1);
    EXPECT_EQ(count_ops(graph.model, "QSAPrepare"), 0);
    EXPECT_EQ(count_ops(graph.model, "SparseAttentionIndexer"), 0);
    const auto ordered_ops = graph.model->get_ordered_ops();
    const auto fused_it = std::find_if(ordered_ops.begin(), ordered_ops.end(), [](const auto& node) {
        return is_type<op::internal::QSAIndexer>(node);
    });
    const auto fused = fused_it == ordered_ops.end() ? nullptr : as_type_ptr<op::internal::QSAIndexer>(*fused_it);
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->get_friendly_name(), indexer_name);
    EXPECT_EQ(fused->get_config().key_block_size, 4);
    EXPECT_EQ(fused->get_config().token_budget, 16);
    EXPECT_FLOAT_EQ(fused->get_config().q_epsilon, 1e-6f);
    EXPECT_FLOAT_EQ(fused->get_config().k_epsilon, 1e-6f);
    const std::vector<std::string> output_names{
        "selected_indices",
        "selected_count",
        "block_index_keys",
        "updated_incomplete_raw_keys",
        "updated_incomplete_rope",
    };
    for (size_t i = 0; i < output_names.size(); ++i) {
        const auto names = fused->output(i).get_tensor().get_names();
        ASSERT_EQ(names.size(), 1);
        EXPECT_TRUE(names.count(output_names[i]));
    }
    ASSERT_EQ(fused->get_input_size(), graph.fused_inputs.size());
    for (size_t i = 0; i < graph.fused_inputs.size(); ++i) {
        EXPECT_EQ(fused->input_value(i), graph.fused_inputs[i]);
    }
    EXPECT_EQ(fused->get_output_partial_shape(2), graph.prepare->get_output_partial_shape(0));
    EXPECT_EQ(fused->get_output_partial_shape(3), graph.prepare->get_output_partial_shape(1));
    EXPECT_EQ(fused->get_output_partial_shape(4), graph.prepare->get_output_partial_shape(2));
    ASSERT_EQ(graph.model->get_results().size(), 4);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(graph.model->get_results()[i], original_results[i]);
        EXPECT_EQ(graph.model->get_results()[i]->get_friendly_name(), result_names[i]);
        EXPECT_EQ(graph.model->get_results()[i]->input_value(0), fused->output(i < 2 ? i : i + 1));
    }
    EXPECT_EQ(fused->get_rt_info().at("qsa_test").as<std::string>(), "runtime-info");

    const auto node_count = graph.model->get_ordered_ops().size();
    run_fusion(graph.model);
    EXPECT_EQ(count_ops(graph.model, "QSAIndexer"), 1);
    EXPECT_EQ(graph.model->get_ordered_ops().size(), node_count);
}

TEST(FuseQSAIndexer, AcceptsDecomposedRMSNormAndReversedOperands) {
    GraphOptions options;
    options.decomposed_norm = true;
    options.reverse_norm_operands = true;
    options.query_epsilon = 1e-5f;
    options.key_epsilon = 1e-5f;
    auto graph = build_graph(options);
    run_fusion(graph.model);
    const auto ordered_ops = graph.model->get_ordered_ops();
    const auto fused_it = std::find_if(ordered_ops.begin(), ordered_ops.end(), [](const auto& node) {
        return is_type<op::internal::QSAIndexer>(node);
    });
    const auto fused = fused_it == ordered_ops.end() ? nullptr : as_type_ptr<op::internal::QSAIndexer>(*fused_it);
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->input_value(0), graph.fused_inputs[0]);
    EXPECT_EQ(fused->input_value(5), graph.fused_inputs[5]);
    EXPECT_FLOAT_EQ(fused->get_config().q_epsilon, options.query_epsilon);
}

TEST(FuseQSAIndexer, ConfiguredEpsilonToleranceAcceptsCloseQueryAndKeyEpsilons) {
    const auto limits = gpu_limits();
    GraphOptions options;
    options.query_epsilon = 1e-5f * (1.0f + *limits.max_epsilon_relative_difference / 2);
    options.key_epsilon = 1e-5f;
    auto graph = build_graph(options);
    run_fusion(graph.model, limits);
    EXPECT_EQ(count_ops(graph.model, "QSAIndexer"), 1);
}

TEST(FuseQSAIndexer, AcceptsDynamicBatchAndSequenceShapes) {
    GraphOptions options;
    options.dynamic_shapes = true;
    auto graph = build_graph(options);
    run_fusion(graph.model);
    const auto ordered_ops = graph.model->get_ordered_ops();
    const auto fused_it = std::find_if(ordered_ops.begin(), ordered_ops.end(), [](const auto& node) {
        return is_type<op::internal::QSAIndexer>(node);
    });
    const auto fused = fused_it == ordered_ops.end() ? nullptr : as_type_ptr<op::internal::QSAIndexer>(*fused_it);
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->get_output_partial_shape(0), (PartialShape{-1, 1, -1, 4}));
    EXPECT_EQ(fused->get_output_partial_shape(3), (PartialShape{-1, 3, 128}));
}

TEST(FuseQSAIndexer, DoesNotRejectFP32) {
    auto graph = build_graph();
    run_fusion(graph.model);
    EXPECT_EQ(count_ops(graph.model, "QSAIndexer"), 1);
}

TEST(FuseQSAIndexer, DoesNotFixHeadCountHeadSizeBlockSizeOrTokenBudget) {
    const auto with = [](auto&& setter) {
        GraphOptions options;
        setter(options);
        return options;
    };
    const std::vector<std::pair<std::string, GraphOptions>> cases{
        {"query_heads", with([](GraphOptions& o) {
             o.query_heads = 8;
         })},
        {"head_size", with([](GraphOptions& o) {
             o.head_size = 64;
         })},
        {"block_size", with([](GraphOptions& o) {
             o.key_block_size = 8;
         })},
        {"token_budget", with([](GraphOptions& o) {
             o.token_budget = 516;
         })},
        {"matching_epsilons_outside_former_range", with([](GraphOptions& o) {
             o.query_epsilon = 2e-5f;
             o.key_epsilon = 2e-5f;
         })},
    };

    for (const auto& [name, options] : cases) {
        SCOPED_TRACE(name);
        auto graph = build_graph(options);
        run_fusion(graph.model, gpu_limits());
        EXPECT_EQ(count_ops(graph.model, "QSAIndexer"), 1);
    }
}

std::vector<std::pair<std::string, GraphOptions>> device_specific_cases() {
    const auto with = [](auto&& setter) {
        GraphOptions options;
        setter(options);
        return options;
    };
    return {
        {"dynamic_query_heads", with([](GraphOptions& o) {
             o.dynamic_query_heads = true;
         })},
        {"dynamic_head_size", with([](GraphOptions& o) {
             o.dynamic_head_size = true;
         })},
        {"num_selection_heads", with([](GraphOptions& o) {
             o.num_selection_heads = 2;
         })},
        {"disabled_rope", with([](GraphOptions& o) {
             o.disable_rope = true;
         })},
        {"mismatched_epsilons", with([](GraphOptions& o) {
             o.query_epsilon = 1e-6f;
             o.key_epsilon = 2e-6f;
         })},
        {"mismatched_decomposed_epsilon", with([](GraphOptions& o) {
             o.decomposed_norm = true;
             o.query_epsilon = 2e-6f;
             o.key_epsilon = 1e-6f;
         })},
    };
}

TEST(FuseQSAIndexer, DefaultConfigFusesGraphsOutsideDeviceKernelLimits) {
    for (const auto& [name, options] : device_specific_cases()) {
        SCOPED_TRACE(name);
        auto graph = build_graph(options);
        run_fusion(graph.model);
        const auto fused = find_fused(graph.model);
        ASSERT_NE(fused, nullptr);
        EXPECT_EQ(fused->get_config().num_selection_heads, options.num_selection_heads);
        EXPECT_FLOAT_EQ(fused->get_config().q_epsilon, options.query_epsilon);
        EXPECT_FLOAT_EQ(fused->get_config().k_epsilon, options.key_epsilon);
    }
}

TEST(FuseQSAIndexer, DefaultConfigPreservesDisabledRoPEStateShape) {
    GraphOptions options;
    options.disable_rope = true;
    auto graph = build_graph(options);
    run_fusion(graph.model);
    const auto fused = find_fused(graph.model);
    ASSERT_NE(fused, nullptr);
    EXPECT_EQ(fused->get_output_partial_shape(4), (PartialShape{2, 0}));
    EXPECT_EQ(graph.model->get_results()[3]->input_value(0), fused->output(4));
}

TEST(FuseQSAIndexer, RejectsGraphsOutsideConfiguredKernelLimits) {
    for (const auto& [name, options] : device_specific_cases()) {
        SCOPED_TRACE(name);
        auto graph = build_graph(options);
        run_fusion(graph.model, gpu_limits());
        EXPECT_EQ(count_ops(graph.model, "QSAIndexer"), 0);
        EXPECT_EQ(count_ops(graph.model, "QSAPrepare"), 1);
        EXPECT_EQ(count_ops(graph.model, "SparseAttentionIndexer"), 1);
    }
}

TEST(FuseQSAIndexer, ConfiguredKernelLimitsAcceptSupportedGraph) {
    auto graph = build_graph();
    run_fusion(graph.model, gpu_limits());
    EXPECT_EQ(count_ops(graph.model, "QSAIndexer"), 1);
}

TEST(FuseQSAIndexer, RejectsInvalidConfigurationAndProvenance) {
    const std::vector<std::pair<std::string, GraphOptions>> negative_cases{{"disabled_key_norm",
                                                                            [] {
                                                                                GraphOptions options;
                                                                                options.disable_key_norm = true;
                                                                                return options;
                                                                            }()},
                                                                           {"mismatched_key_length",
                                                                            [] {
                                                                                GraphOptions options;
                                                                                options.use_distinct_key_length = true;
                                                                                return options;
                                                                            }()},
                                                                           {"intervening_operation",
                                                                            [] {
                                                                                GraphOptions options;
                                                                                options.add_intervening_op = true;
                                                                                return options;
                                                                            }()},
                                                                           {"optional_index_weights",
                                                                            [] {
                                                                                GraphOptions options;
                                                                                options.add_index_weights = true;
                                                                                return options;
                                                                            }()},
                                                                           {"extra_block_key_consumer",
                                                                            [] {
                                                                                GraphOptions options;
                                                                                options.extra_block_key_consumer = true;
                                                                                return options;
                                                                            }()},
                                                                           {"extra_state_consumer", [] {
                                                                                GraphOptions options;
                                                                                options.extra_state_consumer = true;
                                                                                return options;
                                                                            }()}};

    for (const auto& [name, options] : negative_cases) {
        SCOPED_TRACE(name);
        auto graph = build_graph(options);
        run_fusion(graph.model);
        EXPECT_EQ(count_ops(graph.model, "QSAIndexer"), 0);
        EXPECT_EQ(count_ops(graph.model, "QSAPrepare"), 1);
        EXPECT_EQ(count_ops(graph.model, "SparseAttentionIndexer"), 1);
    }
}

TEST(FuseQSAIndexer, DoesNotFuseRegisteredTinyRandomModel) {
    const auto* xml_path = std::getenv("OPENVINO_QSA_TEST_MODEL_XML");
    if (!xml_path) {
        GTEST_SKIP() << "OPENVINO_QSA_TEST_MODEL_XML is not set.";
    }

    ov::Core core;
    auto model = core.read_model(xml_path);
    run_fusion(model);
    EXPECT_EQ(count_ops(model, "QSAIndexer"), 0);
}

}  // namespace
