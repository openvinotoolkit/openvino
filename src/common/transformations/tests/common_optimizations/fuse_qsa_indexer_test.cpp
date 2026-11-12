// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_qsa_indexer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/rms_fusion.hpp"
#include "transformations/utils/utils.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v17 = ov::op::v17;
namespace op_internal = ov::op::internal;
using ov::op::util::make_try_fold;
using QSAIndexerInputIdx = op_internal::QSAIndexer::InputIdx;
using QSAIndexerOutputIdx = op_internal::QSAIndexer::OutputIdx;

enum class QueryLayout {
    BHLD,            // [B, H, L, D] raw query, RMSNorm and RoPE in this layout
    BLHDTransposed,  // [B, L, H, D] raw query transposed to [B, H, L, D] after the norm, RoPE after the Transpose
    BLHDRoPEBeforeTranspose,  // [B, L, H, D] raw query with RMSNorm and RoPE applied before the Transpose
    BLHDTransposeAfterRoPE,   // [B, L, H, D] raw query, followed by a Transpose for SparseAttentionIndexer
};

struct GraphOptions {
    std::string name;
    element::Type precision = element::f32;
    QueryLayout query_layout = QueryLayout::BLHDTransposed;
    // cos / sin have a single row, which QSAPrepare broadcasts over the batch.
    bool single_row_rope = false;
    bool dynamic_shapes = false;
    element::Type key_length_type = element::i32;
    // The block-key cache and the incomplete raw keys of QSAPrepare are also model outputs.
    bool extra_state_consumers = false;
    // The SparseAttentionIndexer query is also a model output.
    bool extra_query_consumer = false;
    // Plugin compilation canonicalizes the decomposed query RMSNorm and RoPE before query absorption.
    // Only for QueryLayout::BLHDTransposed.
    bool decomposed_query = false;
    bool reshape_rope_rows = false;
    // Negative cases.
    bool missing_query_norm = false;
    bool norm_after_rope = false;
    bool wrong_rope_reshape = false;
    bool query_rope = true;
    bool distinct_query_rope = false;
    bool distinct_query_sin = false;
    bool interleaved_query_rope = false;
    bool input_interleaved_query_rope = false;
    bool cached_query_rope = false;
    bool affine_query = false;
    bool converted_query = false;
    bool rms_changes_precision = false;
    int64_t rms_axis = -1;
    int64_t query_rope_axis = 1;
    int64_t query_rotary_dim = 0;
    float query_epsilon = 0;  // 0 is the QSAPrepare epsilon
    int64_t rotary_dim = 0;   // 0 is the whole head
    bool dynamic_query_heads = false;
    bool dynamic_head_size = false;
    bool dynamic_key_length_rank = false;
    bool disable_key_norm = false;
    bool disable_rope = false;
    bool use_distinct_key_length = false;
    bool add_index_weights = false;
    int64_t num_selection_heads = 1;
    int64_t indexer_key_block_size = 4;
    int64_t query_sequence = 3;
    int64_t keys_sequence = 3;
};

constexpr int64_t query_heads = 4;
constexpr int64_t head_size = 16;
constexpr int64_t key_block_size = 4;
constexpr int64_t token_budget = 8;
constexpr float epsilon = 1e-6f;

Output<Node> i64(const std::vector<int64_t>& values) {
    return v0::Constant::create(element::i64, Shape{values.size()}, values);
}

// The model inputs, which FuseQSAIndexer keeps, and the SparseAttentionIndexer query built from the raw query.
struct SharedGraph {
    ParameterVector parameters;
    Output<Node> raw_query;
    Output<Node> query;  // SparseAttentionIndexer query
    Output<Node> raw_keys, cache, key_length, rope_cos, rope_sin, key_weight, incomplete_keys, incomplete_rope;
    Output<Node> indexer_key_length, index_weights;
};

Output<Node> head_transpose(const Output<Node>& data) {
    return std::make_shared<v1::Transpose>(data, i64({0, 2, 1, 3}));
}

// The non-affine RMSNorm as exported: x * 1 / sqrt(mean(x ^ 2) + eps).
Output<Node> decomposed_rms(const Output<Node>& x, int64_t axis, float eps) {
    const auto type = x.get_element_type();
    const auto square = std::make_shared<v1::Power>(x, v0::Constant::create(type, Shape{}, {2.0f}));
    const auto mean = std::make_shared<v1::ReduceMean>(square, i64({axis}), true);
    const auto sum = std::make_shared<v1::Add>(mean, v0::Constant::create(type, Shape{}, {eps}));
    const auto inv =
        std::make_shared<v1::Divide>(v0::Constant::create(type, Shape{}, {1.0f}), std::make_shared<v0::Sqrt>(sum));
    return std::make_shared<v1::Multiply>(x, inv);
}

SharedGraph make_shared_graph(const GraphOptions& o) {
    const auto batch = o.dynamic_shapes ? Dimension::dynamic() : Dimension(2);
    const auto sequence = o.dynamic_shapes ? Dimension::dynamic() : Dimension(o.keys_sequence);
    const auto query_sequence = o.dynamic_shapes ? Dimension::dynamic() : Dimension(o.query_sequence);
    const auto depth = o.dynamic_head_size ? Dimension::dynamic() : Dimension(head_size);
    const auto heads = o.dynamic_query_heads ? Dimension::dynamic() : Dimension(query_heads);
    const bool blhd = o.query_layout != QueryLayout::BHLD;
    const auto rotary = o.rotary_dim != 0 ? o.rotary_dim : head_size;
    const auto rope_shape = PartialShape{o.single_row_rope ? Dimension(1) : batch, sequence, Dimension(rotary)};

    SharedGraph g;
    const auto parameter = [&g](element::Type type, const PartialShape& shape) {
        g.parameters.push_back(std::make_shared<v0::Parameter>(type, shape));
        return g.parameters.back()->output(0);
    };
    g.raw_query = parameter(
        o.rms_changes_precision ? element::f16 : o.precision,
        blhd ? PartialShape{batch, query_sequence, heads, depth} : PartialShape{batch, heads, query_sequence, depth});
    g.raw_keys = parameter(o.precision, PartialShape{batch, sequence, depth});
    g.cache = parameter(o.precision, PartialShape{batch, Dimension::dynamic(), depth});
    g.key_length =
        parameter(o.key_length_type, o.dynamic_key_length_rank ? PartialShape::dynamic() : PartialShape{batch});
    g.incomplete_keys = parameter(o.precision, PartialShape{batch, Dimension(key_block_size - 1), depth});
    const auto cos = parameter(o.precision, rope_shape);
    const auto sin = parameter(o.precision, rope_shape);
    if (o.disable_rope) {
        g.rope_cos = g.rope_sin = g.incomplete_rope = v0::Constant::create(o.precision, Shape{0}, std::vector<float>{});
    } else {
        g.rope_cos = cos;
        g.rope_sin = sin;
        g.incomplete_rope = parameter(o.precision, PartialShape{batch, Dimension(2 * rotary)});
    }
    g.key_weight = o.disable_key_norm ? v0::Constant::create(o.precision, Shape{0}, std::vector<float>{})->output(0)
                                      : parameter(o.precision, PartialShape{depth});
    g.indexer_key_length = o.use_distinct_key_length ? parameter(o.key_length_type, PartialShape{batch}) : g.key_length;
    if (o.add_index_weights) {
        g.index_weights = parameter(o.precision, PartialShape{batch, query_sequence, heads});
    }
    const auto rms_epsilon = o.query_epsilon != 0 ? o.query_epsilon : epsilon;
    const auto rms =
        o.decomposed_query ? decomposed_rms(g.raw_query, o.rms_axis, rms_epsilon)
        : o.affine_query
            ? std::make_shared<op_internal::RMS>(g.raw_query,
                                                 v0::Constant::create(o.precision, Shape{head_size}, {0.5f}),
                                                 rms_epsilon,
                                                 o.precision,
                                                 o.rms_axis)
            : std::make_shared<op_internal::RMS>(g.raw_query, rms_epsilon, o.precision, o.rms_axis);

    const Output<Node> query_input = o.missing_query_norm ? g.raw_query : Output<Node>(rms);
    if (o.decomposed_query) {
        // The rotate-half RoPE of the transposed query, as exported: x * cos + concat(-x2, x1) * sin.
        constexpr int64_t last_axis = 3;
        const auto x = head_transpose(query_input);
        const auto half = rotary / 2;
        const auto halves = std::make_shared<v1::VariadicSplit>(x, i64({last_axis}), i64({half, half}));
        const auto minus_x2 =
            std::make_shared<v1::Multiply>(halves->output(1), v0::Constant::create(o.precision, Shape{}, {-1.0f}));
        const auto rotated = std::make_shared<v0::Concat>(OutputVector{minus_x2, halves->output(0)}, -1);
        const auto mul_cos = std::make_shared<v1::Multiply>(x, std::make_shared<v0::Unsqueeze>(cos, i64({1})));
        const auto mul_sin = std::make_shared<v1::Multiply>(rotated, std::make_shared<v0::Unsqueeze>(sin, i64({1})));
        g.query = std::make_shared<v1::Add>(mul_cos, mul_sin);
    } else if (o.query_rope) {
        const auto query_cos = o.distinct_query_rope ? parameter(o.precision, rope_shape) : cos;
        const auto query_sin = o.distinct_query_sin ? parameter(o.precision, rope_shape) : sin;
        op_internal::RoPE::Config rope_config;
        rope_config.rotary_ndims = static_cast<size_t>(o.query_rotary_dim ? o.query_rotary_dim : rotary);
        rope_config.is_interleaved = o.interleaved_query_rope;
        rope_config.input_interleaved = o.input_interleaved_query_rope;
        rope_config.use_rope_cache = o.cached_query_rope;
        rope_config.input_trans0213 = o.query_layout == QueryLayout::BLHDTransposed;
        rope_config.output_trans0213 = o.query_layout == QueryLayout::BLHDRoPEBeforeTranspose;
        const auto with_head_axis = [&](const Output<Node>& rows) -> Output<Node> {
            if (o.reshape_rope_rows) {
                return std::make_shared<v1::Reshape>(rows,
                                                     i64(o.wrong_rope_reshape ? std::vector<int64_t>{1, 2, 1, rotary}
                                                                              : std::vector<int64_t>{0, 1, 1, rotary}),
                                                     true);
            }
            return std::make_shared<v0::Unsqueeze>(rows, i64({o.query_rope_axis}));
        };
        g.query = std::make_shared<op_internal::RoPE>(
            OutputVector{query_input, with_head_axis(query_cos), with_head_axis(query_sin)},
            rope_config);
        if (o.query_layout == QueryLayout::BLHDTransposeAfterRoPE) {
            g.query = head_transpose(g.query);
        }
    } else {
        g.query = query_input;
    }
    if (o.norm_after_rope) {
        g.query = std::make_shared<op_internal::RMS>(g.query, rms_epsilon, o.precision, o.rms_axis);
    }
    if (o.converted_query) {
        g.query = std::make_shared<v0::Convert>(g.query, o.precision);
    }
    return g;
}

std::shared_ptr<Model> make_model(const SharedGraph& g,
                                  const std::vector<Output<Node>>& outputs,
                                  const GraphOptions& o) {
    ResultVector results;
    for (const auto& output : outputs) {
        results.push_back(std::make_shared<v0::Result>(output));
    }
    if (o.extra_state_consumers) {
        // The block-key cache and the incomplete raw keys, e.g. consumed by cache Assign ops.
        results.push_back(std::make_shared<v0::Result>(outputs[4]));
        results.push_back(std::make_shared<v0::Result>(outputs[2]));
    } else {
        results.pop_back();
    }
    if (o.extra_query_consumer) {
        results.push_back(std::make_shared<v0::Result>(g.query));
    }
    return std::make_shared<Model>(results, g.parameters);
}

std::shared_ptr<Model> getModel(const GraphOptions& o) {
    const auto g = make_shared_graph(o);
    const auto prepare = std::make_shared<v17::QSAPrepare>(g.raw_keys,
                                                           g.cache,
                                                           g.key_length,
                                                           g.rope_cos,
                                                           g.rope_sin,
                                                           g.key_weight,
                                                           g.incomplete_keys,
                                                           g.incomplete_rope,
                                                           key_block_size,
                                                           epsilon);
    std::shared_ptr<v17::SparseAttentionIndexer> indexer;
    if (o.add_index_weights) {
        indexer = std::make_shared<v17::SparseAttentionIndexer>(g.query,
                                                                prepare->output(0),
                                                                g.indexer_key_length,
                                                                g.index_weights,
                                                                o.indexer_key_block_size,
                                                                token_budget,
                                                                o.num_selection_heads);
    } else {
        indexer = std::make_shared<v17::SparseAttentionIndexer>(g.query,
                                                                prepare->output(0),
                                                                g.indexer_key_length,
                                                                o.indexer_key_block_size,
                                                                token_budget,
                                                                o.num_selection_heads);
    }
    const std::vector<Output<Node>> outputs{indexer->output(0),
                                            indexer->output(1),
                                            prepare->output(1),
                                            prepare->output(2),
                                            prepare->output(0)};
    return make_model(g, outputs, o);
}

std::shared_ptr<Model> getModelRef(const GraphOptions& o) {
    const auto g = make_shared_graph(o);
    const auto rotary = o.rotary_dim != 0 ? o.rotary_dim : head_size;
    const auto dims_of = [](const Output<Node>& data, const std::vector<size_t>& axes) {
        return ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(data, axes);
    };

    Output<Node> key_length = g.key_length;
    if (o.dynamic_shapes) {
        key_length = std::make_shared<v3::Broadcast>(key_length, dims_of(g.raw_keys, {0}));
    }
    const auto rope_for_batch = [&](const Output<Node>& rope) -> Output<Node> {
        if (!o.single_row_rope && !o.dynamic_shapes) {
            return rope;
        }
        const auto target = make_try_fold<v0::Concat>(OutputVector{dims_of(g.raw_keys, {0, 1}), i64({rotary})}, 0);
        return std::make_shared<v3::Broadcast>(rope, target);
    };

    // The block-key cache grows by zero blocks up to the complete blocks of the longest sequence.
    const auto longest_i64 = std::make_shared<v0::Convert>(g.key_length, element::i64);
    const auto longest = std::make_shared<v1::ReduceMax>(longest_i64, i64({0}), true);
    const auto needed_blocks = std::make_shared<v1::Divide>(longest, i64({key_block_size}));
    const auto missing_blocks = std::make_shared<v1::Subtract>(needed_blocks, dims_of(g.cache, {1}));
    const auto growth = std::make_shared<v1::Maximum>(missing_blocks, i64({0}));
    const auto growth_shape =
        make_try_fold<v0::Concat>(OutputVector{dims_of(g.raw_keys, {0}), i64({1}), growth, i64({head_size})}, 0);
    const auto zeros =
        std::make_shared<v3::Broadcast>(v0::Constant::create(o.precision, Shape{}, {0.0f}), growth_shape);
    const auto cache =
        std::make_shared<v0::Concat>(OutputVector{std::make_shared<v0::Unsqueeze>(g.cache, i64({1})), zeros}, 2);

    OutputVector inputs(QSAIndexerInputIdx::COUNT);
    inputs[QSAIndexerInputIdx::INDEX_QUERY] = g.raw_query;
    inputs[QSAIndexerInputIdx::RAW_INDEX_KEYS] = std::make_shared<v0::Unsqueeze>(g.raw_keys, i64({2}));
    inputs[QSAIndexerInputIdx::ROPE_COS] = rope_for_batch(g.rope_cos);
    inputs[QSAIndexerInputIdx::ROPE_SIN] = rope_for_batch(g.rope_sin);
    inputs[QSAIndexerInputIdx::KEY_LENGTH] = key_length.get_element_type() == element::i64
                                                 ? std::make_shared<v0::Convert>(key_length, element::i32)
                                                 : key_length;
    inputs[QSAIndexerInputIdx::Q_NORM_WEIGHT] = v0::Constant::create(o.precision, Shape{head_size}, {1.0f});
    inputs[QSAIndexerInputIdx::K_NORM_WEIGHT] = g.key_weight;
    inputs[QSAIndexerInputIdx::BLOCK_INDEX_KEYS_CACHE] = cache;
    inputs[QSAIndexerInputIdx::PAST_INCOMPLETE_RAW_K] = std::make_shared<v0::Unsqueeze>(g.incomplete_keys, i64({1}));
    inputs[QSAIndexerInputIdx::PAST_INCOMPLETE_ROPE] = g.incomplete_rope;

    op_internal::QSAIndexer::Config config;
    config.key_block_size = key_block_size;
    config.block_topk = token_budget / key_block_size;
    config.num_selection_heads = 1;
    config.indexer_rotary_dim = rotary;
    config.indexer_head_dim = head_size;
    config.epsilon = epsilon;
    const auto fused = std::make_shared<op_internal::QSAIndexer>(inputs, config);

    const std::vector<Output<Node>> outputs{
        fused->output(QSAIndexerOutputIdx::SELECTED_BLOCK_INDICES),
        fused->output(QSAIndexerOutputIdx::SELECTED_BLOCK_COUNT),
        std::make_shared<v0::Squeeze>(fused->output(QSAIndexerOutputIdx::PAST_INCOMPLETE_RAW_K), i64({1})),
        fused->output(QSAIndexerOutputIdx::PAST_INCOMPLETE_ROPE),
        std::make_shared<v0::Squeeze>(fused->output(QSAIndexerOutputIdx::BLOCK_INDEX_KEYS_CACHE), i64({1})),
    };
    return make_model(g, outputs, o);
}

class FuseQSAIndexerTest : public TransformationTestsF, public testing::WithParamInterface<GraphOptions> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GraphOptions>& info) {
        return info.param.name;
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
        manager.register_pass<pass::RoPEFusion>(true);
        manager.register_pass<pass::RMSFusion>(false, true);
        manager.register_pass<pass::FuseQSAIndexer>();
    }
};

class FuseQSAIndexerFusedTest : public FuseQSAIndexerTest {};

TEST_P(FuseQSAIndexerFusedTest, FusesSupportedQuerySteps) {
    model = getModel(GetParam());
    model_ref = getModelRef(GetParam());
    // QSAIndexer replaces two multi-output ops and keeps the SparseAttentionIndexer name, so the Results of the
    // incomplete RoPE state, formerly a QSAPrepare output, cannot keep the QSAPrepare name.
    disable_result_friendly_names_check();
}

GraphOptions with(std::string name, void (*setter)(GraphOptions&)) {
    GraphOptions options;
    options.name = std::move(name);
    setter(options);
    return options;
}

const std::vector<GraphOptions> fused_cases{
    with("qwen_indexer_static", [](GraphOptions&) {}),
    with("single_row_rope",
         [](GraphOptions& o) {
             o.single_row_rope = true;
         }),
    with("f16",
         [](GraphOptions& o) {
             o.precision = element::f16;
         }),
    with("dynamic_batch_and_sequence_without_symbols",
         [](GraphOptions& o) {
             o.dynamic_shapes = true;
         }),
    with("i64_key_length",
         [](GraphOptions& o) {
             o.key_length_type = element::i64;
         }),
    with("decomposed_query_norm_and_rope",
         [](GraphOptions& o) {
             o.decomposed_query = true;
         }),
    with("reshaped_rope_rows",
         [](GraphOptions& o) {
             o.reshape_rope_rows = true;
             o.query_sequence = o.keys_sequence = 1;
         }),
    with("partial_rotary_dimension",
         [](GraphOptions& o) {
             o.rotary_dim = head_size / 2;
         }),
    with("extra_state_consumers",
         [](GraphOptions& o) {
             o.extra_state_consumers = true;
         }),
    // The RMSNorm and RoPE stay for the other consumer, while QSAIndexer applies its own.
    with("extra_query_consumer",
         [](GraphOptions& o) {
             o.extra_query_consumer = true;
         }),
};

INSTANTIATE_TEST_SUITE_P(FuseQSAIndexer,
                         FuseQSAIndexerFusedTest,
                         testing::ValuesIn(fused_cases),
                         FuseQSAIndexerTest::getTestCaseName);

TEST_F(TransformationTestsF, FuseQSAIndexerLeavesQueryAndOtherNonAffineRMSNormUnchanged) {
    GraphOptions o;
    o.decomposed_query = true;
    const auto add_other_norm = [](const std::shared_ptr<Model>& m) {
        const auto x = std::make_shared<v0::Parameter>(element::f32, PartialShape{2, head_size});
        const auto norm = decomposed_rms(x, -1, epsilon);
        m->add_parameters({x});
        m->add_results({std::make_shared<v0::Result>(norm)});
    };
    model = getModel(o);
    add_other_norm(model);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<pass::FuseQSAIndexer>();
}

TEST_F(TransformationTestsF, FuseQSAIndexerFusesCanonicalQueryIdempotently) {
    GraphOptions o;
    model = getModel(o);
    model_ref = getModelRef(o);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    disable_result_friendly_names_check();
    manager.register_pass<pass::FuseQSAIndexer>();
    manager.register_pass<pass::FuseQSAIndexer>();
}

TEST_F(TransformationTestsF, FuseQSAIndexerHonorsDisabledPass) {
    model = getModel(GraphOptions{});
    manager.register_pass<pass::FuseQSAIndexer>();
    manager.get_pass_config()->disable<pass::FuseQSAIndexer>();
}

TEST_F(TransformationTestsF, FuseQSAIndexerHonorsCallback) {
    model = getModel(GraphOptions{});
    manager.register_pass<pass::FuseQSAIndexer>();
    manager.get_pass_config()->set_callback<pass::FuseQSAIndexer>([](const std::shared_ptr<const Node>&) {
        return true;
    });
}

class FuseQSAIndexerNotFusedTest : public FuseQSAIndexerTest {};

TEST_P(FuseQSAIndexerNotFusedTest, KeepsSourceOps) {
    model = getModel(GetParam());
}

// Neither source op is fused unless both query preprocessing steps match the runtime implementation.
const std::vector<GraphOptions> unsupported_query_cases{
    with("missing_query_norm",
         [](GraphOptions& o) {
             o.missing_query_norm = true;
         }),
    with("norm_after_rope",
         [](GraphOptions& o) {
             o.missing_query_norm = true;
             o.norm_after_rope = true;
         }),
    with("rope_rows_wrong_reshape",
         [](GraphOptions& o) {
             o.reshape_rope_rows = o.wrong_rope_reshape = true;
             o.query_sequence = o.keys_sequence = 1;
         }),
    with("bhld_query",
         [](GraphOptions& o) {
             o.query_layout = QueryLayout::BHLD;
         }),
    with("bhld_query_without_rope",
         [](GraphOptions& o) {
             o.query_layout = QueryLayout::BHLD;
             o.query_rope = false;
         }),
    with("rope_output_transpose",
         [](GraphOptions& o) {
             o.query_layout = QueryLayout::BLHDRoPEBeforeTranspose;
         }),
    with("explicit_query_transpose",
         [](GraphOptions& o) {
             o.query_layout = QueryLayout::BLHDTransposeAfterRoPE;
         }),
    with("converted_query",
         [](GraphOptions& o) {
             o.converted_query = true;
         }),
    with("query_rope_wrong_unsqueeze_axis",
         [](GraphOptions& o) {
             o.query_rope_axis = 0;
             o.single_row_rope = true;
         }),
    with("distinct_query_rope",
         [](GraphOptions& o) {
             o.distinct_query_rope = true;
         }),
    with("distinct_query_sin",
         [](GraphOptions& o) {
             o.distinct_query_sin = true;
         }),
    with("interleaved_query_rope",
         [](GraphOptions& o) {
             o.interleaved_query_rope = true;
         }),
    with("input_interleaved_query_rope",
         [](GraphOptions& o) {
             o.input_interleaved_query_rope = true;
         }),
    with("cached_query_rope",
         [](GraphOptions& o) {
             o.cached_query_rope = true;
         }),
    with("distinct_query_rotary_dim",
         [](GraphOptions& o) {
             o.query_rotary_dim = head_size / 2;
         }),
    with("affine_query",
         [](GraphOptions& o) {
             o.affine_query = true;
         }),
    with("rms_changes_precision",
         [](GraphOptions& o) {
             o.rms_changes_precision = true;
         }),
    with("rms_wrong_axis",
         [](GraphOptions& o) {
             o.rms_axis = -2;
         }),
    with("distinct_query_epsilon",
         [](GraphOptions& o) {
             o.query_epsilon = 2 * epsilon;
         }),
    with("nearly_equal_query_epsilon",
         [](GraphOptions& o) {
             o.query_epsilon = std::nextafter(epsilon, 2 * epsilon);
         }),
};

INSTANTIATE_TEST_SUITE_P(FuseQSAIndexerUnsupportedQuery,
                         FuseQSAIndexerNotFusedTest,
                         testing::ValuesIn(unsupported_query_cases),
                         FuseQSAIndexerTest::getTestCaseName);

const std::vector<GraphOptions> not_fused_cases{
    // QSAIndexer kernels are specialized for the head count and the head size.
    with("dynamic_query_heads",
         [](GraphOptions& o) {
             o.dynamic_query_heads = true;
         }),
    with("dynamic_head_size",
         [](GraphOptions& o) {
             o.dynamic_head_size = true;
         }),
    with("dynamic_key_length_rank",
         [](GraphOptions& o) {
             o.dynamic_key_length_rank = true;
         }),
    // QSAIndexer always applies the key RoPE and the key norm.
    with("disabled_rope",
         [](GraphOptions& o) {
             o.disable_rope = true;
         }),
    with("disabled_key_norm",
         [](GraphOptions& o) {
             o.disable_key_norm = true;
         }),
    with("distinct_key_length",
         [](GraphOptions& o) {
             o.use_distinct_key_length = true;
         }),
    with("index_weights",
         [](GraphOptions& o) {
             o.add_index_weights = true;
         }),
    // The public ops have a single key head, so QSAIndexer selects with one head.
    with("two_selection_heads",
         [](GraphOptions& o) {
             o.num_selection_heads = 2;
         }),
    with("distinct_key_block_size",
         [](GraphOptions& o) {
             o.indexer_key_block_size = 8;
         }),
    // QSAIndexer takes the raw query and the raw keys for the same tokens.
    with("query_broadcast_by_rope",
         [](GraphOptions& o) {
             o.query_sequence = 1;
         }),
    with("query_longer_than_keys",
         [](GraphOptions& o) {
             o.keys_sequence = 1;
         }),
};

INSTANTIATE_TEST_SUITE_P(FuseQSAIndexer,
                         FuseQSAIndexerNotFusedTest,
                         testing::ValuesIn(not_fused_cases),
                         FuseQSAIndexerTest::getTestCaseName);

}  // namespace
