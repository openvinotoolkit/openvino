// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/x64/pass/sparse_attention_kv_cache_fusion.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/util/variable.hpp"
#include "transformations/cpu_opset/x64/op/sparse_attention_with_kv_cache.hpp"

using namespace ov;
using ov::intel_cpu::SparseAttentionKVCacheFusion;
using ov::intel_cpu::SparseAttentionWithKVCache;

namespace {

constexpr int64_t kKeyBlockSize = 4;

struct Options {
    bool key_length = false;
    bool state_convert = false;       // Convert between ReadValue and Gather, and between Concat and Assign
    bool shape_of_on_concat = false;  // an extra ShapeOf consumer of the key Concat
    int64_t concat_axis = 2;
    bool causal = true;
    bool static_shapes = false;
    size_t layers = 1;  // independent layers, each with its own Variables
    // The options below make the model not fusable.
    bool extra_cache_reader = false;      // a Result reading the ReadValue of the key state
    int64_t query_block_size = 1;         // the fused op supports 1 only
    int64_t gather_axis = 0;              // the beam axis of the cache
    bool assign_other_variable = false;   // the key Assign writes a Variable the key state is not read from
    bool extra_gather_consumer = false;   // a Result reading the beam Gather of the key state
    bool extra_convert_consumer = false;  // a Result reading the Convert of the key state (needs state_convert)
    bool scale = false;                   // the SparseAttention form with an explicit scale
};

// The inputs and the states of one layer, shared by the original and the fused model: the query [N, 4, L, 16] attends
// to the cache of key / value [N, 2, L0 + L, 16] kept in two variables.
struct Layer {
    Layer(const Options& opt, size_t index) {
        const element::Type et = element::f32;
        const auto suffix = index == 0 ? std::string() : std::to_string(index);
        const auto n = opt.static_shapes ? 2 : -1;         // batch
        const auto len = opt.static_shapes ? 3 : -1;       // new tokens
        const auto past_len = opt.static_shapes ? 5 : -1;  // tokens in the cache
        q = std::make_shared<op::v0::Parameter>(et, PartialShape{n, 4, len, 16});
        k = std::make_shared<op::v0::Parameter>(et, PartialShape{n, 2, len, 16});
        v = std::make_shared<op::v0::Parameter>(et, PartialShape{n, 2, len, 16});
        init = std::make_shared<op::v0::Parameter>(et, PartialShape{n, 2, past_len, 16});
        beam_idx = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{n});
        block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{n, 1, len, 8});
        block_count = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{n, 1, len});
        key_length = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{n});
        const PartialShape state_shape{-1, -1, -1, 16};  // a Concat along the heads changes the second dimension
        var_k = std::make_shared<op::util::Variable>(op::util::VariableInfo{state_shape, et, "pastk" + suffix});
        var_v = std::make_shared<op::util::Variable>(op::util::VariableInfo{state_shape, et, "pastv" + suffix});
        var_k_assigned =
            opt.assign_other_variable
                ? std::make_shared<op::util::Variable>(op::util::VariableInfo{state_shape, et, "other" + suffix})
                : var_k;
        read_k = std::make_shared<op::v6::ReadValue>(init, var_k);
        read_v = std::make_shared<op::v6::ReadValue>(init, var_v);
        past_k = read_k;
        past_v = read_v;
        if (opt.state_convert) {
            past_k = std::make_shared<op::v0::Convert>(read_k, et);
            past_v = std::make_shared<op::v0::Convert>(read_v, et);
        }
    }

    // Adds the Results (the attention and the extra consumers) and the Assign sinks of the states of the layer.
    void add_outputs(const Options& opt,
                     const Output<Node>& attention,
                     const Output<Node>& present_k,
                     const Output<Node>& present_v,
                     ResultVector& results,
                     SinkVector& sinks) const {
        results.push_back(std::make_shared<op::v0::Result>(attention));
        if (opt.shape_of_on_concat) {
            results.push_back(std::make_shared<op::v0::Result>(std::make_shared<op::v3::ShapeOf>(present_k)));
        }
        if (opt.extra_cache_reader) {
            results.push_back(std::make_shared<op::v0::Result>(read_k));
        }
        Output<Node> assign_k_input = present_k;
        Output<Node> assign_v_input = present_v;
        if (opt.state_convert) {
            assign_k_input = std::make_shared<op::v0::Convert>(present_k, element::f32);
            assign_v_input = std::make_shared<op::v0::Convert>(present_v, element::f32);
        }
        sinks.push_back(std::make_shared<op::v6::Assign>(assign_k_input, var_k_assigned));
        sinks.push_back(std::make_shared<op::v6::Assign>(assign_v_input, var_v));
    }

    void add_parameters(const Options& opt, ParameterVector& params) const {
        for (const auto& param : {q, k, v, init, beam_idx, block_indices, block_count}) {
            params.push_back(param);
        }
        if (opt.key_length || opt.scale) {  // the form with a scale has the key_length input too
            params.push_back(key_length);
        }
    }

    std::shared_ptr<op::v0::Parameter> q, k, v, init, beam_idx, block_indices, block_count, key_length;
    std::shared_ptr<op::util::Variable> var_k, var_v, var_k_assigned;
    std::shared_ptr<Node> read_k, read_v, past_k, past_v;
};

std::shared_ptr<Node> make_sparse_attention(const Options& opt,
                                            const Layer& in,
                                            const Output<Node>& concat_k,
                                            const Output<Node>& concat_v) {
    if (opt.scale) {
        // the form with a scale needs the key_length input
        auto scale = op::v0::Constant::create(element::f32, {}, {0.25F});
        return std::make_shared<op::v17::SparseAttention>(in.q,
                                                          concat_k,
                                                          concat_v,
                                                          in.block_indices,
                                                          in.block_count,
                                                          in.key_length,
                                                          scale,
                                                          kKeyBlockSize,
                                                          opt.query_block_size,
                                                          opt.causal);
    }
    if (opt.key_length) {
        return std::make_shared<op::v17::SparseAttention>(in.q,
                                                          concat_k,
                                                          concat_v,
                                                          in.block_indices,
                                                          in.block_count,
                                                          in.key_length,
                                                          kKeyBlockSize,
                                                          opt.query_block_size,
                                                          opt.causal);
    }
    return std::make_shared<op::v17::SparseAttention>(in.q,
                                                      concat_k,
                                                      concat_v,
                                                      in.block_indices,
                                                      in.block_count,
                                                      kKeyBlockSize,
                                                      opt.query_block_size,
                                                      opt.causal);
}

// ReadValue -> [Convert] -> Gather(beam_idx) -> Concat(cur) -> SparseAttention
std::shared_ptr<ov::Model> getModel(const Options& opt) {
    ResultVector results;
    SinkVector sinks;
    ParameterVector params;
    for (size_t index = 0; index < opt.layers; index++) {
        Layer in(opt, index);
        auto axis = op::v0::Constant::create(element::i32, {1}, {opt.gather_axis});
        auto gather_k = std::make_shared<op::v8::Gather>(in.past_k, in.beam_idx, axis);
        auto gather_v = std::make_shared<op::v8::Gather>(in.past_v, in.beam_idx, axis);
        auto concat_k = std::make_shared<op::v0::Concat>(OutputVector{gather_k, in.k}, opt.concat_axis);
        auto concat_v = std::make_shared<op::v0::Concat>(OutputVector{gather_v, in.v}, opt.concat_axis);
        auto sparse = make_sparse_attention(opt, in, concat_k, concat_v);
        in.add_outputs(opt, sparse->output(0), concat_k, concat_v, results, sinks);
        if (opt.extra_gather_consumer) {
            results.push_back(std::make_shared<op::v0::Result>(gather_k));
        }
        if (opt.extra_convert_consumer) {
            results.push_back(std::make_shared<op::v0::Result>(in.past_k));
        }
        in.add_parameters(opt, params);
    }
    return std::make_shared<ov::Model>(results, sinks, params, "SparseKV");
}

// ReadValue -> [Convert] -> SparseAttentionWithKVCache; the fusion keeps the init subgraphs out of the fused states
// (MoveReadValueInputsToSubgraph honours the mark).
std::shared_ptr<ov::Model> getModelRef(const Options& opt) {
    ResultVector results;
    SinkVector sinks;
    ParameterVector params;
    for (size_t index = 0; index < opt.layers; index++) {
        Layer in(opt, index);
        in.read_k->get_rt_info()["DisableInitSubgraphFusing"] = true;
        in.read_v->get_rt_info()["DisableInitSubgraphFusing"] = true;
        OutputVector args{in.q, in.k, in.v, in.block_indices, in.block_count};
        if (opt.key_length) {
            args.push_back(in.key_length);
        }
        args.push_back(in.beam_idx);
        args.push_back(in.past_k);
        args.push_back(in.past_v);
        SparseAttentionWithKVCache::Config config;
        config.key_block_size = kKeyBlockSize;
        config.query_block_size = opt.query_block_size;
        config.causal = opt.causal;
        auto fused = std::make_shared<SparseAttentionWithKVCache>(args, config);
        in.add_outputs(opt, fused->output(0), fused->output(1), fused->output(2), results, sinks);
        in.add_parameters(opt, params);
    }
    return std::make_shared<ov::Model>(results, sinks, params, "SparseKV");
}

}  // namespace

class SparseAttentionKVCacheFusionTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
        // the graph comparator casts the Assign sinks through RTTI, which does not work across libraries on Android
        test_skipped = true;
        GTEST_SKIP() << "Skipping SparseAttentionKVCacheFusion tests on Android X64";
#endif
        // the attributes of the fused op: key_block_size, query_block_size, causal
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        manager.register_pass<SparseAttentionKVCacheFusion>();
    }
};

using FusionParams = std::tuple<bool,     // SparseAttention with key_length
                                bool,     // Convert between the states and the Gather / Assign
                                bool,     // ShapeOf consumer of the key Concat
                                int64_t,  // Concat axis: 2 or -2, as PyTorch exports write it
                                bool,     // causal
                                bool>;    // static shapes

class SparseAttentionKVCacheFusionParamTest : public SparseAttentionKVCacheFusionTest,
                                              public testing::WithParamInterface<FusionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FusionParams>& obj) {
        const auto& [key_length, state_convert, shape_of, concat_axis, causal, static_shapes] = obj.param;
        std::ostringstream result;
        result << "KeyLength=" << key_length << "_StateConvert=" << state_convert << "_ShapeOf=" << shape_of
               << "_ConcatAxis=" << concat_axis << "_Causal=" << causal << "_Static=" << static_shapes;
        return result.str();
    }

protected:
    void SetUp() override {
        SparseAttentionKVCacheFusionTest::SetUp();
        if (IsSkipped()) {
            return;
        }
        const auto& [key_length, state_convert, shape_of, concat_axis, causal, static_shapes] = GetParam();
        Options opt;
        opt.key_length = key_length;
        opt.state_convert = state_convert;
        opt.shape_of_on_concat = shape_of;
        opt.concat_axis = concat_axis;
        opt.causal = causal;
        opt.static_shapes = static_shapes;
        model = getModel(opt);
        model_ref = getModelRef(opt);
    }
};

TEST_P(SparseAttentionKVCacheFusionParamTest, Fuses) {}

INSTANTIATE_TEST_SUITE_P(smoke,
                         SparseAttentionKVCacheFusionParamTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(2, -2),
                                            ::testing::Bool(),
                                            ::testing::Bool()),
                         SparseAttentionKVCacheFusionParamTest::getTestCaseName);

// Each layer has its own states: all of them are fused.
TEST_F(SparseAttentionKVCacheFusionTest, FusesEveryLayer) {
    Options opt;
    opt.layers = 2;
    model = getModel(opt);
    model_ref = getModelRef(opt);
}

TEST_F(SparseAttentionKVCacheFusionTest, NoFusionWhenCacheHasOtherReader) {
    Options opt;
    opt.extra_cache_reader = true;
    model = getModel(opt);
}

TEST_F(SparseAttentionKVCacheFusionTest, NoFusionForQueryBlockSizeAboveOne) {
    Options opt;
    opt.query_block_size = 2;
    model = getModel(opt);
}

TEST_F(SparseAttentionKVCacheFusionTest, NoFusionWhenBeamAxisIsNotBatch) {
    Options opt;
    opt.gather_axis = 1;
    model = getModel(opt);
}

TEST_F(SparseAttentionKVCacheFusionTest, NoFusionWhenConcatAxisIsNotSequence) {
    Options opt;
    opt.concat_axis = 1;
    model = getModel(opt);
}

TEST_F(SparseAttentionKVCacheFusionTest, NoFusionWhenAssignWritesAnotherVariable) {
    Options opt;
    opt.assign_other_variable = true;
    model = getModel(opt);
}

TEST_F(SparseAttentionKVCacheFusionTest, NoFusionWhenBeamGatherHasOtherConsumer) {
    Options opt;
    opt.extra_gather_consumer = true;
    model = getModel(opt);
}

TEST_F(SparseAttentionKVCacheFusionTest, NoFusionWhenStateConvertHasOtherConsumer) {
    Options opt;
    opt.state_convert = true;
    opt.extra_convert_consumer = true;
    model = getModel(opt);
}

TEST_F(SparseAttentionKVCacheFusionTest, NoFusionForSparseAttentionWithScale) {
    Options opt;
    opt.scale = true;
    model = getModel(opt);
}
