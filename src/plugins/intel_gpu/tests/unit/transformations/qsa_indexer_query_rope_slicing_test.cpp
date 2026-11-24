// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/qsa_indexer_query_rope_slicing.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

using namespace testing;

namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
using RoPE = ov::op::internal::RoPE;

constexpr int64_t heads = 4;
constexpr int64_t head_size = 128;
constexpr int64_t rotary = 64;

struct Inputs {
    std::shared_ptr<v0::Parameter> query, cos, sin, block_keys, key_length;
    ov::ParameterVector params() const {
        return {query, cos, sin, block_keys, key_length};
    }
};

Inputs make_inputs() {
    Inputs in;
    in.query = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, heads, head_size});
    in.cos = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, rotary});
    in.sin = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, rotary});
    in.block_keys = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, head_size});
    in.key_length = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    return in;
}

RoPE::Config rope_config(bool input_trans0213) {
    RoPE::Config config;
    config.rotary_ndims = rotary;
    config.input_trans0213 = input_trans0213;
    return config;
}

ov::OutputVector with_heads(const Inputs& in) {
    const auto axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    return {std::make_shared<v0::Unsqueeze>(in.cos, axis), std::make_shared<v0::Unsqueeze>(in.sin, axis)};
}

// The head-first query of a single-token paged step: TransposeToReshape turns its Transpose into a Reshape.
std::shared_ptr<ov::Node> head_first_reshape(const Inputs& in) {
    return std::make_shared<v1::Reshape>(in.query, v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, heads, 1, head_size}), true);
}

// The partial RoPE as RoPEFusion leaves it without RoPEFusionIOSlicing: RoPE over the rotary slice, then Concat.
std::shared_ptr<ov::Node> split_rope(const Inputs& in, std::shared_ptr<ov::Node> head_first = nullptr) {
    if (!head_first) {
        head_first = std::make_shared<v1::Transpose>(in.query, v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3}));
    }
    const auto split = std::make_shared<v1::VariadicSplit>(head_first,
                                                           v0::Constant::create(ov::element::i64, ov::Shape{}, {-1}),
                                                           v0::Constant::create(ov::element::i64, ov::Shape{2}, {rotary, head_size - rotary}));
    const auto cos_sin = with_heads(in);
    const auto rope = std::make_shared<RoPE>(ov::OutputVector{split->output(0), cos_sin[0], cos_sin[1]}, rope_config(false));
    return std::make_shared<v0::Concat>(ov::OutputVector{rope, split->output(1)}, -1);
}

std::shared_ptr<ov::Node> indexer(const Inputs& in, const ov::Output<ov::Node>& query) {
    return std::make_shared<ov::op::v17::SparseAttentionIndexer>(ov::OutputVector{query, in.block_keys, in.key_length}, 4, 2048);
}

class QSAIndexerQueryRoPESlicingTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        manager.register_pass<ov::intel_gpu::QSAIndexerQueryRoPESlicing>();
    }
};

}  // namespace

// The indexer query becomes one RoPE over the whole head, taking the token-major query (input_trans0213).
TEST_F(QSAIndexerQueryRoPESlicingTest, FoldsIndexerQuery) {
    {
        const auto in = make_inputs();
        model = std::make_shared<ov::Model>(indexer(in, split_rope(in))->outputs(), in.params());
    }
    {
        const auto in = make_inputs();
        const auto cos_sin = with_heads(in);
        const auto rope = std::make_shared<RoPE>(ov::OutputVector{in.query, cos_sin[0], cos_sin[1]}, rope_config(true));
        model_ref = std::make_shared<ov::Model>(indexer(in, rope)->outputs(), in.params());
    }
}

// A head-first query that is not a Transpose is rotated as it is.
TEST_F(QSAIndexerQueryRoPESlicingTest, FoldsSingleTokenIndexerQuery) {
    {
        const auto in = make_inputs();
        model = std::make_shared<ov::Model>(indexer(in, split_rope(in, head_first_reshape(in)))->outputs(), in.params());
    }
    {
        const auto in = make_inputs();
        const auto cos_sin = with_heads(in);
        const auto rope = std::make_shared<RoPE>(ov::OutputVector{head_first_reshape(in), cos_sin[0], cos_sin[1]}, rope_config(false));
        model_ref = std::make_shared<ov::Model>(indexer(in, rope)->outputs(), in.params());
    }
}

// Any other reader of the partial RoPE would run it on the GPU RoPE kernels, which do not copy the pass-through channels.
TEST_F(QSAIndexerQueryRoPESlicingTest, KeepsRoPEWithOtherReaders) {
    const auto in = make_inputs();
    const auto rope = split_rope(in);
    auto outputs = indexer(in, rope)->outputs();
    outputs.push_back(std::make_shared<v0::Result>(rope));
    model = std::make_shared<ov::Model>(outputs, in.params());
}

TEST_F(QSAIndexerQueryRoPESlicingTest, KeepsRoPENotReadByIndexer) {
    const auto in = make_inputs();
    model = std::make_shared<ov::Model>(ov::OutputVector{split_rope(in)}, in.params());
}
