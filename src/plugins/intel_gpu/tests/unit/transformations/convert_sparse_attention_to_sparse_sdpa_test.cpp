// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/convert_sparse_attention_to_sparse_sdpa.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/sparse_sdpa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sparse_attention.hpp"

using namespace testing;

namespace {

using SparseSDPA = ov::intel_gpu::op::SparseSDPA;
namespace v0 = ov::op::v0;

constexpr int64_t heads = 4;
constexpr int64_t kv_heads = 2;
constexpr int64_t head_size = 32;
constexpr int64_t value_size = 16;
constexpr int64_t key_block_size = 4;
constexpr int64_t topk = 8;

struct Inputs {
    std::shared_ptr<v0::Parameter> query, key, value, indices, count;
};

Inputs make_inputs(ov::element::Type index_type = ov::element::i32,
                   ov::Dimension batch = ov::Dimension::dynamic(),
                   ov::Dimension q_heads = heads,
                   int64_t k_size = head_size) {
    Inputs in;
    in.query = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{batch, q_heads, -1, k_size});
    in.key = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{batch, kv_heads, -1, k_size});
    in.value = std::make_shared<v0::Parameter>(ov::element::f16, ov::PartialShape{batch, kv_heads, -1, value_size});
    in.indices = std::make_shared<v0::Parameter>(index_type, ov::PartialShape{batch, 1, -1, topk});
    in.count = std::make_shared<v0::Parameter>(index_type, ov::PartialShape{batch, 1, -1});
    return in;
}

ov::ParameterVector parameters(const Inputs& in) {
    return {in.query, in.key, in.value, in.indices, in.count};
}

SparseSDPA::Config sdpa_config(bool causal = true) {
    SparseSDPA::Config config;
    config.num_heads = heads;
    config.num_kv_heads = kv_heads;
    config.num_selection_heads = 1;
    config.k_head_size = head_size;
    config.v_head_size = value_size;
    config.key_block_size = key_block_size;
    config.causal = causal;
    return config;
}

ov::Output<ov::Node> dim_of(const ov::Output<ov::Node>& data, int64_t axis) {
    const auto shape = std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i32);
    return std::make_shared<ov::op::v8::Gather>(shape,
                                                v0::Constant::create(ov::element::i32, ov::Shape{1}, {axis}),
                                                v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));
}

std::shared_ptr<ov::Model> make_model(const ov::OutputVector& args, const ov::ParameterVector& params, int64_t query_block_size = 1, bool causal = true) {
    const auto attention = std::make_shared<ov::op::v17::SparseAttention>(args, key_block_size, query_block_size, causal);
    return std::make_shared<ov::Model>(ov::OutputVector{attention}, params);
}

}  // namespace

class ConvertSparseAttentionToSparseSDPATest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
        manager.register_pass<ov::intel_gpu::ConvertSparseAttentionToSparseSDPA>();
    }
};

// Without key_length every row holds the whole key sequence.
TEST_F(ConvertSparseAttentionToSparseSDPATest, WithoutKeyLength) {
    {
        const auto in = make_inputs();
        model = make_model({in.query, in.key, in.value, in.indices, in.count}, parameters(in), 1, false);
    }
    {
        const auto in = make_inputs();
        const auto key_length = std::make_shared<ov::op::v3::Broadcast>(dim_of(in.key, 2), dim_of(in.query, 0));
        const auto sdpa = std::make_shared<SparseSDPA>(ov::OutputVector{in.query, in.key, in.value, in.indices, in.count, key_length}, sdpa_config(false));
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, parameters(in));
    }
}

// An i64 [1] key_length (e.g. ReduceSum of the attention mask) and i64 selections are converted to i32, and the
// key_length is broadcast over the batch.
TEST_F(ConvertSparseAttentionToSparseSDPATest, I64KeyLengthAndSelections) {
    {
        const auto in = make_inputs(ov::element::i64);
        const auto key_length = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1});
        auto params = parameters(in);
        params.push_back(key_length);
        model = make_model({in.query, in.key, in.value, in.indices, in.count, key_length}, params);
    }
    {
        const auto in = make_inputs(ov::element::i64);
        const auto key_length = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1});
        auto params = parameters(in);
        params.push_back(key_length);
        const auto i32 = [](const ov::Output<ov::Node>& value) {
            return std::make_shared<v0::Convert>(value, ov::element::i32);
        };
        const auto per_row = std::make_shared<ov::op::v3::Broadcast>(i32(key_length), dim_of(in.query, 0));
        const auto sdpa = std::make_shared<SparseSDPA>(ov::OutputVector{in.query, in.key, in.value, i32(in.indices), i32(in.count), per_row}, sdpa_config());
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, params);
    }
}

// A key_length with one value per (static) batch row and the scale are passed as they are.
TEST_F(ConvertSparseAttentionToSparseSDPATest, PerRowKeyLengthAndScale) {
    const auto make = [](bool converted) {
        const auto in = make_inputs(ov::element::i32, 2);
        const auto key_length = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{2});
        const auto scale = v0::Constant::create(ov::element::f16, ov::Shape{}, {0.125f});
        auto params = parameters(in);
        params.push_back(key_length);
        const ov::OutputVector args{in.query, in.key, in.value, in.indices, in.count, key_length, scale};
        if (!converted) {
            return make_model(args, params);
        }
        return std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<SparseSDPA>(args, sdpa_config())}, params);
    };
    model = make(false);
    model_ref = make(true);
}

// SparseSDPA only implements query_block_size == 1.
TEST_F(ConvertSparseAttentionToSparseSDPATest, KeepsQueryBlocks) {
    const auto in = make_inputs();
    model = make_model({in.query, in.key, in.value, in.indices, in.count}, parameters(in), 2);
}

// SparseSDPA kernels are specialized for the head counts and the head sizes.
TEST_F(ConvertSparseAttentionToSparseSDPATest, KeepsDynamicQueryHeads) {
    const auto in = make_inputs(ov::element::i32, ov::Dimension::dynamic(), ov::Dimension::dynamic());
    model = make_model({in.query, in.key, in.value, in.indices, in.count}, parameters(in));
}

TEST_F(ConvertSparseAttentionToSparseSDPATest, KeepsUnsupportedHeadSize) {
    const auto in = make_inputs(ov::element::i32, ov::Dimension::dynamic(), heads, 24);
    model = make_model({in.query, in.key, in.value, in.indices, in.count}, parameters(in));
}
