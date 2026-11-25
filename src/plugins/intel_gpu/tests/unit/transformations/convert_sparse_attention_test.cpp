// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/convert_sparse_attention.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/sparse_sdpa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/pass/manager.hpp"

using ov::intel_gpu::ConvertSparseAttentionToInternal;
using SparseSDPA = ov::intel_gpu::op::SparseSDPA;
using In = SparseSDPA::InputIdx;

namespace {

constexpr int64_t key_block_size = 4;

std::shared_ptr<ov::op::v0::Parameter> param(ov::element::Type type, const ov::PartialShape& shape) {
    return std::make_shared<ov::op::v0::Parameter>(type, shape);
}

std::shared_ptr<ov::Node> convert(const std::shared_ptr<ov::Node>& attention) {
    ov::ParameterVector params;
    for (const auto& input : attention->input_values()) {
        if (const auto parameter = ov::as_type_ptr<ov::op::v0::Parameter>(input.get_node_shared_ptr())) {
            params.push_back(parameter);
        }
    }
    const auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(attention)}, params);
    ov::pass::Manager manager;
    manager.register_pass<ConvertSparseAttentionToInternal>();
    manager.run_passes(model);
    return model->get_results()[0]->get_input_node_shared_ptr(0);
}

}  // namespace

// An i64 [1] key_length is converted and broadcast over the batch; the head geometry comes from the shapes and the scale
// is passed on.
TEST_F(TransformationTestsF, ConvertSparseAttentionToInternalBroadcastsKeyLength) {
    const auto f16 = ov::element::f16;
    const auto make_params = [&] {
        return ov::ParameterVector{param(f16, {-1, 4, -1, 32}),
                                   param(f16, {-1, 2, -1, 32}),
                                   param(f16, {-1, 2, -1, 16}),
                                   param(ov::element::i32, {-1, 1, -1, 8}),
                                   param(ov::element::i32, {-1, 1, -1}),
                                   param(ov::element::i64, {1})};
    };
    {
        const auto p = make_params();
        const auto scale = ov::op::v0::Constant::create(f16, ov::Shape{}, {0.125f});
        const auto attention = std::make_shared<ov::op::v17::SparseAttention>(p[0], p[1], p[2], p[3], p[4], p[5], scale, key_block_size);
        model = std::make_shared<ov::Model>(ov::OutputVector{attention}, p);
        manager.register_pass<ConvertSparseAttentionToInternal>();
    }
    {
        const auto p = make_params();
        const auto scale = ov::op::v0::Constant::create(f16, ov::Shape{}, {0.125f});
        const auto batch = std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(p[0], ov::element::i32),
                                                                ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0}),
                                                                ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));
        const auto key_length = std::make_shared<ov::op::v3::Broadcast>(std::make_shared<ov::op::v0::Convert>(p[5], ov::element::i32), batch);
        SparseSDPA::Config config;
        config.num_heads = 4;
        config.num_kv_heads = 2;
        config.k_head_size = 32;
        config.v_head_size = 16;
        config.key_block_size = static_cast<size_t>(key_block_size);
        const auto sdpa = std::make_shared<SparseSDPA>(ov::OutputVector{p[0], p[1], p[2], p[3], p[4], key_length, scale}, config);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, p);
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

// A per-row i32 key_length is read as it is.
TEST(ConvertSparseAttentionToInternal, KeepsPerRowKeyLength) {
    const auto f16 = ov::element::f16;
    const auto key_length = param(ov::element::i32, {2});
    const auto attention = std::make_shared<ov::op::v17::SparseAttention>(param(f16, {2, 4, -1, 32}),
                                                                          param(f16, {2, 2, -1, 32}),
                                                                          param(f16, {2, 2, -1, 32}),
                                                                          param(ov::element::i32, {2, 1, -1, 8}),
                                                                          param(ov::element::i32, {2, 1, -1}),
                                                                          key_length,
                                                                          key_block_size);

    const auto sdpa = ov::as_type_ptr<SparseSDPA>(convert(attention));
    ASSERT_NE(sdpa, nullptr);
    EXPECT_EQ(sdpa->get_input_node_shared_ptr(In::KEY_LENGTH), key_length);
    EXPECT_FALSE(sdpa->has_scale_input());
}

// Without key_length every row holds the whole key sequence, a Constant when the shapes are static.
TEST(ConvertSparseAttentionToInternal, AbsentKeyLengthOfStaticShapesIsConstant) {
    const auto f16 = ov::element::f16;
    const auto attention = std::make_shared<ov::op::v17::SparseAttention>(param(f16, {2, 4, 1, 32}),
                                                                          param(f16, {2, 2, 12, 32}),
                                                                          param(f16, {2, 2, 12, 32}),
                                                                          param(ov::element::i32, {2, 1, 1, 8}),
                                                                          param(ov::element::i32, {2, 1, 1}),
                                                                          key_block_size);

    const auto sdpa = ov::as_type_ptr<SparseSDPA>(convert(attention));
    ASSERT_NE(sdpa, nullptr);
    const auto key_length = ov::as_type_ptr<ov::op::v0::Constant>(sdpa->get_input_node_shared_ptr(In::KEY_LENGTH));
    ASSERT_NE(key_length, nullptr);
    EXPECT_EQ(key_length->get_element_type(), ov::element::i32);
    EXPECT_EQ(key_length->cast_vector<int32_t>(), (std::vector<int32_t>{12, 12}));
}

// A dynamic head count cannot be lowered: the op is kept, so that its lowering reports the reason.
TEST(ConvertSparseAttentionToInternal, KeepsDynamicHeadCount) {
    const auto f16 = ov::element::f16;
    const auto attention = std::make_shared<ov::op::v17::SparseAttention>(param(f16, {-1, -1, -1, 32}),
                                                                          param(f16, {-1, 2, -1, 32}),
                                                                          param(f16, {-1, 2, -1, 32}),
                                                                          param(ov::element::i32, {-1, 1, -1, 8}),
                                                                          param(ov::element::i32, {-1, 1, -1}),
                                                                          param(ov::element::i32, {-1}),
                                                                          key_block_size);

    EXPECT_TRUE(ov::is_type<ov::op::v17::SparseAttention>(convert(attention)));
    std::string reason;
    EXPECT_FALSE(ConvertSparseAttentionToInternal::make_config(*attention, reason).has_value());
    EXPECT_NE(reason.find("must be static"), std::string::npos) << reason;
}
