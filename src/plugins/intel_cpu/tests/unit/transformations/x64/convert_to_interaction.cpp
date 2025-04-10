// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <transformations/cpu_opset/x64/op/interaction.hpp>
#include <transformations/cpu_opset/x64/pass/convert_to_interaction.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/smart_reshape/matmul_sr.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <openvino/pass/manager.hpp>
#include "ov_ops/type_relaxed.hpp"

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;
using namespace ov;

static std::shared_ptr<ov::op::v0::FakeQuantize> createFQ(const std::shared_ptr<ov::Node>& input) {
    auto input_low = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, std::vector<float>{0});
    auto input_high = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, std::vector<float>{49.4914f});
    auto output_low = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, std::vector<float>{0});
    auto output_high = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, std::vector<float>{49.4914f});
    return std::make_shared<ov::op::v0::FakeQuantize>(input, input_low, input_high, output_low, output_high, 256);
}

static std::shared_ptr<ov::Model> makeInteraction(const ov::PartialShape& inputShape, bool intraFQ = false, bool postFQ = false) {
    std::shared_ptr<ov::opset1::Parameter> input = std::make_shared<ov::opset1::Parameter>(element::f32, inputShape);
    std::shared_ptr<ov::Node> dense_feature = nullptr;
    if (intraFQ) {
        dense_feature = createFQ(input);
    } else {
        dense_feature = input;
    }
    NodeVector features{dense_feature};
    ParameterVector inputsParams{input};
    const size_t sparse_feature_num = 26;
    for (size_t i = 0; i < sparse_feature_num; i++) {
        auto sparse_input = std::make_shared<ov::opset1::Parameter>(element::f32, inputShape);
        std::shared_ptr<ov::Node> sparse_feat = nullptr;
        if (intraFQ) {
            sparse_feat = createFQ(sparse_input);
        } else {
            sparse_feat = sparse_input;
        }
        features.push_back(sparse_feat);
        inputsParams.push_back(sparse_input);
    }
    auto shapeof = std::make_shared<ov::op::v3::ShapeOf>(dense_feature);
    auto gather_batch_indices =  std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{1}, std::vector<int32_t>{0});
    auto gather_batch_axis =  std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{}, 0);
    auto gather_batch = std::make_shared<ov::op::v8::Gather>(shapeof, gather_batch_indices, gather_batch_axis);

    auto gather_feature_indices =  std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{1}, std::vector<int32_t>{1});
    auto gather_feature_axis =  std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{1}, 0);
    auto gather_feature = std::make_shared<ov::op::v8::Gather>(shapeof, gather_feature_indices, gather_feature_axis);

    auto reshape_dim2 = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto reshape_shape = std::make_shared<ov::op::v0::Concat>(NodeVector{gather_batch, reshape_dim2, gather_feature}, 0);

    auto concat1 = std::make_shared<ov::op::v0::Concat>(features, 1);
    auto reshape = std::make_shared<ov::op::v1::Reshape>(concat1, reshape_shape, true);
    std::vector<int32_t> transpose1_value = {0, 2, 1};
    auto transpose1_shape =  std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{3}, transpose1_value);
    auto transpose1 = std::make_shared<ov::op::v1::Transpose>(reshape, transpose1_shape);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(reshape, transpose1);
    std::shared_ptr<ov::Node> inter = nullptr;
    if (intraFQ) {
        inter = createFQ(matmul);
    } else {
        inter = matmul;
    }
    std::vector<int32_t> transpose2_value = {1, 2, 0};
    auto transpose2_shape =  std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{3}, transpose2_value);
    auto transpose2 = std::make_shared<ov::op::v1::Transpose>(inter, transpose2_shape);
    std::vector<int32_t> reshape2_value = {729, -1};
    auto reshape2_shape =  std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{2}, reshape2_value);
    auto reshape2 = std::make_shared<ov::op::v1::Reshape>(transpose2, reshape2_shape, true);

    std::vector<int32_t> gather_indices_value;
    for (int i = 1; i < 27; i++) {
        for (int j = 0; j < i; j ++) {
            gather_indices_value.push_back(i * 27 + j);
        }
    }
    auto gather_indices =  std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{351}, gather_indices_value);
    auto gather_axis =  std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{}, 0);
    auto gather = std::make_shared<ov::op::v8::Gather>(reshape2, gather_indices, gather_axis);
    auto reshape3_dim1 = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto reshape3_shape = std::make_shared<ov::op::v0::Concat>(NodeVector{reshape3_dim1, gather_batch}, 0);
    auto reshape3 = std::make_shared<ov::op::v1::Reshape>(gather, reshape3_shape, true);

    std::vector<int32_t> transpose3_value = {1, 0};
    auto transpose3_shape =  std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{2}, transpose3_value);
    auto transpose3 = std::make_shared<ov::op::v1::Transpose>(reshape3, transpose3_shape);

    std::vector<int32_t> reshape4_value = {-1, 351};
    auto reshape4_shape =  std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{2}, reshape4_value);
    auto reshape4 = std::make_shared<ov::op::v1::Reshape>(transpose3, reshape4_shape, true);
    auto concat2 = std::make_shared<ov::op::v0::Concat>(NodeVector{dense_feature, reshape4}, 1);
    std::shared_ptr<ov::Model> model;
    if (postFQ) {
        auto input_low = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, std::vector<float>{-5.12978f});
        auto input_high = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, std::vector<float>{5.08965f});
        auto output_low = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, std::vector<float>{-128});
        auto output_high = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, std::vector<float>{127});
        auto fq = std::make_shared<ov::op::TypeRelaxed<ov::op::v0::FakeQuantize>>(
            ov::op::v0::FakeQuantize(concat2, input_low, input_high, output_low, output_high, 256),
            element::i8);
        model = std::make_shared<ov::Model>(fq, inputsParams, "interaction");
    } else {
        model = std::make_shared<ov::Model>(concat2, inputsParams, "interaction");
    }
    return model;
}

TEST(TransformationTests, ConvertToInteractionTest1) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        using namespace ov;
        //construct interaction graph
        auto inputShape = ov::PartialShape{3, 4};
        {
            f = makeInteraction(inputShape);
            pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<ov::pass::NopElimination>();
            m.register_pass<ov::pass::TransposeMatMul>();
            m.register_pass<ConvertToInteraction>();
            m.run_passes(f);
        }
        //construct ref interaction
        {
            auto dense_feature = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
            NodeVector features{dense_feature};
            ParameterVector inputsParams{dense_feature};
            const size_t sparse_feature_num = 26;
            for (size_t i = 0; i < sparse_feature_num; i++) {
                auto sparse_feat = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
                features.push_back(sparse_feat);
                inputsParams.push_back(sparse_feat);
            }
            auto interaction = std::make_shared<ov::intel_cpu::InteractionNode>(features);
            f_ref = std::make_shared<ov::Model>(interaction, inputsParams, "interaction");
        }
        auto res = compare_functions(f, f_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}

TEST(TransformationTests, FuseFQtoInteractionTest1) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        using namespace ov;
        //construct interaction graph
        auto inputShape = ov::PartialShape{3, 4};
        {
            f = makeInteraction(inputShape, false, true);
            pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<ov::pass::NopElimination>();
            m.register_pass<ov::pass::TransposeMatMul>();
            m.register_pass<ConvertToInteraction>();
            m.register_pass<FuseFQtoInteraction>();
            m.run_passes(f);
        }
        //construct ref interaction
        {
            auto dense_feature = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
            NodeVector features{dense_feature};
            ParameterVector inputsParams{dense_feature};
            const size_t sparse_feature_num = 26;
            for (size_t i = 0; i < sparse_feature_num; i++) {
                auto sparse_feat = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
                features.push_back(sparse_feat);
                inputsParams.push_back(sparse_feat);
            }
            auto interaction = std::make_shared<ov::op::TypeRelaxed<ov::intel_cpu::InteractionNode>>(
                ov::intel_cpu::InteractionNode(features), element::i8);
            f_ref = std::make_shared<ov::Model>(interaction, inputsParams, "interaction");
        }
        auto res = compare_functions(f, f_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}

TEST(TransformationTests, FuseFQtoInteractionTest2) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        using namespace ov;
        //construct interaction graph
        auto inputShape = ov::PartialShape{3, 4};
        {
            f = makeInteraction(inputShape, true);
            pass::Manager m;
            m.register_pass<ov::pass::InitNodeInfo>();
            m.register_pass<ov::pass::NopElimination>();
            m.register_pass<ov::pass::TransposeMatMul>();
            m.register_pass<ConvertInteractionInt8>();
            m.run_passes(f);
        }
        //construct ref interaction
        {
            auto dense_input = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
            auto dense_feature = createFQ(dense_input);
            NodeVector features{dense_feature};
            ParameterVector inputsParams{dense_input};
            const size_t sparse_feature_num = 26;
            for (size_t i = 0; i < sparse_feature_num; i++) {
                auto sparse_input = std::make_shared<ov::op::v0::Parameter>(element::f32, inputShape);
                auto sparse_feat = createFQ(sparse_input);
                features.push_back(sparse_feat);
                inputsParams.push_back(sparse_input);
            }
            auto interaction = std::make_shared<ov::intel_cpu::InteractionNode>(features);
            auto fq = createFQ(interaction);
            f_ref = std::make_shared<ov::Model>(fq, inputsParams, "interaction");
        }
        auto res = compare_functions(f, f_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}
