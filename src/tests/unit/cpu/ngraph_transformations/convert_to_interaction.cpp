// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph_transformations/op/interaction.hpp>
#include <ngraph_transformations/convert_to_interaction.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ie_core.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

TEST(TransformationTests, ConvertToInteractionTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        using namespace ngraph;
        //construct interaction graph
        {
            auto dense_feature = std::make_shared<ngraph::opset1::Parameter>(element::f32, PartialShape{3, 4});
            NodeVector features{dense_feature};
            std::vector<float> emb_table_value = {-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1.,
                1.5, 0.8, -0.7, -0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7};
            std::vector<int32_t> indices_value = {0, 2, 3, 4};
            std::vector<int32_t> offsets = {0, 2, 2};
            for (size_t i = 0; i < 26; i++) {
                auto emb_table = std::make_shared<opset1::Constant>(element::f32, ov::Shape{5, 4}, emb_table_value);
                auto indices = std::make_shared<opset1::Constant>(element::i32, ov::Shape{4}, indices_value);
                auto offset = std::make_shared<opset1::Constant>(element::i32, ov::Shape{3}, offsets);
                features.push_back(std::make_shared<opset8::EmbeddingBagOffsetsSum>(emb_table, indices, offset));
            }

            auto concat1 = std::make_shared<opset1::Concat>(features, 1);
            std::vector<int32_t> reshape_value = {3, 27, 4};
            auto reshape_shape =  std::make_shared<opset1::Constant>(element::i32, ov::Shape{3}, reshape_value);
            auto reshape = std::make_shared<opset1::Reshape>(concat1, reshape_shape, true);
            std::vector<int32_t> transpose1_value = {0, 2, 1};
            auto transpose1_shape =  std::make_shared<opset1::Constant>(element::i32, ov::Shape{3}, transpose1_value);
            auto transpose1 = std::make_shared<opset1::Transpose>(reshape, transpose1_shape);
            auto matmul = std::make_shared<opset1::MatMul>(reshape, transpose1);
            std::vector<int32_t> transpose2_value = {1, 2, 0};
            auto transpose2_shape =  std::make_shared<opset1::Constant>(element::i32, ov::Shape{3}, transpose2_value);
            auto transpose2 = std::make_shared<opset1::Transpose>(matmul, transpose2_shape);
            std::vector<int32_t> reshape2_value = {-1, 3};
            auto reshape2_shape =  std::make_shared<opset1::Constant>(element::i32, ov::Shape{2}, reshape2_value);
            auto reshape2 = std::make_shared<opset1::Reshape>(transpose2, reshape2_shape, true);

            std::vector<int32_t> gather_indices_value;
            for (int i = 0; i < 27; i++) {
                for (int j = i + 1; j < 27; j ++) {
                    gather_indices_value.push_back(i * 27 + j);
                }
            }
            auto gather_indices =  std::make_shared<opset1::Constant>(element::i32, ov::Shape{351}, gather_indices_value);
            auto gather_axis =  std::make_shared<opset1::Constant>(element::i32, ov::Shape{}, 0);
            auto gather = std::make_shared<opset8::Gather>(reshape2, gather_indices, gather_axis);
            std::vector<int32_t> reshape3_value = {-1, 3};
            auto reshape3_shape =  std::make_shared<opset1::Constant>(element::i32, ov::Shape{2}, reshape3_value);
            auto reshape3 = std::make_shared<opset1::Reshape>(gather, reshape3_shape, true);

            std::vector<int32_t> transpose3_value = {1, 0};
            auto transpose3_shape =  std::make_shared<opset1::Constant>(element::i32, ov::Shape{2}, transpose3_value);
            auto transpose3 = std::make_shared<opset1::Transpose>(reshape3, transpose3_shape);

            std::vector<int32_t> reshape4_value = {3, 351};
            auto reshape4_shape =  std::make_shared<opset1::Constant>(element::i32, ov::Shape{2}, reshape4_value);
            auto reshape4 = std::make_shared<opset1::Reshape>(transpose3, reshape4_shape, true);
            auto concat2 = std::make_shared<opset1::Concat>(NodeVector{dense_feature, reshape4}, 1);
            f = std::make_shared<ov::Model>(concat2, ov::ParameterVector{dense_feature}, "interaction");
            ngraph::pass::Manager m;
            m.register_pass<ngraph::pass::InitNodeInfo>();
            m.register_pass<ConvertToInteraction>();
            m.run_passes(f);
        }
        //construct ref interaction
        {
            auto dense_feature = std::make_shared<ngraph::opset1::Parameter>(element::f32, PartialShape{3, 4});
            NodeVector features{dense_feature};
            std::vector<float> emb_table_value = {-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1.,
                1.5, 0.8, -0.7, -0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7};
            std::vector<int32_t> indices_value = {0, 2, 3, 4};
            std::vector<int32_t> offsets = {0, 2, 2};
            for (size_t i = 0; i < 26; i++) {
                auto emb_table = std::make_shared<opset1::Constant>(element::f32, ov::Shape{5, 4}, emb_table_value);
                auto indices = std::make_shared<opset1::Constant>(element::i32, ov::Shape{4}, indices_value);
                auto offset = std::make_shared<opset1::Constant>(element::i32, ov::Shape{3}, offsets);
                features.push_back(std::make_shared<opset8::EmbeddingBagOffsetsSum>(emb_table, indices, offset));
            }
            auto interaction = std::make_shared<ov::intel_cpu::InteractionNode>(features);
            f_ref = std::make_shared<ov::Model>(interaction, ov::ParameterVector{dense_feature}, "interaction");
        }
        auto res = compare_functions(f, f_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
}