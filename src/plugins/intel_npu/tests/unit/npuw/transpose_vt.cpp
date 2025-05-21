// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <common_test_utils/test_common.hpp>
#include "openvino/op/ops.hpp"

namespace npuw_utest{
    using NodePtr = std::shared_ptr<ov::Node>;
}

//TODO: move to header maybe
bool optimize_value_tensors(std::shared_ptr<ov::Model> model);

enum class NetworkKind {
    llama2,
    llama3
};


typedef std::tuple <
    ov::Shape,
    bool,       // withConvert
    bool,       // withTranspose - without transpose node - matcher shouldnt detect subgraph, easy way to negative case
    NetworkKind
> OptimizeVTTestParams;

// based on ConcatWithDifferentChildrenTransformation
class TransposeVTTest : public testing::WithParamInterface<OptimizeVTTestParams>,
                                     public ov::test::TestsCommon {
public:
    void Validate() const {
        auto isValidSubgraph  = std::get<2>(GetParam());
        ASSERT_EQ(isValidSubgraph, optimize_value_tensors(model));
    }

    static std::string getTestCaseName(testing::TestParamInfo<OptimizeVTTestParams> obj) {
        auto inputShape     = std::get<0>(obj.param);
        auto withConvert    = std::get<1>(obj.param);
        auto withTranspose  = std::get<2>(obj.param);
        auto kind           = std::get<3>(obj.param);

        std::ostringstream result;
        result << "npuw_llm_pipeline_" << inputShape << "_" 
               << (kind == NetworkKind::llama3 ?  "LLAMA3" : "LLAMA2") 
               << (withConvert ? "_with_convert" : "")
               << (!withTranspose ? "_NEGATIVE" : "");
        return result.str();
    }    

protected:

    void SetUp() override {
        const auto& inputShape     = std::get<0>(GetParam());
        const auto& withConvert    = std::get<1>(GetParam());
        const auto& withTranspose  = std::get<2>(GetParam());
        const auto& kind           = std::get<3>(GetParam());

        model = CreateModel(inputShape, withConvert, withTranspose, kind);
    }

    std::shared_ptr<ov::Model> CreateModel(const ov::Shape& testShape, 
                                           bool withConvert, 
                                           bool withTranspose,
                                           NetworkKind kind) {

        auto create_shape_constant = [](const std::vector<int64_t> & const_data, const std::string& name) {
            auto pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{const_data.size()}, const_data);
            pattern->set_friendly_name("unsqueese_pattern");
            return pattern;
        };

        // in case of non broadcast number of input channels significantly smaller
        auto numChannels = (kind == NetworkKind::llama3) ? 8 : 32;
        auto input_shape = testShape;
        auto input_2 = static_cast<int>(testShape[2]);
        auto input_3 = static_cast<int>(testShape[3]);

        input_shape.at(1) = numChannels;

        // ov::Model with only a transpose node
        auto param = std::make_shared<ov::op::v0::Parameter>(withConvert ? ov::element::f16 : ov::element::f32, input_shape);
        param->set_friendly_name("past_key_value");

        std::shared_ptr<ov::Node> convert = withConvert ? 
            std::static_pointer_cast<ov::Node>(std::make_shared<ov::op::v0::Convert>(param, ov::element::f32)) : 
            std::static_pointer_cast<ov::Node>(param);
        if (withConvert) {
            convert->set_friendly_name("convert");
        }

        // todo parametrise optional reshape
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, input_shape[3] * numChannels});
        param2->set_friendly_name("new_token");

        auto reshape_pattern = create_shape_constant({0, 0, numChannels, input_3}, "reshape_pattern");
        auto transpose_pattern = create_shape_constant({1, numChannels, 1, input_3}, "transposed_pattern");

        auto reshape = std::make_shared<ov::op::v1::Reshape>(param2, withTranspose ? reshape_pattern : transpose_pattern, true);
        reshape->set_friendly_name("reshape");

        std::shared_ptr<ov::Node> transpose_or_reshape;

        if (withTranspose) {
            auto constOrder = create_shape_constant({0, 2, 1, 3}, "const_order");
            auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape, constOrder);
            transpose->set_friendly_name("transpose");
            transpose_or_reshape = transpose;
        } else {
            transpose_or_reshape = reshape;
        }

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{convert, transpose_or_reshape}, -2);
        concat->set_friendly_name("concat");

        std::shared_ptr<ov::Node> concat_or_reshape = concat;

        if (kind == NetworkKind::llama3) {
            auto unsqueeze_pattern =  create_shape_constant({2}, "unsqueese_pattern"); 
            auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(concat, unsqueeze_pattern);
            unsqueeze->set_friendly_name("unsqueeze");


            auto broadcast_pattern =  create_shape_constant({1, 8, 4, input_2 + 1, input_3}, "broadcast_pattern"); 
            //TODO: v1::Broadcast not working
            auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsqueeze, broadcast_pattern, ov::op::BroadcastType::BIDIRECTIONAL);
            broadcast->set_friendly_name("broadcast");

            auto reshape_pattern2 = create_shape_constant({0, 32, -1, input_3}, "reshape_pattern2");
            auto reshape2 = std::make_shared<ov::op::v1::Reshape>(broadcast, reshape_pattern2, true);
            reshape2->set_friendly_name("reshape2");

            concat_or_reshape = reshape2;
        }


        auto param3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32, 1, input_shape[2] + 1});
        param3->set_friendly_name("param3");

        // TODO: what if v1 softmax???
        auto softmax = std::make_shared<ov::op::v8::Softmax>(param3, -2);
        softmax->set_friendly_name("softmax");

        // entry point matmul for matcher
        auto matmul = std::make_shared<ov::op::v0::MatMul>(softmax, concat_or_reshape);
        matmul->set_friendly_name("matmul");

        auto result = std::make_shared<ov::op::v0::Result>(matmul);
        result->set_friendly_name("res");
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param, param2, param3});
    }    

    std::shared_ptr<ov::Model> model;
};


TEST_P(TransposeVTTest, smoke_Run_MatchAndTransposeVT) {
    Validate();
}
 
namespace {
// eliminate direct shape dependency to match llama2, as in test and in optimize function
const std::vector<ov::Shape> input_shapes{{1, 0, 1151, 128}, {1, 0, 1141, 64}};

const std::vector<bool> withTranspose{true, false};

const std::vector<bool> withBroadCast{true, false};

const std::vector<NetworkKind> networkKind = {
    // llama2 or llama3 type of concat, with convert layer or without
    NetworkKind::llama2,  NetworkKind::llama3
};

 INSTANTIATE_TEST_SUITE_P(smoke_Run_MatchAndTransposeVT,
                          TransposeVTTest,
                          ::testing::Combine(::testing::ValuesIn(input_shapes), 
                          ::testing::ValuesIn(withTranspose), 
                          ::testing::ValuesIn(withBroadCast),
                          ::testing::ValuesIn(networkKind)),
                          TransposeVTTest::getTestCaseName);

}  // namespace
