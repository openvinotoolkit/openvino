// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <map>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/function.hpp>
#include <common_test_utils/ngraph_test_utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/smart_reshape/reshape_with_hc_output.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "cnn_network_ngraph_impl.hpp"

using namespace testing;
using namespace InferenceEngine;

namespace {

using reshape_map = std::map<std::string, std::vector<size_t>>;

struct ReshapeMatMulTestCase {
    bool reshape_is_A_input;
    ngraph::PartialShape A_shape, B_shape;
    std::vector<int64_t> reshape_pattern;
    bool transpose_a, transpose_b;
    reshape_map new_shapes;
};

struct ReshapeMatMulOutputValueTestCase {
    bool reshape_is_A_input;
    ngraph::PartialShape A_shape, B_shape;
    std::vector<int64_t> reshape_pattern;
    bool transpose_a, transpose_b;
    ngraph::PartialShape ref_shape;
};

class CNNNGraphImplSmartReshapeTests : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::tuple<ReshapeMatMulTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& test_case = std::get<0>(GetParam());

        std::shared_ptr<ngraph::Function> ngraph;
        {
            auto input_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.A_shape);
            input_A->set_friendly_name("input_A");
            auto input_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.B_shape);
            input_B->set_friendly_name("input_B");

            auto reshape_pattern = std::make_shared<ngraph::opset4::Constant>(
                    ngraph::element::i64, ngraph::Shape{test_case.reshape_pattern.size()}, test_case.reshape_pattern);
            reshape_pattern->set_friendly_name("reshape_pattern");
            auto reshape = std::make_shared<ngraph::opset4::Reshape>(test_case.reshape_is_A_input ? input_A : input_B, reshape_pattern, true);
            reshape->set_friendly_name("reshape");

            auto mat_mul = std::make_shared<ngraph::opset4::MatMul>(test_case.reshape_is_A_input ? reshape->output(0) : input_A->output(0),
                                                                    test_case.reshape_is_A_input ? input_B->output(0) : reshape->output(0),
                                                                    test_case.transpose_a, test_case.transpose_b);
            mat_mul->set_friendly_name("matmul");

            auto result = std::make_shared<ngraph::op::Result>(mat_mul);
            ngraph::ParameterVector params = {input_A, input_B};
            ngraph::ResultVector results = {result};
            ngraph = std::make_shared<ngraph::Function>(results, params);
        }

        InferenceEngine::details::CNNNetworkNGraphImpl network(ngraph);
        const auto & resp = network.reshape(test_case.new_shapes, nullptr);
        ASSERT_EQ(resp, StatusCode::OK);
    }
};

class CNNNGraphImplSmartReshapeOutputValueTests : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::tuple<ReshapeMatMulOutputValueTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& test_case = std::get<0>(GetParam());

        std::shared_ptr<ngraph::Function> ngraph;
        {
            auto input_A = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.A_shape);
            input_A->set_friendly_name("input_A");
            auto input_B = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.B_shape);
            input_B->set_friendly_name("input_B");

            auto reshape_pattern = std::make_shared<ngraph::opset4::Constant>(
                    ngraph::element::i64, ngraph::Shape{test_case.reshape_pattern.size()}, test_case.reshape_pattern);
            reshape_pattern->set_friendly_name("reshape_pattern");
            auto reshape = std::make_shared<ngraph::opset4::Reshape>(test_case.reshape_is_A_input ? input_A : input_B, reshape_pattern, true);
            reshape->set_friendly_name("reshape");

            auto mat_mul = std::make_shared<ngraph::opset4::MatMul>(test_case.reshape_is_A_input ? reshape->output(0) : input_A->output(0),
                                                                    test_case.reshape_is_A_input ? input_B->output(0) : reshape->output(0),
                                                                    test_case.transpose_a, test_case.transpose_b);
            mat_mul->set_friendly_name("matmul");

            auto result = std::make_shared<ngraph::op::Result>(mat_mul);
            ngraph::ParameterVector params = {input_A, input_B};
            ngraph::ResultVector results = {result};
            ngraph = std::make_shared<ngraph::Function>(results, params);

            ngraph::pass::Manager manager;
            manager.register_pass<ngraph::pass::InitNodeInfo>();
            manager.register_pass<ngraph::pass::ReshapeAMatMul>();
            manager.register_pass<ngraph::pass::ReshapeBMatMul>();
            manager.register_pass<ngraph::pass::ConstantFolding>();
            manager.run_passes(ngraph);
        }
        for (auto op : ngraph->get_ops()) {
            if (auto concat = ngraph::as_type_ptr<ngraph::opset4::Concat>(op)) {
                auto shape = concat->get_output_partial_shape(0);
                ASSERT_EQ(shape, test_case.ref_shape);
            }
        }
    }
};


TEST_P(CNNNGraphImplSmartReshapeTests, ReshapeMatMul) {
}

TEST_P(CNNNGraphImplSmartReshapeOutputValueTests, ReshapeMatmulOutputCheck){
}

INSTANTIATE_TEST_CASE_P(NGraph, CNNNGraphImplSmartReshapeTests, testing::Values(
        ReshapeMatMulTestCase{true, {1, 20, 30}, {30, 40}, {20, -1}, false, false, {{"input_A", {2, 20, 30}}}},
        ReshapeMatMulTestCase{true, {1, 20, 30}, {40, 30}, {20, -1}, false, true, {{"input_A", {2, 20, 30}}}},
        ReshapeMatMulTestCase{true, {1, 30, 20}, {30, 20}, {-1, 20}, true, false, {{"input_A", {2, 30, 20}}}},
        ReshapeMatMulTestCase{true, {1, 30, 20}, {40, 30}, {-1, 20}, true, true, {{"input_A", {2, 30, 20}}}},
        ReshapeMatMulTestCase{false, {20, 30}, {1, 30, 40}, {-1, 40}, false, false, {{"input_B", {2, 30, 40}}}},
        ReshapeMatMulTestCase{false, {20, 30}, {1, 40, 30}, {40, -1}, false, true, {{"input_B", {2, 40, 30}}}},
        ReshapeMatMulTestCase{false, {30, 20}, {1, 30, 40}, {-1, 40}, true, false, {{"input_B", {2, 30, 40}}}},
        ReshapeMatMulTestCase{false, {30, 20}, {1, 40, 30}, {40, -1}, true, true, {{"input_B", {2, 40, 30}}}},
        ReshapeMatMulTestCase{true, {1, 2, 60, 20}, {1, 20, 30}, {2, 60, 20}, false, false, {{"input_A", {2, 2, 60, 20}}}},
        ReshapeMatMulTestCase{true, {1, 2, 60, 20}, {1, 30, 20}, {2, 60, 20}, false, true, {{"input_A", {2, 2, 60, 20}}}},
        ReshapeMatMulTestCase{true, {1, 2, 20, 60}, {1, 20, 30}, {2, 20, 60}, true, false, {{"input_A", {2, 2, 20, 60}}}},
        ReshapeMatMulTestCase{true, {1, 2, 20, 60}, {1, 30, 20}, {2, 20, 60}, true, true, {{"input_A", {2, 2, 20, 60}}}},
        ReshapeMatMulTestCase{false, {1, 30, 20}, {1, 2, 20, 60}, {2, 20, 60}, false, false, {{"input_B", {2, 2, 20, 60}}}},
        ReshapeMatMulTestCase{false, {1, 30, 20}, {1, 2, 60, 20}, {2, 60, 20}, false, true, {{"input_B", {2, 2, 60, 20}}}},
        ReshapeMatMulTestCase{false, {1, 20, 30}, {1, 2, 20, 60}, {2, 20, 60}, true, false, {{"input_B", {2, 2, 20, 60}}}},
        ReshapeMatMulTestCase{false, {1, 20, 30}, {1, 2, 60, 20}, {2, 60, 20}, true, true, {{"input_B", {2, 2, 60, 20}}}},
        ReshapeMatMulTestCase{true, {1, 5, 2, 60, 20,}, {1, 2, 20, 30}, {5, 2, 60, 20}, false, false, {{"input_A", {2, 5, 2, 60, 20}}}},
        ReshapeMatMulTestCase{true, {1, 5, 2, 60, 20,}, {1, 2, 30, 20}, {5, 2, 60, 20}, false, true, {{"input_A", {2, 5, 2, 60, 20}}}},
        ReshapeMatMulTestCase{true, {1, 5, 2, 20, 60,}, {1, 2, 20, 30}, {5, 2, 20, 60}, true, false, {{"input_A", {2, 5, 2, 20, 60}}}},
        ReshapeMatMulTestCase{true, {1, 5, 2, 20, 60,}, {1, 2, 30, 20}, {5, 2, 20, 60}, true, true, {{"input_A", {2, 5, 2, 20, 60}}}},
        ReshapeMatMulTestCase{false, {1, 2, 30, 20}, {1, 5, 2, 20, 60}, {5, 2, 20, 60}, false, false, {{"input_B", {2, 5, 2, 20, 60}}}},
        ReshapeMatMulTestCase{false, {1, 2, 30, 20}, {1, 5, 2, 60, 20}, {5, 2, 60, 20}, false, true, {{"input_B", {2, 5, 2, 60, 20}}}},
        ReshapeMatMulTestCase{false, {1, 2, 20, 30}, {1, 5, 2, 20, 60}, {5, 2, 20, 60}, true, false, {{"input_B", {2, 5, 2, 20, 60}}}},
        ReshapeMatMulTestCase{false, {1, 2, 20, 30}, {1, 5, 2, 60, 20}, {5, 2, 60, 20}, true, true, {{"input_B", {2, 5, 2, 60, 20}}}}));

INSTANTIATE_TEST_CASE_P(NGraph, CNNNGraphImplSmartReshapeOutputValueTests, testing::Values(
        ReshapeMatMulOutputValueTestCase{true, {1, 20, 30}, {30, 40}, {20, -1}, false, false, {-1, 30}},
        ReshapeMatMulOutputValueTestCase{true, {1, 20, 30}, {40, 30}, {20, -1}, false, true, {-1, 30}},
        ReshapeMatMulOutputValueTestCase{true, {1, 30, 20}, {30, 20}, {-1, 20}, true, false, {30, -1}},
        ReshapeMatMulOutputValueTestCase{true, {1, 30, 20}, {40, 30}, {-1, 20}, true, true, {30, -1}},
        ReshapeMatMulOutputValueTestCase{false, {20, 30}, {1, 30, 40}, {-1, 40}, false, false, {30, -1}},
        ReshapeMatMulOutputValueTestCase{false, {20, 30}, {1, 40, 30}, {40, -1}, false, true, {-1, 30}},
        ReshapeMatMulOutputValueTestCase{false, {30, 20}, {1, 30, 40}, {-1, 40}, true, false, {30, -1}},
        ReshapeMatMulOutputValueTestCase{false, {30, 20}, {1, 40, 30}, {40, -1}, true, true, {-1, 30}},
        ReshapeMatMulOutputValueTestCase{true, {1, 2, 60, 20}, {1, 20, 30}, {2, 60, 20}, false, false, {-1, 60, 20}},
        ReshapeMatMulOutputValueTestCase{true, {1, 2, 60, 20}, {1, 30, 20}, {2, 60, 20}, false, true, {-1, 60, 20}},
        ReshapeMatMulOutputValueTestCase{true, {1, 2, 20, 60}, {1, 20, 30}, {2, 20, 60}, true, false, {-1, 20, 60}},
        ReshapeMatMulOutputValueTestCase{true, {1, 2, 20, 60}, {1, 30, 20}, {2, 20, 60}, true, true, {-1, 20, 60}},
        ReshapeMatMulOutputValueTestCase{false, {1, 30, 20}, {1, 2, 20, 60}, {2, 20, 60}, false, false, {-1, 20, 60}},
        ReshapeMatMulOutputValueTestCase{false, {1, 30, 20}, {1, 2, 60, 20}, {2, 60, 20}, false, true, {-1, 60, 20}},
        ReshapeMatMulOutputValueTestCase{false, {1, 20, 30}, {1, 2, 20, 60}, {2, 20, 60}, true, false, {-1, 20, 60}},
        ReshapeMatMulOutputValueTestCase{false, {1, 20, 30}, {1, 2, 60, 20}, {2, 60, 20}, true, true, {-1, 60, 20}},
        ReshapeMatMulOutputValueTestCase{true, {1, 5, 2, 60, 20,}, {1, 2, 20, 30}, {5, 2, 60, 20}, false, false, {-1, 2, 60, 20}},
        ReshapeMatMulOutputValueTestCase{true, {1, 5, 2, 60, 20,}, {1, 2, 30, 20}, {5, 2, 60, 20}, false, true, {-1, 2, 60, 20}},
        ReshapeMatMulOutputValueTestCase{true, {1, 5, 2, 20, 60,}, {1, 2, 20, 30}, {5, 2, 20, 60}, true, false, {-1, 2, 20, 60}},
        ReshapeMatMulOutputValueTestCase{true, {1, 5, 2, 20, 60,}, {1, 2, 30, 20}, {5, 2, 20, 60}, true, true, {-1, 2, 20, 60}},
        ReshapeMatMulOutputValueTestCase{false, {1, 2, 30, 20}, {1, 5, 2, 20, 60}, {5, 2, 20, 60}, false, false, {-1, 2, 20, 60}},
        ReshapeMatMulOutputValueTestCase{false, {1, 2, 30, 20}, {1, 5, 2, 60, 20}, {5, 2, 60, 20}, false, true, {-1, 2, 60, 20}},
        ReshapeMatMulOutputValueTestCase{false, {1, 2, 20, 30}, {1, 5, 2, 20, 60}, {5, 2, 20, 60}, true, false, {-1, 2, 20, 60}},
        ReshapeMatMulOutputValueTestCase{false, {1, 2, 20, 30}, {1, 5, 2, 60, 20}, {5, 2, 60, 20}, true, true, {-1, 2, 60, 20}}
        ));
}  // namespace