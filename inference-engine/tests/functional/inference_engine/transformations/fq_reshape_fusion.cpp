// Copyright (C) 2018-2021 Intel Corporation
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
#include <transformations/common_optimizations/fq_reshape_fusion.hpp>
#include <transformations/init_node_info.hpp>

#include "cnn_network_ngraph_impl.hpp"

using namespace testing;
using namespace InferenceEngine;

namespace {

ngraph::Shape DO_NOT_RESHAPE = ngraph::Shape{0};

struct FQReshapeFusionTestCase {
    ngraph::Shape data_shape, il_shape, ih_shape, ol_shape, oh_shape;
    std::vector<int64_t> reshape_pattern;
    ngraph::Shape new_il_shape, new_ih_shape, new_ol_shape, new_oh_shape;
    bool is_negative;
};

class nGraphFQReshapeFusionTests : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::tuple<FQReshapeFusionTestCase>> {
public:
    std::shared_ptr<ngraph::Function> f, ref_f;

    void SetUp() override {
        const auto& test_case = std::get<0>(GetParam());
        f = get_initial_function(test_case);
        if (test_case.is_negative)
            ref_f = get_initial_function(test_case);
        else
            ref_f = get_reference_function(test_case);
    }

private:
    std::shared_ptr<ngraph::Function> get_initial_function(const FQReshapeFusionTestCase & test_case) {
        const auto & data =  std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, test_case.data_shape, 0);
        auto il = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.il_shape);
        auto ih = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.ih_shape);
        auto ol = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.ol_shape);
        auto oh = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.oh_shape);

        auto fq = std::make_shared<ngraph::opset4::FakeQuantize>(data, il, ih, ol, oh, 42);

        auto reshape_pattern = std::make_shared<ngraph::opset4::Constant>(
                ngraph::element::i64, ngraph::Shape{test_case.reshape_pattern.size()}, test_case.reshape_pattern);
        auto reshape = std::make_shared<ngraph::opset4::Reshape>(fq, reshape_pattern, true);

        auto result = std::make_shared<ngraph::op::Result>(reshape);
        ngraph::ParameterVector params = {il, ih, ol, oh};
        ngraph::ResultVector results = {result};
        return std::make_shared<ngraph::Function>(results, params);
    }

    std::shared_ptr<ngraph::Function> get_reference_function(const FQReshapeFusionTestCase & test_case) {
        const auto & data =  std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, test_case.data_shape, 0);
        const auto & reshaped_data = std::make_shared<ngraph::opset4::Reshape>(
                data,
                std::make_shared<ngraph::opset4::Constant>(ngraph::element::i64, ngraph::Shape{test_case.reshape_pattern.size()}, test_case.reshape_pattern),
                true);

        const auto & p_il = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.il_shape);
        ngraph::Output<ngraph::Node> il = p_il;
        const auto & p_ih = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.ih_shape);
        ngraph::Output<ngraph::Node> ih = p_ih;
        const auto & p_ol = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.ol_shape);
        ngraph::Output<ngraph::Node> ol = p_ol;
        const auto & p_oh = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.oh_shape);
        ngraph::Output<ngraph::Node> oh = p_oh;

        if (test_case.new_il_shape != DO_NOT_RESHAPE)
            il = std::make_shared<ngraph::opset4::Reshape>(
                    il, ngraph::opset4::Constant::create(ngraph::element::i64, {test_case.new_il_shape.size()}, test_case.new_il_shape), true);
        if (test_case.new_ih_shape != DO_NOT_RESHAPE)
            ih = std::make_shared<ngraph::opset4::Reshape>(
                    ih, ngraph::opset4::Constant::create(ngraph::element::i64, {test_case.new_ih_shape.size()}, test_case.new_ih_shape), true);
        if (test_case.new_ol_shape != DO_NOT_RESHAPE)
            ol = std::make_shared<ngraph::opset4::Reshape>(
                    ol, ngraph::opset4::Constant::create(ngraph::element::i64, {test_case.new_ol_shape.size()}, test_case.new_ol_shape), true);
        if (test_case.new_oh_shape != DO_NOT_RESHAPE)
            oh = std::make_shared<ngraph::opset4::Reshape>(
                    oh, ngraph::opset4::Constant::create(ngraph::element::i64, {test_case.new_oh_shape.size()}, test_case.new_oh_shape), true);

        auto fq = std::make_shared<ngraph::opset4::FakeQuantize>(reshaped_data, il, ih, ol, oh, 42);

        auto result = std::make_shared<ngraph::op::Result>(fq);
        ngraph::ParameterVector params = {p_il, p_ih, p_ol, p_oh};
        ngraph::ResultVector results = {result};
        return std::make_shared<ngraph::Function>(results, params);
    }
};

TEST_P(nGraphFQReshapeFusionTests, ReshapeMatMul) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::FakeQuantizeReshapeFusion>();

    manager.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, ref_f);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_SUITE_P(NGraph, nGraphFQReshapeFusionTests, testing::Values(
    // positive
    FQReshapeFusionTestCase{{1, 2, 1, 3}, {2, 1, 1}, {1}, {1, 1}, {1, 2, 1, 1}, {2, 3}, {2, 1}, {1, 1}, DO_NOT_RESHAPE, {2, 1}, false},
    FQReshapeFusionTestCase{{1, 2, 1, 3}, {2, 1, 1}, {1}, {1, 1}, {1, 2, 1, 1}, {1, 2, 1, 3}, {1, 2, 1, 1}, {1, 1, 1, 1},  {1, 1, 1, 1}, DO_NOT_RESHAPE, false},
    FQReshapeFusionTestCase{{2, 3}, {2, 1}, {1}, {1, 1}, {1, 1}, {1, 2, 1, 3}, {1, 2, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, false},
    // negative
    FQReshapeFusionTestCase{{1, 2, 1, 3}, {2, 1, 3}, {1}, {1, 1}, {1, 2, 1, 1}, {1, 2, 1, 3}, {}, {},  {}, {}, true},
    FQReshapeFusionTestCase{{1, 2, 1, 3}, {2, 1, 1}, {1}, {1, 1}, {1, 2, 1, 1}, {6}, {}, {},  {}, {}, true}));
}  // namespace

TEST(nGraphFQReshapeFusionTests, FQReshapeGroupConvolution) {
    auto get_function = [](const FQReshapeFusionTestCase & test_case) {
        const auto & data =  std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, test_case.data_shape, 0);
        auto il = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.il_shape);
        auto ih = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.ih_shape);
        auto ol = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.ol_shape);
        auto oh = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.oh_shape);

        auto fq = std::make_shared<ngraph::opset4::FakeQuantize>(data, il, ih, ol, oh, 42);

        auto reshape_pattern = std::make_shared<ngraph::opset4::Constant>(
                ngraph::element::i64, ngraph::Shape{test_case.reshape_pattern.size()}, test_case.reshape_pattern);
        auto reshape = std::make_shared<ngraph::opset4::Reshape>(fq, reshape_pattern, true);

        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, test_case.data_shape);
        ngraph::Strides stride{1, 1};
        ngraph::CoordinateDiff pad{0, 0};
        auto group_conv = std::make_shared<ngraph::opset4::GroupConvolution>(input, reshape, stride, pad, pad, stride);

        auto result = std::make_shared<ngraph::op::Result>(group_conv);
        ngraph::ParameterVector params = {il, ih, ol, oh, input};
        ngraph::ResultVector results = {result};
        return std::make_shared<ngraph::Function>(results, params);
    };

    FQReshapeFusionTestCase params;
    params.data_shape = {1, 2, 1, 3};
    params.il_shape = {2, 1, 1};
    params.ih_shape = {1};
    params.ol_shape = {1, 1};
    params.oh_shape = {1, 2, 1, 1};
    params.reshape_pattern = {2, 3, 1, 1, 1};

    auto f = get_function(params);

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::FakeQuantizeReshapeFusion>();
    manager.run_passes(f);

    ASSERT_NO_THROW(check_rt_info(f));

    auto res = compare_functions(f, get_function(params));
    ASSERT_TRUE(res.first) << res.second;
}