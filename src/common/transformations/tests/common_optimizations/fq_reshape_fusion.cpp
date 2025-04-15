// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_reshape_fusion.hpp"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/fq_mul_fusion.hpp"
#include "transformations/common_optimizations/pull_transpose_through_fq.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace testing;

namespace {

Shape DO_NOT_RESHAPE = Shape{0};

struct FQReshapeFusionTestCase {
    Shape data_shape, il_shape, ih_shape, ol_shape, oh_shape;
    std::vector<int64_t> reshape_pattern;
    Shape new_il_shape, new_ih_shape, new_ol_shape, new_oh_shape;
    bool is_negative;
};

class FQReshapeFusionTests : public ov::test::TestsCommon,
                             public testing::WithParamInterface<std::tuple<FQReshapeFusionTestCase>> {
public:
    std::shared_ptr<ov::Model> f, ref_f;

    void SetUp() override {
        const auto& test_case = std::get<0>(GetParam());
        f = get_initial_function(test_case);
        if (test_case.is_negative)
            ref_f = get_initial_function(test_case);
        else
            ref_f = get_reference_function(test_case);
    }

private:
    std::shared_ptr<ov::Model> get_initial_function(const FQReshapeFusionTestCase& test_case) {
        const auto& data = std::make_shared<opset4::Constant>(element::f32, test_case.data_shape, 0);
        auto il = std::make_shared<opset4::Parameter>(element::f32, test_case.il_shape);
        auto ih = std::make_shared<opset4::Parameter>(element::f32, test_case.ih_shape);
        auto ol = std::make_shared<opset4::Parameter>(element::f32, test_case.ol_shape);
        auto oh = std::make_shared<opset4::Parameter>(element::f32, test_case.oh_shape);

        auto fq = std::make_shared<opset4::FakeQuantize>(data, il, ih, ol, oh, 42);

        auto reshape_pattern = std::make_shared<opset4::Constant>(element::i64,
                                                                  Shape{test_case.reshape_pattern.size()},
                                                                  test_case.reshape_pattern);
        auto reshape = std::make_shared<opset4::Reshape>(fq, reshape_pattern, true);

        auto result = std::make_shared<op::v0::Result>(reshape);
        ParameterVector params = {il, ih, ol, oh};
        ResultVector results = {result};
        return std::make_shared<ov::Model>(results, params);
    }

    std::shared_ptr<ov::Model> get_reference_function(const FQReshapeFusionTestCase& test_case) {
        auto shape = PartialShape(test_case.reshape_pattern).to_shape();
        const auto& data = std::make_shared<opset4::Constant>(element::f32, shape, 0);

        const auto& p_il = std::make_shared<opset4::Parameter>(element::f32, test_case.il_shape);
        Output<Node> il = p_il;
        const auto& p_ih = std::make_shared<opset4::Parameter>(element::f32, test_case.ih_shape);
        Output<Node> ih = p_ih;
        const auto& p_ol = std::make_shared<opset4::Parameter>(element::f32, test_case.ol_shape);
        Output<Node> ol = p_ol;
        const auto& p_oh = std::make_shared<opset4::Parameter>(element::f32, test_case.oh_shape);
        Output<Node> oh = p_oh;

        if (test_case.new_il_shape != DO_NOT_RESHAPE)
            il = std::make_shared<opset4::Reshape>(
                il,
                opset4::Constant::create(element::i64, {test_case.new_il_shape.size()}, test_case.new_il_shape),
                true);
        if (test_case.new_ih_shape != DO_NOT_RESHAPE)
            ih = std::make_shared<opset4::Reshape>(
                ih,
                opset4::Constant::create(element::i64, {test_case.new_ih_shape.size()}, test_case.new_ih_shape),
                true);
        if (test_case.new_ol_shape != DO_NOT_RESHAPE)
            ol = std::make_shared<opset4::Reshape>(
                ol,
                opset4::Constant::create(element::i64, {test_case.new_ol_shape.size()}, test_case.new_ol_shape),
                true);
        if (test_case.new_oh_shape != DO_NOT_RESHAPE)
            oh = std::make_shared<opset4::Reshape>(
                oh,
                opset4::Constant::create(element::i64, {test_case.new_oh_shape.size()}, test_case.new_oh_shape),
                true);

        auto fq = std::make_shared<opset4::FakeQuantize>(data, il, ih, ol, oh, 42);

        auto result = std::make_shared<op::v0::Result>(fq);
        ParameterVector params = {p_il, p_ih, p_ol, p_oh};
        ResultVector results = {result};
        return std::make_shared<ov::Model>(results, params);
    }
};

TEST_P(FQReshapeFusionTests, ReshapeMatMul) {
    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    pass::Manager manager;
    manager.register_pass<ov::pass::InitUniqueNames>(unh);
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::FakeQuantizeReshapeFusion>();
    manager.register_pass<ov::pass::CheckUniqueNames>(unh);

    manager.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));

    auto fc =
        FunctionsComparator::no_default().enable(FunctionsComparator::PRECISIONS).enable(FunctionsComparator::NODES);
    auto res = fc.compare(f, ref_f);
    ASSERT_TRUE(res.valid) << res.message;
}

INSTANTIATE_TEST_SUITE_P(
    NGraph,
    FQReshapeFusionTests,
    testing::Values(
        // positive
        FQReshapeFusionTestCase{{1, 2, 1, 3},
                                {2, 1, 1},
                                {1},
                                {1, 1},
                                {1, 2, 1, 1},
                                {2, 3},
                                {2, 1},
                                {1, 1},
                                DO_NOT_RESHAPE,
                                {2, 1},
                                false},
        FQReshapeFusionTestCase{{1, 2, 1, 3},
                                {2, 1, 1},
                                {1},
                                {1, 1},
                                {1, 2, 1, 1},
                                {1, 2, 1, 3},
                                {1, 2, 1, 1},
                                {1, 1, 1, 1},
                                {1, 1, 1, 1},
                                DO_NOT_RESHAPE,
                                false},
        FQReshapeFusionTestCase{{2, 3},
                                {2, 1},
                                {1},
                                {1, 1},
                                {1, 1},
                                {1, 2, 1, 3},
                                {1, 2, 1, 1},
                                {1, 1, 1, 1},
                                {1, 1, 1, 1},
                                {1, 1, 1, 1},
                                false},
        // negative
        FQReshapeFusionTestCase{{1, 2, 1, 3}, {2, 1, 3}, {1}, {1, 1}, {1, 2, 1, 1}, {1, 2, 1, 3}, {}, {}, {}, {}, true},
        FQReshapeFusionTestCase{{1, 2, 1, 3}, {2, 1, 1}, {1}, {1, 1}, {1, 2, 1, 1}, {6}, {}, {}, {}, {}, true}));
}  // namespace

TEST_F(TransformationTestsF, FQReshapeGroupConvolution) {
    auto get_function = [](const FQReshapeFusionTestCase& test_case) {
        const auto& data = std::make_shared<opset4::Constant>(element::f32, test_case.data_shape, 0);
        auto il = std::make_shared<opset4::Parameter>(element::f32, test_case.il_shape);
        auto ih = std::make_shared<opset4::Parameter>(element::f32, test_case.ih_shape);
        auto ol = std::make_shared<opset4::Parameter>(element::f32, test_case.ol_shape);
        auto oh = std::make_shared<opset4::Parameter>(element::f32, test_case.oh_shape);

        auto fq = std::make_shared<opset4::FakeQuantize>(data, il, ih, ol, oh, 42);

        auto reshape_pattern = std::make_shared<opset4::Constant>(element::i64,
                                                                  Shape{test_case.reshape_pattern.size()},
                                                                  test_case.reshape_pattern);
        auto reshape = std::make_shared<opset4::Reshape>(fq, reshape_pattern, true);

        auto input = std::make_shared<opset4::Parameter>(element::f32, test_case.data_shape);
        Strides stride{1, 1};
        CoordinateDiff pad{0, 0};
        auto group_conv = std::make_shared<opset4::GroupConvolution>(input, reshape, stride, pad, pad, stride);

        auto result = std::make_shared<op::v0::Result>(group_conv);
        ParameterVector params = {il, ih, ol, oh, input};
        ResultVector results = {result};
        return std::make_shared<ov::Model>(results, params);
    };

    FQReshapeFusionTestCase params;
    params.data_shape = {1, 2, 1, 3};
    params.il_shape = {2, 1, 1};
    params.ih_shape = {1};
    params.ol_shape = {1, 1};
    params.oh_shape = {1, 2, 1, 1};
    params.reshape_pattern = {2, 3, 1, 1, 1};

    model = get_function(params);

    pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::FakeQuantizeReshapeFusion>();
}

TEST_F(TransformationTestsF, FQOptimizations) {
    {
        const auto& data = std::make_shared<opset4::Constant>(element::u8, Shape{9, 32}, 0);
        const auto& convert = std::make_shared<opset4::Convert>(data, element::f32);

        const auto& il = op::v0::Constant::create(element::f32, Shape{1}, {0});
        const auto& ih = op::v0::Constant::create(element::f32, Shape{1}, {254});
        const auto& ol = op::v0::Constant::create(element::f32, Shape{32}, {-14.22});
        const auto& oh = op::v0::Constant::create(element::f32, Shape{32}, {14.22});

        const auto& fq = std::make_shared<opset4::FakeQuantize>(convert, il, ih, ol, oh, 255);

        const auto& reshape =
            std::make_shared<opset4::Reshape>(fq,
                                              op::v0::Constant::create(element::i64, Shape{4}, {3, 3, 32, 1}),
                                              true);

        const auto& multiply =
            std::make_shared<opset4::Multiply>(reshape,
                                               op::v0::Constant::create(element::f32, Shape{1, 1, 32, 1}, {0.1140}));

        const auto& transpose =
            std::make_shared<opset4::Transpose>(multiply,
                                                op::v0::Constant::create(element::i64, Shape{4}, {2, 3, 0, 1}));

        const auto& reshape_to_weight =
            std::make_shared<opset4::Reshape>(transpose,
                                              op::v0::Constant::create(element::i64, Shape{5}, {32, 1, 1, 3, 3}),
                                              true);

        const auto& input = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(4));
        const auto& group_conv = std::make_shared<opset4::GroupConvolution>(input,
                                                                            reshape_to_weight,
                                                                            Strides{1, 1},
                                                                            CoordinateDiff{0, 0},
                                                                            CoordinateDiff{0, 0},
                                                                            Strides{1, 1});

        model = std::make_shared<ov::Model>(OutputVector{group_conv}, ParameterVector{input});

        auto fq_fusions = manager.register_pass<ov::pass::GraphRewrite>();
        fq_fusions->add_matcher<ov::pass::FakeQuantizeMulFusion>();
        fq_fusions->add_matcher<ov::pass::FakeQuantizeReshapeFusion>();
        fq_fusions->add_matcher<ov::pass::PullTransposeThroughFQUp>();
        fq_fusions->set_name("ov::pass::FakeQuantizeFusions");
    }
    {
        const auto& data = std::make_shared<opset4::Constant>(element::u8, Shape{32, 1, 3, 3}, 0);
        const auto& convert = std::make_shared<opset4::Convert>(data, element::f32);

        const auto& il = op::v0::Constant::create(element::f32, Shape{1, 1, 1, 1}, {0});
        const auto& ih = op::v0::Constant::create(element::f32, Shape{1, 1, 1, 1}, {254});
        const auto& ol = op::v0::Constant::create(element::f32, Shape{32, 1, 1, 1}, {-14.22 * 0.1140});
        const auto& oh = op::v0::Constant::create(element::f32, Shape{32, 1, 1, 1}, {14.22 * 0.1140});

        const auto& fq = std::make_shared<opset4::FakeQuantize>(convert, il, ih, ol, oh, 255);

        const auto& reshape_to_weight =
            std::make_shared<opset4::Reshape>(fq,
                                              op::v0::Constant::create(element::i64, Shape{5}, {32, 1, 1, 3, 3}),
                                              true);

        const auto& input = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(4));
        const auto& group_conv = std::make_shared<opset4::GroupConvolution>(input,
                                                                            reshape_to_weight,
                                                                            Strides{1, 1},
                                                                            CoordinateDiff{0, 0},
                                                                            CoordinateDiff{0, 0},
                                                                            Strides{1, 1});

        model_ref = std::make_shared<ov::Model>(OutputVector{group_conv}, ParameterVector{input});
    }
}
