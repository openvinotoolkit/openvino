// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/op_conversions/convert_broadcast3.hpp>
#include <ngraph_ops/convolution_ie.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

using InputShape = PartialShape;
using TargetShape = Shape;

void convert_broadcast3_test(std::shared_ptr<Function> f, std::shared_ptr<Function> f_ref) {
    pass::Manager manager;
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertBroadcast3>();
    manager.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

class ConvertBroadcast3NUMPYTest: public CommonTestUtils::TestsCommon,
                                  public testing::WithParamInterface<std::tuple<InputShape, TargetShape>> {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& target_shape = std::get<1>(GetParam());

        f = get_initial_function(input_shape, target_shape);
        f_ref = get_reference_broadcast(input_shape, target_shape);
    }

    std::shared_ptr<Function> get_initial_function(const InputShape & input_shape,
                                                   const TargetShape & target_shape) {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto target_shape_node = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{target_shape.size()}, target_shape);
        auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(input, target_shape_node, op::BroadcastType::NUMPY);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input});
    }

    std::shared_ptr<Function> get_reference_broadcast(const InputShape & input_shape,
                                                      const TargetShape & target_shape) {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto target_shape_node = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{target_shape.size()}, target_shape);
        auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(input, target_shape_node, op::AutoBroadcastType::NUMPY);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input});
    }
};

class ConvertBroadcast3BIDIRECTMulTest: public CommonTestUtils::TestsCommon,
                                        public testing::WithParamInterface<std::tuple<InputShape, TargetShape>> {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& target_shape = std::get<1>(GetParam());

        f = get_initial_function(input_shape, target_shape);
        f_ref = get_reference_broadcast(input_shape, target_shape);
    }

    std::shared_ptr<Function> get_initial_function(const InputShape & input_shape,
                                                   const TargetShape & target_shape) {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto target_shape_node = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{target_shape.size()}, target_shape);
        auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(input, target_shape_node, op::BroadcastType::BIDIRECTIONAL);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input});
    }

    std::shared_ptr<Function> get_reference_broadcast(const InputShape & input_shape,
                                                      const TargetShape & target_shape) {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto const_node = ngraph::opset1::Constant::create(ngraph::element::f32, Shape{target_shape}, {1});
        auto mul = std::make_shared<ngraph::opset1::Multiply>(input, const_node);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});
    }
};

class ConvertBroadcast3BIDIRECTBroadcastTest: public CommonTestUtils::TestsCommon,
                                              public testing::WithParamInterface<std::tuple<InputShape, TargetShape, TargetShape>> {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& target_shape = std::get<1>(GetParam());
        const auto& aligned_target_shape = std::get<2>(GetParam());

        f = get_initial_function(input_shape, target_shape);
        f_ref = get_reference_broadcast(input_shape, aligned_target_shape);
    }

    std::shared_ptr<Function> get_initial_function(const InputShape & input_shape,
                                                   const TargetShape & target_shape) {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto target_shape_node = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{target_shape.size()}, target_shape);
        auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(input, target_shape_node, op::BroadcastType::BIDIRECTIONAL);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input});
    }

    std::shared_ptr<Function> get_reference_broadcast(const InputShape & input_shape,
                                                      const TargetShape & aligned_target_shape) {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto target_shape_node = ngraph::opset1::Constant::create(ngraph::element::i64, Shape{aligned_target_shape.size()}, aligned_target_shape);
        auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(input, target_shape_node, op::AutoBroadcastType::NUMPY);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input});
    }
};

class ConvertBroadcast3BIDIRECTBroadcastMultiplyTest: public CommonTestUtils::TestsCommon,
                                                      public testing::WithParamInterface<std::tuple<InputShape, TargetShape>> {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& target_shape = std::get<1>(GetParam());

        f = get_initial_function(input_shape, target_shape);
        f_ref = get_reference_broadcast(input_shape, target_shape);
    }

    std::shared_ptr<Function> get_initial_function(const InputShape & input_shape,
                                                   const TargetShape & target_shape) {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto target_shape_node = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i64, target_shape);
        auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(input, target_shape_node, op::BroadcastType::BIDIRECTIONAL);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input, target_shape_node});
    }

    std::shared_ptr<Function> get_reference_broadcast(const InputShape & input_shape,
                                                      const TargetShape & target_shape) {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto target_shape_node = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i64, target_shape);
        auto constant_one = opset1::Constant::create(ngraph::element::f32, {1}, {1});
        auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(constant_one, target_shape_node, op::AutoBroadcastType::NUMPY);
        auto mul = std::make_shared<ngraph::opset1::Multiply>(input, broadcast);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input, target_shape_node});
    }
};

class ConvertBroadcast3BIDIRECTBroadcastLogicalOrTest: public CommonTestUtils::TestsCommon,
                                                       public testing::WithParamInterface<std::tuple<InputShape, TargetShape>> {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& target_shape = std::get<1>(GetParam());

        f = get_initial_function(input_shape, target_shape);
        f_ref = get_reference_broadcast(input_shape, target_shape);
    }

    std::shared_ptr<Function> get_initial_function(const InputShape & input_shape,
                                                   const TargetShape & target_shape) {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::boolean, input_shape);
        auto target_shape_node = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i64, target_shape);
        auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(input, target_shape_node, op::BroadcastType::BIDIRECTIONAL);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input, target_shape_node});
    }

    std::shared_ptr<Function> get_reference_broadcast(const InputShape & input_shape,
                                                      const TargetShape & target_shape) {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::boolean, input_shape);
        auto target_shape_node = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i64, target_shape);
        auto constant_one = opset1::Constant::create(ngraph::element::boolean, {1}, {1});
        auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(constant_one, target_shape_node, op::AutoBroadcastType::NUMPY);
        auto mul = std::make_shared<ngraph::opset1::LogicalOr>(input, broadcast);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input, target_shape_node});
    }
};

TEST_P(ConvertBroadcast3NUMPYTest, CompareFunctions) {
    convert_broadcast3_test(f, f_ref);
}

TEST_P(ConvertBroadcast3BIDIRECTMulTest, CompareFunctions) {
    convert_broadcast3_test(f, f_ref);
}

TEST_P(ConvertBroadcast3BIDIRECTBroadcastTest, CompareFunctions) {
    convert_broadcast3_test(f, f_ref);
}

TEST_P(ConvertBroadcast3BIDIRECTBroadcastMultiplyTest, CompareFunctions) {
    convert_broadcast3_test(f, f_ref);
}

TEST_P(ConvertBroadcast3BIDIRECTBroadcastLogicalOrTest, CompareFunctions) {
    convert_broadcast3_test(f, f_ref);
}

INSTANTIATE_TEST_SUITE_P(ConvertBroadcast3NUMPY, ConvertBroadcast3NUMPYTest,
        testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, TargetShape{1, 2, 3, 4, 5}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64, 64},      TargetShape{8, 3, 64, 64, 64}),
                        std::make_tuple(InputShape{2, DYN, 64, 64, 64},      TargetShape{2, 3, 64, 64, 64}),
                        std::make_tuple(InputShape{3, 1, DYN, 64, 64},       TargetShape{3, 3, 3, 64, 64}),
                        std::make_tuple(InputShape{3, 3, 64, DYN, 64},       TargetShape{3, 3, 64, 64, 64}),
                        std::make_tuple(InputShape{3, 3, 64, 64, DYN},       TargetShape{3, 3, 64, 64, 3}),
                        std::make_tuple(InputShape{1, 3, 64, 64},       TargetShape{6, 3, 64, 64}),
                        std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, TargetShape{7, 3, 1, 1}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64},     TargetShape{8, 3, 64, 64}),
                        std::make_tuple(InputShape{2, DYN, 64, 64},     TargetShape{2, 3, 64, 64}),
                        std::make_tuple(InputShape{3, 3, DYN, 64},      TargetShape{3, 3, 3, 64}),
                        std::make_tuple(InputShape{3, 3, 64, DYN},      TargetShape{3, 3, 64, 4}),
                        std::make_tuple(InputShape{DYN, DYN, DYN},      TargetShape{5, 3, 1}),
                        std::make_tuple(InputShape{DYN, 3, 10},         TargetShape{3, 3, 10}),
                        std::make_tuple(InputShape{2, DYN, 9},          TargetShape{2, 3, 9}),
                        std::make_tuple(InputShape{3, 3, DYN},          TargetShape{3, 3, 3})));

INSTANTIATE_TEST_SUITE_P(ConvertBroadcast3BIDIRECT, ConvertBroadcast3BIDIRECTMulTest,
        testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, TargetShape{1, 2, 3, 4, 5}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64, 64},      TargetShape{1, 3, 64, 64, 64}),
                        std::make_tuple(InputShape{2, DYN, 64, 64, 64},      TargetShape{2, 1, 64, 64, 64}),
                        std::make_tuple(InputShape{3, 1, DYN, 64, 64},       TargetShape{3, 3, 1, 64, 64}),
                        std::make_tuple(InputShape{DYN, 1, DYN, 64, DYN},    TargetShape{3, 3, 3, 64, 1}),
                        std::make_tuple(InputShape{3, 3, 64, DYN, 64},       TargetShape{3, 3, 64, 1, 64}),
                        std::make_tuple(InputShape{3, 3, 64, 64, DYN},       TargetShape{3, 3, 64, 64, 1}),
                        std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, TargetShape{7, 3, 1, 1}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64},     TargetShape{1, 3, 64, 64}),
                        std::make_tuple(InputShape{2, DYN, 64, 64},     TargetShape{2, 1, 64, 64}),
                        std::make_tuple(InputShape{3, 3, DYN, 64},      TargetShape{3, 3, 1, 64}),
                        std::make_tuple(InputShape{DYN, 3, DYN, 64},    TargetShape{3, 3, 64}),
                        std::make_tuple(InputShape{3, 3, 64, DYN},      TargetShape{3, 3, 64, 1}),
                        std::make_tuple(InputShape{DYN, DYN, DYN},      TargetShape{5, 3, 1}),
                        std::make_tuple(InputShape{DYN, 3, 10},         TargetShape{1, 3, 10}),
                        std::make_tuple(InputShape{DYN, 3, 10},         TargetShape{10}),
                        std::make_tuple(InputShape{2, DYN, 9},          TargetShape{2, 1, 9}),
                        std::make_tuple(InputShape{3, 3, DYN},          TargetShape{3, 3, 1})));

INSTANTIATE_TEST_SUITE_P(ConvertBroadcast3BIDIRECT, ConvertBroadcast3BIDIRECTBroadcastTest,
        testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, TargetShape{2, 2, 3, 4, 5},    TargetShape{2, 2, 3, 4, 5}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64, 64},      TargetShape{3, 3, 64, 64, 64}, TargetShape{3, 3, 64, 64, 64}),
                        std::make_tuple(InputShape{2, DYN, 64, 64, 64},      TargetShape{2, 3, 64, 64, 1},  TargetShape{2, 3, 64, 64, 64}),
                        std::make_tuple(InputShape{3, 1, DYN, 64, 64},       TargetShape{1, 3, 3, 64, 64},  TargetShape{3, 3, 3, 64, 64}),
                        std::make_tuple(InputShape{3, 1, DYN, 64, DYN},      TargetShape{1, 3, 3, 64, 3},   TargetShape{3, 3, 3, 64, 3}),
                        std::make_tuple(InputShape{3, 3, 64, DYN, 64},       TargetShape{1, 1, 1, 2, 64},   TargetShape{3, 3, 64, 2, 64}),
                        std::make_tuple(InputShape{3, 3, 64, 64, DYN},       TargetShape{3, 3, 64, 64, 3},  TargetShape{3, 3, 64, 64, 3}),
                        std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, TargetShape{7, 3, 2, 3},    TargetShape{7, 3, 2, 3}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64},     TargetShape{3, 3, 64, 64},  TargetShape{3, 3, 64, 64}),
                        std::make_tuple(InputShape{2, DYN, 64, 64},     TargetShape{2, 3, 64, 64},  TargetShape{2, 3, 64, 64}),
                        std::make_tuple(InputShape{3, 3, DYN, 64},      TargetShape{1, 3, 1},       TargetShape{3, 3, 3, 64}),
                        std::make_tuple(InputShape{3, 3, DYN, 64},      TargetShape{3, 3, 64},      TargetShape{3, 3, 3, 64}),
                        std::make_tuple(InputShape{3, 3, 64, DYN},      TargetShape{64},            TargetShape{3, 3, 64, 64}),
                        std::make_tuple(InputShape{DYN, DYN, DYN},      TargetShape{5, 3, 3},       TargetShape{5, 3, 3}),
                        std::make_tuple(InputShape{1, 3, DYN},          TargetShape{3, 3, 10},      TargetShape{3, 3, 10}),
                        std::make_tuple(InputShape{2, DYN, 9},          TargetShape{2, 2, 1},       TargetShape{2, 2, 9}),
                        std::make_tuple(InputShape{3, 3, DYN},          TargetShape{3},             TargetShape{3, 3, 3})));

INSTANTIATE_TEST_SUITE_P(ConvertBroadcast3BIDIRECT, ConvertBroadcast3BIDIRECTBroadcastMultiplyTest,
        testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, TargetShape{5}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64, 64},      TargetShape{4}),
                        std::make_tuple(InputShape{2, DYN, 64, 64, 64},      TargetShape{3}),
                        std::make_tuple(InputShape{3, 1, DYN, 64, 64},       TargetShape{2}),
                        std::make_tuple(InputShape{3, 3, 64, DYN, 64},       TargetShape{1}),
                        std::make_tuple(InputShape{1, 3, 64, 64},       TargetShape{5}),
                        std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, TargetShape{4}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64},     TargetShape{3}),
                        std::make_tuple(InputShape{2, DYN, 64, 64},     TargetShape{2}),
                        std::make_tuple(InputShape{3, 3, DYN, 64},      TargetShape{1}),
                        std::make_tuple(InputShape{DYN, DYN, DYN},      TargetShape{5}),
                        std::make_tuple(InputShape{DYN, 3, 10},         TargetShape{4}),
                        std::make_tuple(InputShape{2, DYN, 9},          TargetShape{3}),
                        std::make_tuple(InputShape{3, 3, DYN},          TargetShape{2})));

INSTANTIATE_TEST_SUITE_P(ConvertBroadcast3BIDIRECT, ConvertBroadcast3BIDIRECTBroadcastLogicalOrTest,
        testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, TargetShape{5}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64, 64},      TargetShape{4}),
                        std::make_tuple(InputShape{2, DYN, 64, 64, 64},      TargetShape{3}),
                        std::make_tuple(InputShape{3, 1, DYN, 64, 64},       TargetShape{2}),
                        std::make_tuple(InputShape{3, 3, 64, DYN, 64},       TargetShape{1}),
                        std::make_tuple(InputShape{1, 3, 64, 64},       TargetShape{5}),
                        std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, TargetShape{4}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64},     TargetShape{3}),
                        std::make_tuple(InputShape{2, DYN, 64, 64},     TargetShape{2}),
                        std::make_tuple(InputShape{3, 3, DYN, 64},      TargetShape{1}),
                        std::make_tuple(InputShape{DYN, DYN, DYN},      TargetShape{5}),
                        std::make_tuple(InputShape{DYN, 3, 10},         TargetShape{4}),
                        std::make_tuple(InputShape{2, DYN, 9},          TargetShape{3}),
                        std::make_tuple(InputShape{3, 3, DYN},          TargetShape{2})));


// Broadcast-3 is converted directly to Broadcast-1 for modes NUMPY, NONE and PDPD
TEST(TransformationTests, ConvertBroadcast3WithNumpyModeToBroadcast1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto target_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(input1, target_shape, ngraph::op::BroadcastType::NUMPY);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertBroadcast3>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto target_shape = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(input1, target_shape, ngraph::op::AutoBroadcastType::NUMPY);
        broadcast->set_friendly_name("broadcast");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto broadcast_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = broadcast_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(broadcast_node->get_friendly_name() == "broadcast") << "Transformation ConvertBroadcast3 should keep output names.\n";
}

TEST(TransformationTests, ConvertBroadcast3WithPDPDModeToBroadcast1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto target_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(input1, target_shape, ngraph::op::BroadcastType::PDPD);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertBroadcast3>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto target_shape = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(input1, target_shape, ngraph::op::AutoBroadcastType::PDPD);
        broadcast->set_friendly_name("broadcast");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto broadcast_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = broadcast_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(broadcast_node->get_friendly_name() == "broadcast") << "Transformation ConvertBroadcast3 should keep output names.\n";
}

TEST(TransformationTests, ConvertBroadcast3WithExplicitModeToBroadcast1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 5, 2});
        auto brodcast_axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{0, 1, 2});
        auto target_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(input1, target_shape, brodcast_axis, ngraph::op::BroadcastType::EXPLICIT);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertBroadcast3>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 5, 2});
        auto brodcast_axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{0, 1, 2});
        auto target_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(input1, target_shape, brodcast_axis, ngraph::op::AutoBroadcastType::EXPLICIT);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto broadcast_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = broadcast_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(broadcast_node->get_friendly_name() == "broadcast") << "Transformation ConvertBroadcast3 should keep output names.\n";
}

// Broadcast-3 with mode BIDIRECTIONAL is converted to Multiply with constant with 1s of the corresponding type
TEST(TransformationTests, ConvertBroadcast3WithBidirectionalModeToBroadcast1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 2});
        auto target_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 1});
        auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(input1, target_shape, ngraph::op::BroadcastType::BIDIRECTIONAL);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConvertBroadcast3>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 2});
        auto target_shape = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(input, target_shape, ngraph::op::AutoBroadcastType::NUMPY);
        broadcast->set_friendly_name("broadcast");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto result_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = result_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(result_node->get_friendly_name() == "broadcast") << "Transformation ConvertBroadcast3 should keep output names.\n";
}
