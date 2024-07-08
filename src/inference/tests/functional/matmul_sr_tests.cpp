// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/smart_reshape/matmul_sr.hpp"

using namespace testing;

namespace {

using reshape_map = std::map<std::string, ov::PartialShape>;

struct ReshapeMatMulTestCase {
    bool reshape_is_A_input;
    ov::PartialShape A_shape, B_shape;
    std::vector<int64_t> reshape_pattern;
    bool transpose_a, transpose_b;
    reshape_map new_shapes;
};

class SmartReshapeMatMulTests : public ov::test::TestsCommon,
                                public testing::WithParamInterface<std::tuple<ReshapeMatMulTestCase>> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<ReshapeMatMulTestCase>> obj) {
        std::ostringstream result;
        const auto& value = std::get<0>(obj.param);
        result << "reshape_is_A_input=" << value.reshape_is_A_input << "_";
        result << "A_shape=" << value.A_shape << "_";
        result << "B_shape=" << value.B_shape << "_";
        result << "reshape_pattern=[";
        for (size_t i = 0; i < value.reshape_pattern.size(); i++) {
            if (i)
                result << ",";
            result << value.reshape_pattern[i];
        }
        result << "]_";
        result << "transpose_a=" << value.transpose_a << "_";
        result << "transpose_b=" << value.transpose_b << "_";
        result << "new_shapes={";
        for (const auto& it : value.new_shapes) {
            result << it.first << "=[";
            for (size_t i = 0; i < it.second.size(); i++) {
                if (i)
                    result << ",";
                result << it.second[i];
            }
            result << "]";
        }
        result << "}";
        return result.str();
    }

    void SetUp() override {
        const auto& test_case = std::get<0>(GetParam());

        std::shared_ptr<ov::Model> model;
        {
            auto input_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, test_case.A_shape);
            input_A->set_friendly_name("input_A");
            input_A->output(0).set_names({"input_A"});
            auto input_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, test_case.B_shape);
            input_B->set_friendly_name("input_B");
            input_B->output(0).set_names({"input_B"});

            auto reshape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                          ov::Shape{test_case.reshape_pattern.size()},
                                                                          test_case.reshape_pattern);
            reshape_pattern->set_friendly_name("reshape_pattern");
            auto reshape = std::make_shared<ov::op::v1::Reshape>(test_case.reshape_is_A_input ? input_A : input_B,
                                                                 reshape_pattern,
                                                                 true);
            reshape->set_friendly_name("reshape");

            auto mat_mul = std::make_shared<ov::op::v0::MatMul>(
                test_case.reshape_is_A_input ? reshape->output(0) : input_A->output(0),
                test_case.reshape_is_A_input ? input_B->output(0) : reshape->output(0),
                test_case.transpose_a,
                test_case.transpose_b);
            reshape->set_friendly_name("matmul");

            auto result = std::make_shared<ov::op::v0::Result>(mat_mul);
            ov::ParameterVector params = {input_A, input_B};
            ov::ResultVector results = {result};
            model = std::make_shared<ov::Model>(results, params);
        }
        OV_ASSERT_NO_THROW(model->reshape(test_case.new_shapes));
    }
};

TEST_P(SmartReshapeMatMulTests, ReshapeMatMul) {}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    OVModel,
    SmartReshapeMatMulTests,
    testing::Values(
        ReshapeMatMulTestCase{true, {1, 20, 30}, {30, 40}, {20, -1}, false, false, {{"input_A", {2, 20, 30}}}},
        ReshapeMatMulTestCase{true, {1, 20, 30}, {40, 30}, {20, -1}, false, true, {{"input_A", {2, 20, 30}}}},
        ReshapeMatMulTestCase{true, {1, 30, 20}, {30, 20}, {-1, 20}, true, false, {{"input_A", {2, 30, 20}}}},
        ReshapeMatMulTestCase{true, {1, 30, 20}, {40, 30}, {-1, 20}, true, true, {{"input_A", {2, 30, 20}}}},
        ReshapeMatMulTestCase{true, {-1, 30, 40}, {-1, 1, 1200}, {1200, 1200}, false, true, {{"input_A", {1200, 30, 40}}}},
        ReshapeMatMulTestCase{false, {20, 30}, {1, 30, 40}, {-1, 40}, false, false, {{"input_B", {2, 30, 40}}}},
        ReshapeMatMulTestCase{false, {20, 30}, {1, 40, 30}, {40, -1}, false, true, {{"input_B", {2, 40, 30}}}},
        ReshapeMatMulTestCase{false, {30, 20}, {1, 30, 40}, {-1, 40}, true, false, {{"input_B", {2, 30, 40}}}},
        ReshapeMatMulTestCase{false, {30, 20}, {1, 40, 30}, {40, -1}, true, true, {{"input_B", {2, 40, 30}}}},
        ReshapeMatMulTestCase{false, {-1, 1, 1200}, {-1, 30, 40}, {1200, 1200}, false, false, {{"input_B", {1200, 30, 40}}}}),
    SmartReshapeMatMulTests::getTestCaseName);
// clang-format on
}  // namespace

TEST(SmartReshapeTransposeMatMulTests, TransposeAMatMulFuse) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 2});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 5});
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(data_A, order);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose, data_B, false, false);
        f = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});

        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::TransposeMatMul>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 2});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 5});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_A, data_B, true, false);
        f_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(SmartReshapeTransposeMatMulTests, TransposeBMatMulFuse) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 5, 3});
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(data_B, order);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_A, transpose, false, false);
        f = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});

        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::TransposeMatMul>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 5, 3});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_A, data_B, false, true);
        f_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(SmartReshapeTransposeMatMulTests, TransposeAMatMulWithAttrFuse) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 5});
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(data_A, order);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose, data_B, true, false);
        f = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});

        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::TransposeMatMul>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 5});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_A, data_B, false, false);
        f_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(SmartReshapeTransposeMatMulTests, TransposeBMatMulWithAttrFuse) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 5});
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(data_B, order);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_A, transpose, false, true);
        f = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});

        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::TransposeMatMul>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 5});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_A, data_B, false, false);
        f_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
TEST(SmartReshapeTransposeMatMulTests, TransposeAMatMulSideAttrFuse) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 5, 3});
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(data_A, order);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose, data_B, true, true);
        f = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});

        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::TransposeMatMul>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 5, 3});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_A, data_B, false, true);
        f_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(SmartReshapeTransposeMatMulTests, TransposeBMatMulSideAttrFuse) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 2});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 5});
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(data_B, order);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_A, transpose, true, true);
        f = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});

        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::TransposeMatMul>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 2});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 5});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_A, data_B, true, false);
        f_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(SmartReshapeTransposeMatMulTests, TransposeBothMatMulFuse) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 2});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 5, 3});
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        auto transpose_A = std::make_shared<ov::op::v1::Transpose>(data_A, order);
        auto transpose_B = std::make_shared<ov::op::v1::Transpose>(data_B, order);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose_A, transpose_B, false, false);
        f = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});

        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::TransposeMatMul>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 2});
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 5, 3});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_A, data_B, true, true);
        f_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
TEST(SmartReshapeTransposeMatMulTests, TransposeBothMatMulWithAttrFuse) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 2});
        auto split_A = std::make_shared<ov::op::v1::VariadicSplit>(
            data_A,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1}));
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 5});
        auto split_B = std::make_shared<ov::op::v1::VariadicSplit>(
            data_B,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1}));
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
        auto transpose_A = std::make_shared<ov::op::v1::Transpose>(split_A->output(0), order);
        auto transpose_B = std::make_shared<ov::op::v1::Transpose>(split_B->output(1), order);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose_A, transpose_B, false, true);
        f = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});

        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::TransposeMatMul>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 2});
        auto split_A = std::make_shared<ov::op::v1::VariadicSplit>(
            data_A,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1}));
        auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 5});
        auto split_B = std::make_shared<ov::op::v1::VariadicSplit>(
            data_B,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1}));
        auto matmul = std::make_shared<ov::op::v0::MatMul>(split_A->output(0), split_B->output(1), true, false);
        f_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST_F(TransformationTestsF, SmartReshapeReshapeAMatMulSeveralConsumers) {
    // Reshape has 2 consumers: matmul and reduce.
    // Since reshape movement leads to loop creation (circular dependencies), the transformation can't be applied
    auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 2, 3});
    auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, {2}, {3, 6});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(data_A, reshape_const, false);

    auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6, 12});
    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i32, {2}, {0, 1});
    auto reduce = std::make_shared<ov::op::v1::ReduceMax>(reshape, reduction_axes);
    auto sum = std::make_shared<ov::op::v1::Add>(data_B, reduce);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(reshape, sum);
    model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});
    manager.register_pass<ov::pass::ReshapeAMatMul>();
}

TEST_F(TransformationTestsF, SmartReshapeReshapeA_1DOtherInput) {
    {
        auto input_to_reshape = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 2, 3});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, {2}, {3, 6});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input_to_reshape, reshape_const, false);

        auto other_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(reshape, other_input);
        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input_to_reshape, other_input});
        manager.register_pass<ov::pass::ReshapeAMatMul>();
    }
    {
        auto input_to_reshape = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 2, 3});
        auto other_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6});
        const auto in_C_0 = std::make_shared<ov::op::v3::ShapeOf>(other_input);
        const auto in_C_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        const auto in_C_2 = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
        const auto C = std::make_shared<ov::op::v8::Gather>(in_C_0, in_C_1, in_C_2);

        const auto N = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        const auto new_reshape_pattern = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{N, C}, 0);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input_to_reshape, new_reshape_pattern, false);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(reshape, other_input);
        model_ref =
            std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input_to_reshape, other_input});
    }
}

TEST_F(TransformationTestsF, SmartReshapeReshapeB_1DOtherInput) {
    {
        auto input_to_reshape = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 2, 3});
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, {2}, {3, 6});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input_to_reshape, reshape_const, false);

        auto other_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(other_input, reshape);
        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input_to_reshape, other_input});
        manager.register_pass<ov::pass::ReshapeBMatMul>();
    }
    {
        auto input_to_reshape = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 2, 3});
        auto other_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
        const auto in_C_0 = std::make_shared<ov::op::v3::ShapeOf>(other_input);
        const auto in_C_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        const auto in_C_2 = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
        const auto C = std::make_shared<ov::op::v8::Gather>(in_C_0, in_C_1, in_C_2);

        const auto N = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        const auto new_reshape_pattern = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{C, N}, 0);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input_to_reshape, new_reshape_pattern, false);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(other_input, reshape);
        model_ref =
            std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input_to_reshape, other_input});
    }
}

TEST_F(TransformationTestsF, SmartReshapeReshapeBMatMulSeveralConsumers) {
    // Reshape has 2 consumers: matmul and reduce.
    // Since reshape movement leads to loop creation (circular dependencies), the transformation can't be applied
    auto data_B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 2, 3});
    auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, {2}, {6, 3});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(data_B, reshape_const, false);

    auto data_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{12, 6});
    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i32, {2}, {0, 1});
    auto reduce = std::make_shared<ov::op::v1::ReduceMax>(reshape, reduction_axes);
    auto sum = std::make_shared<ov::op::v1::Add>(data_A, reduce);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(sum, reshape);
    model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data_A, data_B});
    manager.register_pass<ov::pass::ReshapeBMatMul>();
}

TEST_F(TransformationTestsF, SmartReshape_ReshapeAMatMul_ReshapeInputSeveralConsumers) {
    // Const will be reused as shared input for reshape and add operation
    // param      param        const
    //   |          |           | |
    //   |          +-----------+ |
    //   |                 |      |
    //   |              Reshape   |
    //   |                 |      |
    //   +-----------------+      |
    //             |              |
    //           MatMul           |
    //             |              |
    //             +--------------+
    //            Add
    //             |
    //           Result
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, {2}, {1, 10});
        auto data_reshape = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 5});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(data_reshape, reshape_const, false);
        auto data_matmul = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{10, 1});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(reshape, data_matmul);
        auto add = std::make_shared<ov::op::v1::Add>(matmul, reshape_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{data_matmul, data_reshape});
        ;
        manager.register_pass<ov::pass::ReshapeAMatMul>();
    }
    {
        auto reshape_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 5});
        auto shape_of_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{10, 1});
        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(shape_of_param, ov::element::i64);
        auto const_gather_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {-2});
        auto const_gather_2 = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
        auto gather = std::make_shared<ov::op::v8::Gather>(shape_of, const_gather_1, const_gather_2);
        auto const_concat_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{const_concat_1, gather}, 0);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(reshape_param, concat, false);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(reshape, shape_of_param);
        auto const_add_1 = ov::op::v0::Constant::create(ov::element::i32, {2}, {1, 10});
        auto add = std::make_shared<ov::op::v1::Add>(matmul, const_add_1);

        model_ref =
            std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{reshape_param, shape_of_param});
        ov::pass::Manager m;
        m.run_passes(model_ref);
    }
}
