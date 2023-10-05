// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>

#include "cnn_network_ngraph_impl.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_common.hpp"
#include "ie_common.h"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
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
        ASSERT_NO_THROW(model->reshape(test_case.new_shapes));
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
        ASSERT_NO_THROW(check_rt_info(f));
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
        ASSERT_NO_THROW(check_rt_info(f));
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
        ASSERT_NO_THROW(check_rt_info(f));
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
        ASSERT_NO_THROW(check_rt_info(f));
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
        ASSERT_NO_THROW(check_rt_info(f));
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
        ASSERT_NO_THROW(check_rt_info(f));
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
        ASSERT_NO_THROW(check_rt_info(f));
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
        ASSERT_NO_THROW(check_rt_info(f));
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
