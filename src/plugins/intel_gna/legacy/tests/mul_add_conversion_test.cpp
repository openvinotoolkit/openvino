// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <legacy/ngraph_ops/eltwise.hpp>
#include <legacy/ngraph_ops/power.hpp>
#include <legacy/ngraph_ops/scaleshift.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_mul_add_to_scaleshift_or_power.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_mul_or_add_finally.hpp>
#include <map>
#include <memory>
#include <ngraph/pass/visualize_tree.hpp>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/manager.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "ov_lpt_models/common/builders.hpp"

using namespace testing;

using InputShape = ov::PartialShape;
struct ConstantParams {
    ov::Shape shape;
    float value;
    bool skip;
    ConstantParams() : skip(true) {}
    ConstantParams(const ov::Shape& shape, float value) : shape(shape), value(value), skip(false) {}
};
using MulConstant = ConstantParams;
using AddConstant = ConstantParams;
using IsDequantization = bool;
using RefFunction = std::function<
    std::shared_ptr<ov::Model>(const InputShape&, const MulConstant&, const AddConstant&, const IsDequantization&)>;

class MulAddConversionTests
    : public ov::test::TestsCommon,
      public testing::WithParamInterface<
          std::tuple<std::tuple<InputShape, MulConstant, AddConstant, IsDequantization>, RefFunction>> {
public:
    std::shared_ptr<ov::Model> f, f_ref;

    void SetUp() override {
        const auto& attrs = std::get<0>(GetParam());
        const auto& input_shape = std::get<0>(attrs);
        const auto& mul_const = std::get<1>(attrs);
        const auto& add_const = std::get<2>(attrs);
        const auto& is_dequantization = std::get<3>(attrs);
        const auto& get_ref_model = std::get<1>(GetParam());

        f = get_initial_model(input_shape, mul_const, add_const, is_dequantization);
        f_ref = get_ref_model(input_shape, mul_const, add_const, is_dequantization);
    }

    static std::shared_ptr<ov::Model> get_initial_model(const InputShape& input_shape,
                                                        const MulConstant& mul_const,
                                                        const AddConstant& add_const,
                                                        const IsDequantization& is_dequantization) {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
        ov::Output<ov::Node> last = input;
        if (!mul_const.skip) {
            last = std::make_shared<ov::opset1::Multiply>(last, create_constant(mul_const.shape, mul_const.value));
        }
        if (!add_const.skip) {
            last = std::make_shared<ov::opset1::Add>(last, create_constant(add_const.shape, add_const.value));
        }
        last = std::make_shared<ov::opset1::Relu>(last);
        return std::make_shared<ov::Model>(ov::NodeVector{last.get_node_shared_ptr()}, ov::ParameterVector{input});
    }

    static std::shared_ptr<ov::Model> get_scale_shift_reference(const InputShape& input_shape,
                                                                const MulConstant& mul_const,
                                                                const AddConstant& add_const,
                                                                const IsDequantization& is_dequanization) {
        if (mul_const.skip && add_const.skip) {
            OPENVINO_THROW("Invalid arguments");
        }

        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
        auto scsh = std::make_shared<ngraph::op::ScaleShiftIE>(
            input,
            (!mul_const.skip ? create_constant(mul_const.shape, mul_const.value) : create_constant(add_const.shape, 1)),
            (!add_const.skip ? create_constant(add_const.shape, add_const.value)
                             : create_constant(mul_const.shape, 0)));
        auto relu = std::make_shared<ov::opset1::Relu>(scsh);
        return std::make_shared<ov::Model>(ov::NodeVector{relu}, ov::ParameterVector{input});
    }

    static std::shared_ptr<ov::Model> get_power_reference(const InputShape& input_shape,
                                                          const MulConstant& mul_const,
                                                          const AddConstant& add_const,
                                                          const IsDequantization& is_dequanization) {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
        float scale(1), shift(0);
        if (!mul_const.skip)
            scale = mul_const.value;
        if (!add_const.skip)
            shift = add_const.value;
        auto pow = std::make_shared<ngraph::op::PowerIE>(input, 1.f, scale, shift);
        auto relu = std::make_shared<ov::opset1::Relu>(pow);
        return std::make_shared<ov::Model>(ov::NodeVector{relu}, ov::ParameterVector{input});
    }

    static std::shared_ptr<ov::Model> get_eltwise_add_reference(const InputShape& input_shape,
                                                                const MulConstant& mul_const,
                                                                const AddConstant& add_const,
                                                                const IsDequantization& is_dequanization) {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
        auto add = std::make_shared<ngraph::op::Eltwise>(input,
                                                         create_constant(add_const.shape, add_const.value),
                                                         ELTWISE_TYPE::Sum);
        auto relu = std::make_shared<ov::opset1::Relu>(add);
        return std::make_shared<ov::Model>(ov::NodeVector{relu}, ov::ParameterVector{input});
    }

    static std::shared_ptr<ov::Model> get_eltwise_mul_reference(const InputShape& input_shape,
                                                                const MulConstant& mul_const,
                                                                const AddConstant& add_const,
                                                                const IsDequantization& is_dequanization) {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
        auto mul = std::make_shared<ngraph::op::Eltwise>(input,
                                                         create_constant(mul_const.shape, mul_const.value),
                                                         ELTWISE_TYPE::Prod);
        auto relu = std::make_shared<ov::opset1::Relu>(mul);
        return std::make_shared<ov::Model>(ov::NodeVector{relu}, ov::ParameterVector{input});
    }

    static std::shared_ptr<ov::opset1::Constant> create_constant(const ov::Shape& shape, float init_value) {
        return ov::opset1::Constant::create(ov::element::f32, shape, {init_value});
    }
};

class MulOrAddConversionTests : public MulAddConversionTests {};

TEST_P(MulAddConversionTests, CompareFunctions) {
    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitUniqueNames>(unh);
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::ConvertMulAddToScaleShiftOrPower>();
    manager.register_pass<ov::pass::CheckUniqueNames>(unh);
    manager.register_pass<ov::pass::InjectionPass>([](std::shared_ptr<ov::Model> f) {
        check_rt_info(f);
    });
    manager.register_pass<ov::pass::ConstantFolding>();
    ASSERT_NO_THROW(manager.run_passes(f));
    f->validate_nodes_and_infer_types();

    auto fc =
        FunctionsComparator::no_default().enable(FunctionsComparator::NODES).enable(FunctionsComparator::PRECISIONS);
    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST_P(MulOrAddConversionTests, CompareFunctions) {
    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitUniqueNames>(unh);
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::ConvertMulOrAddFinally>();
    manager.register_pass<ov::pass::CheckUniqueNames>(unh);
    manager.register_pass<ov::pass::InjectionPass>([](std::shared_ptr<ov::Model> f) {
        check_rt_info(f);
    });
    manager.register_pass<ov::pass::ConstantFolding>();
    ASSERT_NO_THROW(manager.run_passes(f));

    f->validate_nodes_and_infer_types();

    auto fc =
        FunctionsComparator::no_default().enable(FunctionsComparator::NODES).enable(FunctionsComparator::PRECISIONS);
    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

#define CONST(A, B)  ConstantParams(A, B)
#define NONE         ConstantParams()
#define SCALESHIFT   MulAddConversionTests::get_scale_shift_reference
#define POWER        MulAddConversionTests::get_power_reference
#define SAME         MulAddConversionTests::get_initial_model
#define ELTWISE_SUM  MulAddConversionTests::get_eltwise_add_reference
#define ELTWISE_PROD MulAddConversionTests::get_eltwise_mul_reference

INSTANTIATE_TEST_SUITE_P(MulAddToScaleShift,
                         MulAddConversionTests,
                         testing::Combine(testing::Values(std::make_tuple(InputShape{DYN, 3, 64, 64},
                                                                          CONST(ov::Shape({1, 3, 1, 1}), 0.5),
                                                                          CONST(ov::Shape({1, 3, 1, 1}), 0.5),
                                                                          false),
                                                          std::make_tuple(InputShape{DYN, 3, DYN, 64},
                                                                          CONST(ov::Shape({1, 3, 1, 1}), 0.5),
                                                                          CONST(ov::Shape({1, 3, 1, 1}), 0.5),
                                                                          false),
                                                          std::make_tuple(InputShape{DYN, 3, DYN, DYN},
                                                                          CONST(ov::Shape({1, 3, 1, 1}), 0.5),
                                                                          CONST(ov::Shape({1, 3, 1, 1}), 0.5),
                                                                          false)),
                                          testing::Values(SCALESHIFT)));

INSTANTIATE_TEST_SUITE_P(
    MulToScaleShift,
    MulOrAddConversionTests,
    testing::Combine(
        testing::Values(
            std::make_tuple(InputShape{DYN, 3, 64, 64}, CONST(ov::Shape({1, 3, 1, 1}), 0.5), NONE, false),
            std::make_tuple(InputShape{DYN, 3, DYN, 64}, CONST(ov::Shape({1, 3, 1, 1}), 0.5), NONE, false),
            std::make_tuple(InputShape{DYN, 3, DYN, DYN}, CONST(ov::Shape({1, 3, 1, 1}), 0.5), NONE, false)),
        testing::Values(SCALESHIFT)));

INSTANTIATE_TEST_SUITE_P(
    AddToScaleShift,
    MulOrAddConversionTests,
    testing::Combine(
        testing::Values(
            std::make_tuple(InputShape{DYN, 3, 64, 64}, NONE, CONST(ov::Shape({1, 3, 1, 1}), 0.5), false),
            std::make_tuple(InputShape{DYN, 3, DYN, 64}, NONE, CONST(ov::Shape({1, 3, 1, 1}), 0.5), false),
            std::make_tuple(InputShape{DYN, 3, DYN, DYN}, NONE, CONST(ov::Shape({1, 3, 1, 1}), 0.5), false)),
        testing::Values(SCALESHIFT)));

INSTANTIATE_TEST_SUITE_P(
    MulAddToPower,
    MulAddConversionTests,
    testing::Combine(
        testing::Values(
            std::make_tuple(InputShape{DYN, 3, 64, 64}, CONST(ov::Shape({1}), 0.5), CONST(ov::Shape({1}), 0.5), false),
            std::make_tuple(InputShape{DYN, 3, DYN, 64}, CONST(ov::Shape({1}), 0.5), CONST(ov::Shape({1}), 0.5), false),
            std::make_tuple(InputShape{DYN, 3, DYN, DYN},
                            CONST(ov::Shape({1}), 0.5),
                            CONST(ov::Shape({1}), 0.5),
                            false)),
        testing::Values(POWER)));

INSTANTIATE_TEST_SUITE_P(
    MulToPower,
    MulOrAddConversionTests,
    testing::Combine(
        testing::Values(std::make_tuple(InputShape{DYN, 3, 64, 64}, CONST(ov::Shape({1}), 0.5), NONE, false),
                        std::make_tuple(InputShape{DYN, 3, DYN, 64}, CONST(ov::Shape({1}), 0.5), NONE, false),
                        std::make_tuple(InputShape{DYN, 3, DYN, DYN}, CONST(ov::Shape({1}), 0.5), NONE, false)),
        testing::Values(POWER)));

INSTANTIATE_TEST_SUITE_P(
    AddToPower,
    MulOrAddConversionTests,
    testing::Combine(
        testing::Values(std::make_tuple(InputShape{DYN, 3, 64, 64}, NONE, CONST(ov::Shape({1}), 0.5), false),
                        std::make_tuple(InputShape{DYN, 3, DYN, 64}, NONE, CONST(ov::Shape({1}), 0.5), false),
                        std::make_tuple(InputShape{DYN, 3, DYN, DYN}, NONE, CONST(ov::Shape({1}), 0.5), false)),
        testing::Values(POWER)));

INSTANTIATE_TEST_SUITE_P(
    MulAddNegative,
    MulAddConversionTests,
    testing::Combine(testing::Values(std::make_tuple(InputShape{DYN, 3, DYN},
                                                     CONST(ov::Shape({1, 1, 3, 1}), 0.5),
                                                     CONST(ov::Shape({3, 1}), 0.5) /*detect broadcast case*/,
                                                     false),
                                     std::make_tuple(InputShape{DYN, 3, DYN},
                                                     CONST(ov::Shape({3, 1}), 0.5),
                                                     CONST(ov::Shape({1, 1, 3, 1}), 0.5) /*detect broadcast case*/,
                                                     false),
                                     std::make_tuple(InputShape{DYN, DYN, DYN, DYN},
                                                     CONST(ov::Shape({1, 3, 1, 1}), 0.5),
                                                     CONST(ov::Shape({1, 3, 1, 1}), 0.5),
                                                     false),
                                     std::make_tuple(InputShape{DYN, DYN, DYN, DYN},
                                                     CONST(ov::Shape({1, 3, 2, 1}), 0.5),
                                                     CONST(ov::Shape({1, 3, 1, 1}), 0.5),
                                                     false),
                                     std::make_tuple(InputShape{1, 3, 2},
                                                     CONST(ov::Shape({1, 3, 1}), 0.5),
                                                     CONST(ov::Shape({1, 3, 2}), 0.5),
                                                     false),
                                     std::make_tuple(InputShape{1, DYN, 64, 64},
                                                     CONST(ov::Shape({1, 3, 1, 1}), 0.5),
                                                     CONST(ov::Shape({1, 3, 1, 1}), 0.5),
                                                     false)),
                     testing::Values(SAME)));

INSTANTIATE_TEST_SUITE_P(
    MulToEltwise,
    MulOrAddConversionTests,
    testing::Combine(
        testing::Values(
            std::make_tuple(InputShape{DYN, 3, 64}, CONST(ov::Shape({1, 1, 64}), 0.5), NONE, false),
            std::make_tuple(InputShape{DYN, 3, DYN}, CONST(ov::Shape({1, 1, 3, 1}), 0.5), NONE, false),
            std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, CONST(ov::Shape({1, 3, 1, 1}), 0.5), NONE, false),
            std::make_tuple(InputShape{DYN, 3, DYN, DYN}, CONST(ov::Shape({1, 3, 2, 1}), 0.5), NONE, false),
            std::make_tuple(InputShape{1, 3, 2}, CONST(ov::Shape({1, 3, 2}), 0.5), NONE, false),
            std::make_tuple(InputShape{1, DYN, 64, 64}, CONST(ov::Shape({1, 3, 1, 1}), 0.5), NONE, false),
            std::make_tuple(InputShape{64, 1, 64}, CONST(ov::Shape({64, 64, 64}), 1), NONE, false),
            std::make_tuple(InputShape{64, 64, 1}, CONST(ov::Shape({1, 1, 64}), 1), NONE, false),
            std::make_tuple(InputShape{DYN, 1, 64}, CONST(ov::Shape({64, 1, 64}), 1), NONE, false),
            std::make_tuple(InputShape{1, 1024, 768}, CONST(ov::Shape({768}), 0.5), NONE, true)),
        testing::Values(ELTWISE_PROD)));

INSTANTIATE_TEST_SUITE_P(
    AddToEltwise,
    MulOrAddConversionTests,
    testing::Combine(
        testing::Values(
            std::make_tuple(InputShape{DYN, 3, 64}, NONE, CONST(ov::Shape({1, 1, 64}), 0.5), false),
            std::make_tuple(InputShape{DYN, 3, DYN}, NONE, CONST(ov::Shape({1, 1, 3, 1}), 0.5), false),
            std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, NONE, CONST(ov::Shape({1, 3, 1, 1}), 0.5), false),
            std::make_tuple(InputShape{DYN, 3, DYN, DYN}, NONE, CONST(ov::Shape({1, 3, 2, 1}), 0.5), false),
            std::make_tuple(InputShape{1, 3, 2}, NONE, CONST(ov::Shape({1, 3, 2}), 0.5), false),
            std::make_tuple(InputShape{1, DYN, 64, 64}, NONE, CONST(ov::Shape({1, 3, 1, 1}), 0.5), false),
            std::make_tuple(InputShape{1, 1024, 768}, NONE, CONST(ov::Shape({768}), 0.5), true)),
        testing::Values(ELTWISE_SUM)));

#undef CONST
#undef SCALESHIFT
#undef POWER
#undef SAME
#undef ELTWISE_PROD
#undef ELTWISE_SUM
