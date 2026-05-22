// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include "openvino/op/atan.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/select.hpp"

#include <plugin/transformations/increase_atan2_div_precision.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

// Builds the post-ConvertDivide form of atan2(imag, real), i.e.
//   Atan( Multiply(real, Power(imag, -1)) )
// which is what the pass actually sees in the GPU pipeline.
struct Atan2DivModel {
    std::shared_ptr<ov::op::v0::Parameter> real;
    std::shared_ptr<ov::op::v0::Parameter> imag;
    std::shared_ptr<ov::op::v1::Power> power;
    std::shared_ptr<ov::op::v1::Multiply> mul;
    std::shared_ptr<ov::op::v0::Atan> atan;
    std::shared_ptr<ov::Model> model;
};

Atan2DivModel build_atan2_div_model(ov::element::Type et) {
    Atan2DivModel m;
    m.real = std::make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{-1, -1});
    m.imag = std::make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{-1, -1});
    auto exponent = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{-1.0f});
    m.power = std::make_shared<ov::op::v1::Power>(m.imag, exponent);
    m.mul = std::make_shared<ov::op::v1::Multiply>(m.real, m.power);
    m.atan = std::make_shared<ov::op::v0::Atan>(m.mul);
    m.model = std::make_shared<ov::Model>(ov::OutputVector{m.atan},
                                          ov::ParameterVector{m.real, m.imag});
    return m;
}

}  // namespace

TEST(IncreaseAtan2DivPrecisionTest, PromoteAndGuardOnFp16) {
    auto m = build_atan2_div_model(ov::element::f16);

    // Sanity: pre-pass
    EXPECT_EQ(m.power->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(m.mul->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(m.atan->get_output_element_type(0), ov::element::f16);

    ov::pass::Manager manager;
    manager.register_pass<IncreaseAtan2DivPrecision>();
    manager.run_passes(m.model);

    // Power and Multiply should now run in f32.
    EXPECT_EQ(m.power->get_output_element_type(0), ov::element::f32)
        << "Power output should be promoted to f32";
    EXPECT_EQ(m.mul->get_output_element_type(0), ov::element::f32)
        << "Multiply output should be promoted to f32";
    EXPECT_EQ(m.atan->get_output_element_type(0), ov::element::f32)
        << "Atan output should be promoted to f32";

    // Power's f16 inputs (imag + Constant(-1)) must be wrapped in Convert(f32).
    for (size_t i = 0; i < m.power->get_input_size(); ++i) {
        auto input_node = m.power->get_input_node_shared_ptr(i);
        ASSERT_TRUE(ov::as_type_ptr<ov::op::v0::Convert>(input_node) != nullptr)
            << "Power input " << i << " should be a Convert node";
        EXPECT_EQ(input_node->get_output_element_type(0), ov::element::f32);
    }

    // Multiply's f16 real input must be wrapped in Convert(f32). The other
    // input comes from Power (already f32) so a Convert is not strictly
    // required there; either Power directly or a Convert(f32->f32) is fine.
    {
        auto real_side = m.mul->get_input_node_shared_ptr(0);
        ASSERT_TRUE(ov::as_type_ptr<ov::op::v0::Convert>(real_side) != nullptr)
            << "Multiply input 0 (real) should be a Convert node";
        EXPECT_EQ(real_side->get_output_element_type(0), ov::element::f32);
    }

    // Guard: Atan's input should now be a Select node (the imag==0 guard),
    // not the Multiply directly.
    auto atan_input = m.atan->get_input_node_shared_ptr(0);
    auto select = ov::as_type_ptr<ov::op::v1::Select>(atan_input);
    ASSERT_TRUE(select != nullptr)
        << "Atan input should be a Select (imag==0 guard) after the pass";

    // Select condition: Equal(imag, 0)
    auto cond = select->get_input_node_shared_ptr(0);
    ASSERT_TRUE(ov::as_type_ptr<ov::op::v1::Equal>(cond) != nullptr)
        << "Select condition should be an Equal node";

    // Select's "else" branch should trace back to the Multiply.
    auto select_else = select->get_input_node_shared_ptr(2);
    EXPECT_EQ(select_else.get(), m.mul.get())
        << "Select else input should be the original Multiply output";

    // Multiply should now have exactly one consumer: the Select guard.
    EXPECT_EQ(m.mul->get_users().size(), 1u);
    EXPECT_EQ(m.mul->get_users()[0].get(), select.get());

    // Atan output should feed into a Convert(f16) restore.
    EXPECT_EQ(m.atan->get_users().size(), 1u);
    auto atan_user = m.atan->get_users()[0];
    auto restore = ov::as_type_ptr<ov::op::v0::Convert>(atan_user);
    ASSERT_TRUE(restore != nullptr) << "Atan output should feed a Convert node";
    EXPECT_EQ(restore->get_output_element_type(0), ov::element::f16)
        << "Restore Convert should produce f16";
}

TEST_F(TransformationTestsF, IncreaseAtan2DivPrecision_AlreadyFp32_NoOp) {
    {
        auto m = build_atan2_div_model(ov::element::f32);
        model = m.model;
        manager.register_pass<IncreaseAtan2DivPrecision>();
    }
    {
        auto m = build_atan2_div_model(ov::element::f32);
        model_ref = m.model;
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreaseAtan2DivPrecision_NoPowerInput_NoOp) {
    // Multiply(a, b) -> Atan, where neither input is a Power node.
    {
        auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1});
        auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1});
        auto mul = std::make_shared<ov::op::v1::Multiply>(a, b);
        auto atan = std::make_shared<ov::op::v0::Atan>(mul);
        model = std::make_shared<ov::Model>(ov::OutputVector{atan}, ov::ParameterVector{a, b});
        manager.register_pass<IncreaseAtan2DivPrecision>();
    }
    {
        auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1});
        auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1});
        auto mul = std::make_shared<ov::op::v1::Multiply>(a, b);
        auto atan = std::make_shared<ov::op::v0::Atan>(mul);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{atan}, ov::ParameterVector{a, b});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreaseAtan2DivPrecision_NonConstantExponent_NoOp) {
    // Power's exponent is a Parameter (non-Constant), so the pattern should
    // not match — atan2 decomposition always uses a Constant -1 exponent.
    {
        auto real = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1});
        auto imag = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1});
        auto exponent = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{});
        auto power = std::make_shared<ov::op::v1::Power>(imag, exponent);
        auto mul = std::make_shared<ov::op::v1::Multiply>(real, power);
        auto atan = std::make_shared<ov::op::v0::Atan>(mul);
        model = std::make_shared<ov::Model>(ov::OutputVector{atan},
                                            ov::ParameterVector{real, imag, exponent});
        manager.register_pass<IncreaseAtan2DivPrecision>();
    }
    {
        auto real = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1});
        auto imag = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1});
        auto exponent = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{});
        auto power = std::make_shared<ov::op::v1::Power>(imag, exponent);
        auto mul = std::make_shared<ov::op::v1::Multiply>(real, power);
        auto atan = std::make_shared<ov::op::v0::Atan>(mul);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{atan},
                                                ov::ParameterVector{real, imag, exponent});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
