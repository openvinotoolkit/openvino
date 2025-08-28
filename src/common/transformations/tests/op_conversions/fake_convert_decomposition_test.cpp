// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/fake_convert_decomposition.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/fake_convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/opsets/opset13_decl.hpp"
#include "openvino/opsets/opset1_decl.hpp"

using namespace ov;

class FakeConvertDecompositionTestBase : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
        manager.register_pass<ov::pass::FakeConvertDecomposition>();
    }
};

using FakeConvertDecompositionParams = std::tuple<Shape,            // data shape
                                                  Shape,            // scale shape
                                                  Shape,            // shift shape
                                                  element::Type_t,  // input precision
                                                  element::Type_t,  // destination precision
                                                  bool>;            // default shift

class FakeConvertDecompositionTest : public FakeConvertDecompositionTestBase,
                                     public ::testing::WithParamInterface<FakeConvertDecompositionParams> {
public:
    static std::string getTestCaseName(::testing::TestParamInfo<FakeConvertDecompositionParams> obj) {
        FakeConvertDecompositionParams params = obj.param;

        const auto& [data_shape, scale_shape, shift_shape, data_prec, dst_prec, default_shift] = params;

        std::ostringstream result;
        result << "dataShape=" << ov::test::utils::vec2str(data_shape) << "_";
        result << "scaleShape=" << ov::test::utils::vec2str(scale_shape) << "_";
        result << "shiftShape=" << ov::test::utils::vec2str(shift_shape) << "_";
        result << "dataPrecision=" << element::Type(data_prec) << "_";
        result << "destinationPrecision=" << element::Type(dst_prec) << "_";
        if (default_shift)
            result << "defaultShift=true";
        else
            result << "defaultShift=false";
        return result.str();
    }

protected:
    void SetUp() override {
        FakeConvertDecompositionTestBase::SetUp();
        const auto [data_shape, scale_shape, shift_shape, data_prec, dst_prec, default_shift] = this->GetParam();
        {
            const auto data = std::make_shared<opset1::Parameter>(data_prec, PartialShape(data_shape));
            const auto scale = std::make_shared<opset1::Constant>(data_prec, scale_shape, 4.f);
            const auto shift = std::make_shared<opset1::Constant>(data_prec, shift_shape, 12.f);

            const auto fake_convert = default_shift
                                          ? std::make_shared<opset13::FakeConvert>(data, scale, dst_prec)
                                          : std::make_shared<opset13::FakeConvert>(data, scale, shift, dst_prec);
            model = std::make_shared<ov::Model>(OutputVector{fake_convert}, ParameterVector{data});
        }

        {
            const auto data = std::make_shared<opset1::Parameter>(data_prec, PartialShape(data_shape));
            const auto input_scale = std::make_shared<opset1::Constant>(data_prec, scale_shape, 4.f);
            const auto input_shift = std::make_shared<opset1::Constant>(data_prec, shift_shape, 12.f);

            const auto lower_bound = dst_prec == ov::element::f8e4m3
                                         ? static_cast<float>(std::numeric_limits<ov::float8_e4m3>::lowest())
                                         : static_cast<float>(std::numeric_limits<ov::float8_e5m2>::lowest());
            const auto upper_bound = dst_prec == ov::element::f8e4m3
                                         ? static_cast<float>(std::numeric_limits<ov::float8_e4m3>::max())
                                         : static_cast<float>(std::numeric_limits<ov::float8_e5m2>::max());

            std::shared_ptr<Node> result;
            const auto scale = std::make_shared<ov::op::v1::Multiply>(data, input_scale);
            if (default_shift) {
                const auto clamp = std::make_shared<ov::op::v0::Clamp>(scale, lower_bound, upper_bound);
                const auto downconvert = std::make_shared<ov::op::v0::Convert>(clamp, dst_prec);
                result = std::make_shared<ov::op::v0::Convert>(downconvert, data_prec);
            } else {
                const auto shift = std::make_shared<ov::op::v1::Subtract>(scale, input_shift);

                const auto clamp = std::make_shared<ov::op::v0::Clamp>(shift, lower_bound, upper_bound);
                const auto downconvert = std::make_shared<ov::op::v0::Convert>(clamp, dst_prec);
                const auto upconvert = std::make_shared<ov::op::v0::Convert>(downconvert, data_prec);

                const auto output_shift = std::make_shared<ov::op::v0::Constant>(data_prec, shift_shape, -12.f);
                result = std::make_shared<ov::op::v1::Subtract>(upconvert, output_shift);
            }
            const auto output_scale = std::make_shared<opset1::Constant>(data_prec, scale_shape, 1.f / 4.f);
            result = std::make_shared<ov::op::v1::Multiply>(result, output_scale);
            model_ref = std::make_shared<ov::Model>(OutputVector{result}, ParameterVector{data});
        }
    }
};

TEST_P(FakeConvertDecompositionTest, CompareFunctions) {}

const std::vector<element::Type_t> data_precisions = {element::Type_t::f32,
                                                      element::Type_t::f16,
                                                      element::Type_t::bf16};

const std::vector<element::Type_t> destination_precisions = {element::Type_t::f8e4m3, element::Type_t::f8e5m2};

const std::vector<bool> default_shift = {true, false};

const auto simple_fake_convert_params = ::testing::Combine(::testing::Values(Shape{2, 3, 4, 5}),
                                                           ::testing::Values(Shape{1}),
                                                           ::testing::Values(Shape{1}),
                                                           ::testing::ValuesIn(data_precisions),
                                                           ::testing::ValuesIn(destination_precisions),
                                                           ::testing::ValuesIn(default_shift));

const auto broadcast_fake_convert_params = ::testing::Combine(::testing::Values(Shape{2, 3, 4, 5}),
                                                              ::testing::Values(Shape{2, 3, 1, 1}),
                                                              ::testing::Values(Shape{2, 3, 1, 1}),
                                                              ::testing::ValuesIn(data_precisions),
                                                              ::testing::ValuesIn(destination_precisions),
                                                              ::testing::ValuesIn(default_shift));

const auto elementwise_fake_convert_params = ::testing::Combine(::testing::Values(Shape{2, 3, 4, 5}),
                                                                ::testing::Values(Shape{2, 3, 4, 5}),
                                                                ::testing::Values(Shape{2, 3, 4, 5}),
                                                                ::testing::ValuesIn(data_precisions),
                                                                ::testing::ValuesIn(destination_precisions),
                                                                ::testing::ValuesIn(default_shift));

INSTANTIATE_TEST_SUITE_P(SimpleFakeConvert_Decomposition,
                         FakeConvertDecompositionTest,
                         simple_fake_convert_params,
                         FakeConvertDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(BroadcastFakeConvert_Decomposition,
                         FakeConvertDecompositionTest,
                         broadcast_fake_convert_params,
                         FakeConvertDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ElementwiseFakeConvert_Decomposition,
                         FakeConvertDecompositionTest,
                         elementwise_fake_convert_params,
                         FakeConvertDecompositionTest::getTestCaseName);

TEST_F(FakeConvertDecompositionTestBase, FakeConvert_Decomposition_TrivialShift) {
    const ov::element::Type_t data_prec = ov::element::f32;
    const ov::PartialShape data_shape{2, 3, 4, 5};
    const ov::element::Type_t dst_prec = ov::element::f8e4m3;
    {
        const auto data = std::make_shared<opset1::Parameter>(data_prec, PartialShape(data_shape));
        const auto scale = std::make_shared<opset1::Constant>(data_prec, Shape{}, 4.f);
        const auto shift = std::make_shared<opset1::Constant>(data_prec, Shape{}, 0.f);

        const auto fake_convert = std::make_shared<opset13::FakeConvert>(data, scale, shift, dst_prec);
        model = std::make_shared<ov::Model>(OutputVector{fake_convert}, ParameterVector{data});
    }

    {
        const auto data = std::make_shared<opset1::Parameter>(data_prec, PartialShape(data_shape));
        const auto input_scale = std::make_shared<opset1::Constant>(data_prec, Shape{}, 4.f);

        const auto lower_bound = static_cast<float>(std::numeric_limits<ov::float8_e4m3>::lowest());
        const auto upper_bound = static_cast<float>(std::numeric_limits<ov::float8_e4m3>::max());

        const auto scale = std::make_shared<ov::op::v1::Multiply>(data, input_scale);
        const auto clamp = std::make_shared<ov::op::v0::Clamp>(scale, lower_bound, upper_bound);
        const auto downconvert = std::make_shared<ov::op::v0::Convert>(clamp, dst_prec);
        const auto upconvert = std::make_shared<ov::op::v0::Convert>(downconvert, data_prec);
        const auto output_scale = std::make_shared<opset1::Constant>(data_prec, Shape{}, 1.f / 4.f);
        const auto out_multiply = std::make_shared<ov::op::v1::Multiply>(upconvert, output_scale);
        model_ref = std::make_shared<ov::Model>(OutputVector{out_multiply}, ParameterVector{data});
    }
}
