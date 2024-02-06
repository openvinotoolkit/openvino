// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_convertaligntypes.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_align_types.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"

namespace {

struct ConvertAlignTypesTestParams {
    ov::PartialShape lhs_shape;
    ov::element::Type lhs_type;
    ov::PartialShape rhs_shape;
    ov::element::Type rhs_type;
    ov::element::Type expected_type;
};

class ConvertConvertAlignTypesTest : public TransformationTestsF,
                                     public testing::WithParamInterface<ConvertAlignTypesTestParams> {
private:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& parameters = GetParam();
        const auto& lhsType = parameters.lhs_type;
        const auto& lhsShape = parameters.lhs_shape;
        const auto& rhsType = parameters.rhs_type;
        const auto& rhsShape = parameters.rhs_shape;
        const auto& alignType = parameters.expected_type;
        model = transform(lhsShape, lhsType, rhsShape, rhsType);
        model_ref = reference(lhsShape, lhsType, rhsShape, rhsType, alignType);
        manager.register_pass<ov::pass::ConvertConvertAlignTypes>();
    }

protected:
    static std::shared_ptr<ov::Model> transform(const ov::PartialShape& lhsShape,
                                                const ov::element::Type& lhsType,
                                                const ov::PartialShape& rhsShape,
                                                const ov::element::Type& rhsType) {
        const auto lhs = std::make_shared<ov::op::v0::Parameter>(lhsType, lhsShape);
        const auto rhs = std::make_shared<ov::op::v0::Parameter>(rhsType, rhsShape);
        const auto convert_align_types = std::make_shared<ov::op::v14::ConvertAlignTypes>(lhs, rhs, true);
        return std::make_shared<ov::Model>(convert_align_types->outputs(), ov::ParameterVector{lhs, rhs}, "Actual");
    }

    static std::shared_ptr<ov::Model> reference(const ov::PartialShape& lhsShape,
                                                const ov::element::Type& lhsType,
                                                const ov::PartialShape& rhsShape,
                                                const ov::element::Type& rhsType,
                                                const ov::element::Type& alignType) {
        const auto lhs = std::make_shared<ov::op::v0::Parameter>(lhsType, lhsShape);
        const auto rhs = std::make_shared<ov::op::v0::Parameter>(rhsType, rhsShape);
        const auto lhs_converted = std::make_shared<ov::op::v0::Convert>(lhs, alignType);
        const auto rhs_converted = std::make_shared<ov::op::v0::Convert>(rhs, alignType);
        return std::make_shared<ov::Model>(ov::NodeVector{lhs_converted, rhs_converted},
                                           ov::ParameterVector{lhs, rhs},
                                           "Reference");
    }
};
INSTANTIATE_TEST_SUITE_P(
    ConvertAlignTypesDecomposition,
    ConvertConvertAlignTypesTest,
    testing::Values(
        ConvertAlignTypesTestParams{ov::PartialShape::dynamic(),
                                    ov::element::f32,
                                    ov::PartialShape::dynamic(),
                                    ov::element::f32,
                                    ov::element::f32},
        ConvertAlignTypesTestParams{ov::PartialShape::dynamic(),
                                    ov::element::u16,
                                    {5, 6, 7},
                                    ov::element::i16,
                                    ov::element::i32},
        ConvertAlignTypesTestParams{{1, 2, 3, 4},
                                    ov::element::u16,
                                    ov::PartialShape::dynamic(),
                                    ov::element::f32,
                                    ov::element::f32},
        ConvertAlignTypesTestParams{{1, {3, 7}, -1, 4},
                                    ov::element::f8e4m3,
                                    {0, 6, 7},
                                    ov::element::f8e5m2,
                                    ov::element::f16},
        ConvertAlignTypesTestParams{{}, ov::element::bf16, {5, 6, 7}, ov::element::f16, ov::element::f32},
        ConvertAlignTypesTestParams{{1, 2, 3, 4}, ov::element::u16, {}, ov::element::boolean, ov::element::u16},
        ConvertAlignTypesTestParams{{}, ov::element::u16, {}, ov::element::u1, ov::element::u16},
        ConvertAlignTypesTestParams{{-1}, ov::element::u64, {1}, ov::element::i16, ov::element::f32},
        ConvertAlignTypesTestParams{{1, 2, 3, 4},
                                    ov::element::boolean,
                                    {5, 6, 7},
                                    ov::element::boolean,
                                    ov::element::boolean},
        ConvertAlignTypesTestParams{{1, 2, 3, 4}, ov::element::i64, {5, 6, 7}, ov::element::f16, ov::element::f16}));
TEST_P(ConvertConvertAlignTypesTest, CompareFunctions) {}
}  // namespace
