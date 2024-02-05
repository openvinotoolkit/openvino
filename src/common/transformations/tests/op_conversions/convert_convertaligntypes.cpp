// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_convertaligntypes.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_align_types.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;
using namespace ov::op;

namespace {

struct ConvertAlignTypesTestParams {
    PartialShape lhs_shape;
    element::Type lhs_type;
    PartialShape rhs_shape;
    element::Type rhs_type;
    element::Type expected_type;
};

class ConvertConvertAlignTypesTest : public TransformationTestsF,
                                     public testing::WithParamInterface<ConvertAlignTypesTestParams> {
public:
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
        manager.register_pass<pass::ConvertConvertAlignTypes>();
    }

protected:
    static std::shared_ptr<Model> transform(const PartialShape& lhsShape,
                                            const element::Type& lhsType,
                                            const PartialShape& rhsShape,
                                            const element::Type& rhsType) {
        const auto lhs = std::make_shared<v0::Parameter>(lhsType, lhsShape);
        const auto rhs = std::make_shared<v0::Parameter>(rhsType, rhsShape);
        const auto convert_align_types = std::make_shared<v14::ConvertAlignTypes>(lhs, rhs, true);
        return std::make_shared<Model>(convert_align_types->outputs(), ParameterVector{lhs, rhs}, "Actual");
    }

    static std::shared_ptr<Model> reference(const PartialShape& lhsShape,
                                            const element::Type& lhsType,
                                            const PartialShape& rhsShape,
                                            const element::Type& rhsType,
                                            const element::Type& alignType) {
        const auto lhs = std::make_shared<v0::Parameter>(lhsType, lhsShape);
        const auto rhs = std::make_shared<v0::Parameter>(rhsType, rhsShape);
        const auto lhs_converted = std::make_shared<v0::Convert>(lhs, alignType);
        const auto rhs_converted = std::make_shared<v0::Convert>(rhs, alignType);
        return std::make_shared<Model>(NodeVector{lhs_converted, rhs_converted},
                                       ParameterVector{lhs, rhs},
                                       "Reference");
    }
};
INSTANTIATE_TEST_SUITE_P(
    smoke_NGraph,
    ConvertConvertAlignTypesTest,
    testing::Values(
        ConvertAlignTypesTestParams{PartialShape::dynamic(),
                                    element::f32,
                                    PartialShape::dynamic(),
                                    element::f32,
                                    element::f32},
        ConvertAlignTypesTestParams{PartialShape::dynamic(), element::u16, {5, 6, 7}, element::i16, element::i32},
        ConvertAlignTypesTestParams{{1, 2, 3, 4}, element::u16, PartialShape::dynamic(), element::f32, element::f32},
        ConvertAlignTypesTestParams{{1, {3, 7}, -1, 4}, element::f8e4m3, {0, 6, 7}, element::f8e5m2, element::f16},
        ConvertAlignTypesTestParams{{}, element::bf16, {5, 6, 7}, element::f16, element::f32},
        ConvertAlignTypesTestParams{{1, 2, 3, 4}, element::u16, {}, element::boolean, element::u16},
        ConvertAlignTypesTestParams{{}, element::u16, {}, element::u1, element::u16},
        ConvertAlignTypesTestParams{{-1}, element::u64, {1}, element::i16, element::f32},
        ConvertAlignTypesTestParams{{1, 2, 3, 4}, element::boolean, {5, 6, 7}, element::boolean, element::boolean},
        ConvertAlignTypesTestParams{{1, 2, 3, 4}, element::i64, {5, 6, 7}, element::f16, element::f16}));
TEST_P(ConvertConvertAlignTypesTest, CompareFunctions) {}
}  // namespace
