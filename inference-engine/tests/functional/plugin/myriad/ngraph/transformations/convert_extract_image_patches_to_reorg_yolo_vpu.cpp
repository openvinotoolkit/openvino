// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/ngraph/transformations/convert_extract_image_patches_to_reorg_yolo.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include <string>
#include <memory>

namespace {

using EIPParams = std::tuple<
        ngraph::PartialShape, ngraph::Shape, ngraph::Strides, ngraph::Shape, ngraph::op::PadType>;

class ConvertEIPToReorgYoloTest : public CommonTestUtils::TestsCommon,
                                  public testing::WithParamInterface<EIPParams> {
public:
    std::pair<bool, std::string> compare() {
        const auto& parameters = GetParam();
        const auto& dataShape  = std::get<0>(parameters);
        const auto& sizes = std::get<1>(parameters);
        const auto& strides = std::get<2>(parameters);
        const auto& rates = std::get<3>(parameters);
        const auto& padMode = std::get<4>(parameters);

        return compare_functions(
                transform(dataShape, sizes, strides, rates, padMode),
                reference(dataShape, strides));
    }
protected:
    static std::shared_ptr<ngraph::Function> transform(
            const ngraph::PartialShape& dataShape,
            const ngraph::Shape& sizes,
            const ngraph::Strides& strides,
            const ngraph::Shape& rates,
            const ngraph::op::PadType& padMode) {
        const auto param = std::make_shared<ngraph::opset5::Parameter>(
                ngraph::element::f32,
                dataShape);

        const auto eip = std::make_shared<ngraph::opset5::ExtractImagePatches>(
                param, sizes, strides, rates, padMode);

        auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{eip},
                ngraph::ParameterVector{param},
                "Actual");

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<vpu::ConvertExtractImagePatchesToReorgYolo>();
        manager.run_passes(function);

        return function;
    }

    static std::shared_ptr<ngraph::Function> reference(
            const ngraph::PartialShape& dataShape,
            const ngraph::Strides& strides) {
        const auto param = std::make_shared<ngraph::opset5::Parameter>(
                ngraph::element::f32,
                dataShape);

        auto reorgYolo = std::make_shared<ngraph::opset5::ReorgYolo>(param, strides);

        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{reorgYolo},
                ngraph::ParameterVector{param},
                "Expected");
    }
};

//
// Positive tests
//

class ConvertEIPToReorgYoloPositiveTest : public ConvertEIPToReorgYoloTest {};
TEST_P(ConvertEIPToReorgYoloPositiveTest, CompareFunctions) {
    const auto res = compare();
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, ConvertEIPToReorgYoloPositiveTest, testing::Combine(
        testing::Values(ngraph::Shape{1, 64, 500, 500}),
        testing::Values(ngraph::Shape{5, 5}),
        testing::Values(ngraph::Strides{5, 5}),
        testing::Values(ngraph::Shape{1, 1}),
        testing::Values(
                ngraph::op::PadType::VALID,
                ngraph::op::PadType::SAME_LOWER,
                ngraph::op::PadType::SAME_UPPER)
));

//
// Negative tests
//

class DoNotConvertEIPToReorgYoloOnDiffSizeAndStride : public ConvertEIPToReorgYoloTest {};
TEST_P(DoNotConvertEIPToReorgYoloOnDiffSizeAndStride, CompareFunctions) {
    const auto res = compare();
    ASSERT_FALSE(res.first) << res.second;
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DoNotConvertEIPToReorgYoloOnDiffSizeAndStride, testing::Combine(
        testing::Values(ngraph::PartialShape{1, 64, 500, 500}),
        testing::Values(ngraph::Shape{5, 5}),
        testing::Values(ngraph::Strides{4, 4}),
        testing::Values(ngraph::Shape{1, 1}),
        testing::Values(ngraph::op::PadType::VALID)
));

class DoNotConvertEIPToReorgYoloOnNot4DInput : public ConvertEIPToReorgYoloTest {};
TEST_P(DoNotConvertEIPToReorgYoloOnNot4DInput, CompareFunctions) {
    const auto& parameters = GetParam();
    const auto& dataShape  = std::get<0>(parameters);
    const auto& sizes = std::get<1>(parameters);
    const auto& strides = std::get<2>(parameters);
    const auto& rates = std::get<3>(parameters);
    const auto& padMode = std::get<4>(parameters);

    EXPECT_ANY_THROW(transform(dataShape, sizes, strides, rates, padMode));
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DoNotConvertEIPToReorgYoloOnNot4DInput, testing::Combine(
        testing::Values(ngraph::PartialShape{1, 1, 64, 500, 500},
                        ngraph::PartialShape{64, 500, 500},
                        ngraph::PartialShape{500, 500},
                        ngraph::PartialShape{500},
                        ngraph::PartialShape::dynamic()),
        testing::Values(ngraph::Shape{5, 5}),
        testing::Values(ngraph::Strides{5, 5}),
        testing::Values(ngraph::Shape{1, 1}),
        testing::Values(ngraph::op::PadType::VALID)
));

class DoNotConvertEIPToReorgYoloOnNotStaticInput : public ConvertEIPToReorgYoloTest {};
TEST_P(DoNotConvertEIPToReorgYoloOnNotStaticInput, CompareFunctions) {
    const auto& parameters = GetParam();
    const auto& dataShape  = std::get<0>(parameters);
    const auto& sizes = std::get<1>(parameters);
    const auto& strides = std::get<2>(parameters);
    const auto& rates = std::get<3>(parameters);
    const auto& padMode = std::get<4>(parameters);

    const auto& function = transform(dataShape, sizes, strides, rates, padMode);
    const auto& ops = function->get_ops();
    const auto reorgIt = std::find_if(ops.begin(), ops.end(), [](const std::shared_ptr<ngraph::Node>& op) {
        return ngraph::is_type<ngraph::opset5::ReorgYolo>(op); });
    ASSERT_TRUE(reorgIt == ops.end());
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DoNotConvertEIPToReorgYoloOnNotStaticInput, testing::Combine(
        testing::Values(ngraph::PartialShape{1, 64, ngraph::Dimension::dynamic(), 500},
                        ngraph::PartialShape{1, 64, 500, ngraph::Dimension::dynamic()}),
        testing::Values(ngraph::Shape{5, 5}),
        testing::Values(ngraph::Strides{5, 5}),
        testing::Values(ngraph::Shape{1, 1}),
        testing::Values(ngraph::op::PadType::VALID)
));

class DoNotConvertEIPToReorgYoloOnNonSingleRates : public ConvertEIPToReorgYoloTest {};
TEST_P(DoNotConvertEIPToReorgYoloOnNonSingleRates, CompareFunctions) {
    const auto res = compare();
    ASSERT_FALSE(res.first) << res.second;
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DoNotConvertEIPToReorgYoloOnNonSingleRates, testing::Combine(
        testing::Values(ngraph::Shape{1, 64, 500, 500}),
        testing::Values(ngraph::Shape{5, 5}),
        testing::Values(ngraph::Strides{5, 5}),
        testing::Values(ngraph::Shape{2, 2}),
        testing::Values(ngraph::op::PadType::VALID)
));

}  // namespace
