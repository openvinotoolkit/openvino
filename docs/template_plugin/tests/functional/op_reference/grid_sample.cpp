// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grid_sample.hpp"

#include "base_reference_test.hpp"
#include "gtest/gtest.h"

using namespace ov;
using namespace reference_tests;

namespace {
struct GridSampleParams {
    GridSampleParams(const reference_tests::Tensor& data,
                     const reference_tests::Tensor& grid,
                     const op::v9::GridSample::Attributes& attrs,
                     const reference_tests::Tensor& expected,
                     const std::string& name)
        : data_tensor{data},
          grid_tensor{grid},
          attributes{attrs},
          expected_tensor{expected},
          test_case_name{name} {}

    reference_tests::Tensor data_tensor;
    reference_tests::Tensor grid_tensor;
    op::v9::GridSample::Attributes attributes;
    reference_tests::Tensor expected_tensor;
    std::string test_case_name;
};

class ReferenceGridSample : public testing::TestWithParam<GridSampleParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.data_tensor, params.grid_tensor, params.attributes);
        inputData = {params.data_tensor.data, params.grid_tensor.data};
        refOutData = {params.expected_tensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GridSampleParams>& obj) {
        return obj.param.test_case_name;
    }

private:
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& data,
                                                 const reference_tests::Tensor& grid,
                                                 const op::v9::GridSample::Attributes& attributes) {
        const auto in1 = std::make_shared<op::v0::Parameter>(data.type, data.shape);
        const auto in2 = std::make_shared<op::v0::Parameter>(grid.type, grid.shape);
        const auto grid_sample = std::make_shared<op::v9::GridSample>(in1, in2, attributes);
        return std::make_shared<Model>(NodeVector{grid_sample}, ParameterVector{in1, in2});
    }
};

constexpr auto GS_BICUBIC{op::v9::GridSample::InterpolationMode::BICUBIC};
constexpr auto GS_BILINEAR{op::v9::GridSample::InterpolationMode::BILINEAR};
constexpr auto GS_NEAREST{op::v9::GridSample::InterpolationMode::NEAREST};

constexpr auto GS_BORDER{op::v9::GridSample::PaddingMode::BORDER};
constexpr auto GS_REFLECTION{op::v9::GridSample::PaddingMode::REFLECTION};
constexpr auto GS_ZEROS{op::v9::GridSample::PaddingMode::ZEROS};

constexpr std::array<op::v9::GridSample::PaddingMode, 3> padding_modes{GS_ZEROS, GS_BORDER, GS_REFLECTION};
constexpr std::array<bool, 2> align_modes{false, true};

std::vector<GridSampleParams> generateNearestParamsOddDimensionsInnerGrids() {
    std::vector<GridSampleParams> params;

    reference_tests::Tensor data_odd_dims{{1, 1, 3, 5},
                                          element::f32,
                                          std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    reference_tests::Tensor grid_inner{
        {1, 3, 4, 2},
        element::f32,
        std::vector<float>{-0.1, -0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.5, -0.5, -0.5, 0.5,
                           0.5,  -0.5, 0.5,  0.5, -1., -1.,  -1., 1.,  1.,   -1.,  1.,   1.}};
    reference_tests::Tensor output{{1, 1, 3, 4},
                                   element::f32,
                                   std::vector<float>{8, 8, 8, 8, 2, 12, 4, 14, 1, 11, 5, 15}};
    for (const auto& padding : padding_modes) {
        for (const auto align : align_modes) {
            std::stringstream name;
            name << "nearest_" << padding << (align ? "_align" : "_noalign") << "_odd_dims_inner";
            params.emplace_back(
                data_odd_dims,
                grid_inner,
                op::v9::GridSample::Attributes{align, GS_NEAREST, padding},
                output,
                name.str());
        }
    }
    return params;
}

std::vector<GridSampleParams> generateNearestParamsOddDimensionsOuterGrids() {
    std::vector<GridSampleParams> params;
    reference_tests::Tensor data_odd_dims{{1, 1, 3, 5},
                                          element::f32,
                                          std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    reference_tests::Tensor grid_outer{
        {1, 1, 7, 2},
        element::f32,
        std::vector<float>{-10.1, -9.7, -7.55, 0.37, -77., 11.56, 0.5, 2.55, 1.7, 1.1, 3., -0.17, 1.301, -1.001}};

    params.emplace_back(data_odd_dims,
                        grid_outer,
                        op::v9::GridSample::Attributes{false, GS_NEAREST, GS_ZEROS},
                        reference_tests::Tensor{{1, 1, 1, 7}, element::f32, std::vector<float>{0, 0, 0, 0, 0, 0, 0}},
                        "nearest_zeros_noalign_odd_dims_outer");

    params.emplace_back(data_odd_dims,
                        grid_outer,
                        op::v9::GridSample::Attributes{true, GS_NEAREST, GS_ZEROS},
                        reference_tests::Tensor{{1, 1, 1, 7}, element::f32, std::vector<float>{0, 0, 0, 0, 0, 0, 0}},
                        "nearest_zeros_align_odd_dims_outer");

    params.emplace_back(
        data_odd_dims,
        grid_outer,
        op::v9::GridSample::Attributes{false, GS_NEAREST, GS_BORDER},
        reference_tests::Tensor{{1, 1, 1, 7}, element::f32, std::vector<float>{1, 11, 11, 14, 15, 10, 5}},
        "nearest_border_noalign_odd_dims_outer");

    params.emplace_back(
        data_odd_dims,
        grid_outer,
        op::v9::GridSample::Attributes{true, GS_NEAREST, GS_BORDER},
        reference_tests::Tensor{{1, 1, 1, 7}, element::f32, std::vector<float>{1, 6, 11, 14, 15, 10, 5}},
        "nearest_border_align_odd_dims_outer");

    params.emplace_back(
        data_odd_dims,
        grid_outer,
        op::v9::GridSample::Attributes{false, GS_NEAREST, GS_REFLECTION},
        reference_tests::Tensor{{1, 1, 1, 7}, element::f32, std::vector<float>{8, 14, 1, 4, 14, 6, 5}},
        "nearest_reflection_noalign_odd_dims_outer");

    params.emplace_back(
        data_odd_dims,
        grid_outer,
        op::v9::GridSample::Attributes{true, GS_NEAREST, GS_REFLECTION},
        reference_tests::Tensor{{1, 1, 1, 7}, element::f32, std::vector<float>{8, 9, 6, 4, 14, 6, 4}},
        "nearest_reflection_align_odd_dims_outer");

    return params;
}

std::vector<GridSampleParams> generateNearestParamsEvenDimensions() {
    std::vector<GridSampleParams> params;
    reference_tests::Tensor data_even_dims{
        {1, 1, 4, 6},
        element::f32,
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    reference_tests::Tensor grid_inner{
        {1, 1, 8, 2},
        element::f32,
        std::vector<float>{-0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -1, 1, 1, -1, -0.1, -0.1, 0.1, 0.1}};

    reference_tests::Tensor output_align{{1, 1, 1, 8}, element::f32, std::vector<float>{8, 14, 11, 17, 19, 6, 9, 16}};
    reference_tests::Tensor output_noalign{{1, 1, 1, 8}, element::f32, std::vector<float>{2, 14, 5, 17, 19, 6, 9, 16}};
    reference_tests::Tensor output_zeros_noalign{{1, 1, 1, 8},
                                                 element::f32,
                                                 std::vector<float>{2, 14, 5, 17, 0, 0, 9, 16}};

    for (const auto& padding : padding_modes) {
        std::stringstream name1, name2;
        name1 << "nearest_" << padding << "_noalign"
              << "_even_dims_inner";
        params.emplace_back(data_even_dims,
                            grid_inner,
                            op::v9::GridSample::Attributes{false, GS_NEAREST, padding},
                            padding == GS_ZEROS ? output_zeros_noalign : output_noalign,
                            name1.str());

        name2 << "nearest_" << padding << "_align"
              << "_even_dims_inner";
        params.emplace_back(data_even_dims,
                            grid_inner,
                            op::v9::GridSample::Attributes{true, GS_NEAREST, padding},
                            output_align,
                            name2.str());
    }

    return params;
}

std::vector<GridSampleParams> generateBilinearParamsOddDimensionsInnerGrids() {
    std::vector<GridSampleParams> params;
    reference_tests::Tensor data_odd_dims{{1, 1, 3, 5},
                                          element::f32,
                                          std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    reference_tests::Tensor grid_inner{
        {1, 3, 4, 2},
        element::f32,
        std::vector<float>{-0.1, -0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.5, -0.5, -0.5, 0.5,
                           0.5,  -0.5, 0.5,  0.5, -1., -1.,  -1., 1.,  1.,   -1.,  1.,   1.}};

    reference_tests::Tensor output_align{{1, 1, 3, 4},
                                         element::f32,
                                         std::vector<float>{7.3, 8.3, 7.7, 8.7, 4.5, 9.5, 6.5, 11.5, 1, 11, 5, 15}};
    reference_tests::Tensor output_noalign{{1, 1, 3, 4},
                                           element::f32,
                                           std::vector<float>{7, 8.5, 7.5, 9, 3, 10.5, 5.5, 13, 1, 11, 5, 15}};
    reference_tests::Tensor output_zeros_noalign{
        {1, 1, 3, 4},
        element::f32,
        std::vector<float>{7, 8.5, 7.5, 9, 3, 10.5, 5.5, 13, 0.25, 2.75, 1.25, 3.75}};

    params.emplace_back(data_odd_dims,
                        grid_inner,
                        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_ZEROS},
                        output_zeros_noalign,
                        "bilinear_zeros_noalign_odd_dims_inner");

    params.emplace_back(data_odd_dims,
                        grid_inner,
                        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_ZEROS},
                        output_align,
                        "bilinear_zeros_align_odd_dims_inner");

    params.emplace_back(
        data_odd_dims,
        grid_inner,
        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_BORDER},
        output_noalign,
        "bilinear_border_noalign_odd_dims_inner");

    params.emplace_back(
        data_odd_dims,
        grid_inner,
        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_BORDER},
        output_align,
        "bilinear_border_align_odd_dims_inner");

    params.emplace_back(
        data_odd_dims,
        grid_inner,
        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_REFLECTION},
        output_noalign,
        "bilinear_reflection_noalign_odd_dims_inner");

    params.emplace_back(
        data_odd_dims,
        grid_inner,
        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_REFLECTION},
        output_align,
        "bilinear_reflection_align_odd_dims_inner");

    return params;
}

std::vector<GridSampleParams> generateBilinearParamsOddDimensionsOuterGrids() {
    std::vector<GridSampleParams> params;

    reference_tests::Tensor data_odd_dims{{1, 1, 3, 5},
                                          element::f32,
                                          std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};

    reference_tests::Tensor grid_outer{
        {1, 1, 7, 2},
        element::f32,
        std::vector<float>{-10.1, -9.7, -7.55, 0.37, -77., 11.56, 0.5, 2.55, 1.7, 1.1, 3., -0.17, 1.301, -1.001}};

    params.emplace_back(data_odd_dims,
                        grid_outer,
                        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_ZEROS},
                        reference_tests::Tensor{{1, 1, 1, 7}, element::f32, std::vector<float>{0, 0, 0, 0, 0, 0, 0}},
                        "bilinear_zeros_noalign_odd_dims_outer");

    params.emplace_back(
        data_odd_dims,
        grid_outer,
        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_ZEROS},
        reference_tests::Tensor{{1, 1, 1, 7}, element::f32, std::vector<float>{0, 0, 0, 0, 0, 0, 1.9880099}},
        "bilinear_zeros_align_odd_dims_outer");

    params.emplace_back(
        data_odd_dims,
        grid_outer,
        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_BORDER},
        reference_tests::Tensor{{1, 1, 1, 7}, element::f32, std::vector<float>{1, 8.775, 11, 14.25, 15, 8.725, 5}},
        "bilinear_border_noalign_odd_dims_outer");

    params.emplace_back(
        data_odd_dims,
        grid_outer,
        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_BORDER},
        reference_tests::Tensor{{1, 1, 1, 7}, element::f32, std::vector<float>{1, 7.85, 11, 14, 15, 9.15, 5}},
        "bilinear_border_align_odd_dims_outer");

    params.emplace_back(
        data_odd_dims,
        grid_outer,
        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_REFLECTION},
        reference_tests::Tensor{{1, 1, 1, 7},
                                element::f32,
                                std::vector<float>{5.9999995, 11.9, 2.7000031, 5.1250005, 13.75, 4.725, 4.7475}},
        "bilinear_reflection_noalign_odd_dims_outer");

    params.emplace_back(
        data_odd_dims,
        grid_outer,
        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_REFLECTION},
        reference_tests::Tensor{{1, 1, 1, 7},
                                element::f32,
                                std::vector<float>{6.7, 10.75, 3.800002, 6.25, 13.099999, 5.15, 4.4030004}},
        "bilinear_reflection_align_odd_dims_outer");

    return params;
}

std::vector<GridSampleParams> generateBilinearParamsEvenDimensions() {
    std::vector<GridSampleParams> params;
    reference_tests::Tensor data_even_dims{
        {1, 1, 4, 6},
        element::f32,
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};
    reference_tests::Tensor grid_inner{
        {1, 1, 8, 2},
        element::f32,
        std::vector<float>{-0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -1, 1, 1, -1, -0.1, -0.1, 0.1, 0.1}};

    params.emplace_back(
        data_even_dims,
        grid_inner,
        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_ZEROS},
        reference_tests::Tensor{{1, 1, 1, 7}, element::f32, std::vector<float>{5, 17, 8, 20, 4.75, 1.5, 11, 14}},
        "bilinear_zeros_noalign_even_dims_inner");

    params.emplace_back(data_even_dims,
                        grid_inner,
                        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_ZEROS},
                        reference_tests::Tensor{{1, 1, 1, 7},
                                                element::f32,
                                                std::vector<float>{6.75, 15.75, 9.25, 18.25, 19, 6, 11.35, 13.65}},
                        "bilinear_zeros_align_even_dims_inner");

    params.emplace_back(
        data_even_dims,
        grid_inner,
        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_BORDER},
        reference_tests::Tensor{{1, 1, 1, 7}, element::f32, std::vector<float>{5, 17, 8, 20, 19, 6, 11, 14}},
        "bilinear_border_noalign_even_dims_inner");

    params.emplace_back(data_even_dims,
                        grid_inner,
                        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_BORDER},
                        reference_tests::Tensor{{1, 1, 1, 7},
                                                element::f32,
                                                std::vector<float>{6.75, 15.75, 9.25, 18.25, 19, 6, 11.35, 13.65}},
                        "bilinear_border_align_even_dims_inner");

    params.emplace_back(
        data_even_dims,
        grid_inner,
        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_REFLECTION},
        reference_tests::Tensor{{1, 1, 1, 7}, element::f32, std::vector<float>{5, 17, 8, 20, 19, 6, 11, 14}},
        "bilinear_reflection_noalign_even_dims_inner");

    params.emplace_back(data_even_dims,
                        grid_inner,
                        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_REFLECTION},
                        reference_tests::Tensor{{1, 1, 1, 7},
                                                element::f32,
                                                std::vector<float>{6.75, 15.75, 9.25, 18.25, 19, 6, 11.35, 13.65}},
                        "bilinear_reflection_align_even_dims_inner");

    return params;
}

std::vector<GridSampleParams> generateGridSampleParams() {
    std::vector<std::vector<GridSampleParams>> all_params{generateNearestParamsOddDimensionsInnerGrids(),
                                                          generateNearestParamsOddDimensionsOuterGrids(),
                                                          generateNearestParamsEvenDimensions(),
                                                          generateBilinearParamsOddDimensionsInnerGrids(),
                                                          generateBilinearParamsOddDimensionsOuterGrids(),
                                                          generateBilinearParamsEvenDimensions()};
    std::vector<GridSampleParams> test_params;
    for (auto& params : all_params)
        std::move(params.begin(), params.end(), std::back_inserter(test_params));
    return test_params;
}
}  // namespace

TEST_P(ReferenceGridSample, LayerTest) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         ReferenceGridSample,
                         ::testing::ValuesIn(generateGridSampleParams()),
                         ReferenceGridSample::getTestCaseName);
