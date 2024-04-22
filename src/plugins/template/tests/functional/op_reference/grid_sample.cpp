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
constexpr std::array<bool, 2> align_corners_modes{false, true};

std::string param_types_str(const element::Type& data_et, const element::Type& grid_et) {
    std::stringstream types;
    types << "_data_et_" << data_et << "_grid_et_" << grid_et;
    return types.str();
}

template <ov::element::Type_t DATA_ET,
          ov::element::Type_t GRID_ET,
          class DT = ov::fundamental_type_for<DATA_ET>,
          class GT = ov::fundamental_type_for<GRID_ET>>
std::vector<GridSampleParams> generateNearestParamsOddDimensionsInnerGrids() {
    std::vector<GridSampleParams> params;

    reference_tests::Tensor data_odd_dims{{1, 1, 3, 5},
                                          DATA_ET,
                                          std::vector<DT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    reference_tests::Tensor grid_inner{{1, 3, 4, 2}, GRID_ET, std::vector<GT>{-0.1, -0.1, -0.1, 0.1,  0.1,  -0.1,
                                                                              0.1,  0.1,  -0.5, -0.5, -0.5, 0.5,
                                                                              0.5,  -0.5, 0.5,  0.5,  -1.,  -1.,
                                                                              -1.,  1.,   1.,   -1.,  1.,   1.}};
    reference_tests::Tensor output{{1, 1, 3, 4}, DATA_ET, std::vector<DT>{8, 8, 8, 8, 2, 12, 4, 14, 1, 11, 5, 15}};

    for (const auto& padding : padding_modes) {
        for (const auto align : align_corners_modes) {
            std::stringstream name;
            name << "nearest_" << padding << (align ? "_align" : "_noalign") << "_odd_dims_inner";
            name << param_types_str(DATA_ET, GRID_ET);
            params.emplace_back(data_odd_dims,
                                grid_inner,
                                op::v9::GridSample::Attributes{align, GS_NEAREST, padding},
                                output,
                                name.str());
        }
    }
    return params;
}

template <ov::element::Type_t DATA_ET,
          ov::element::Type_t GRID_ET,
          class DT = ov::fundamental_type_for<DATA_ET>,
          class GT = ov::fundamental_type_for<GRID_ET>>
std::vector<GridSampleParams> generateNearestParamsOddDimensionsOuterGrids() {
    std::vector<GridSampleParams> params;
    reference_tests::Tensor data_odd_dims{{1, 1, 3, 5},
                                          DATA_ET,
                                          std::vector<DT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    reference_tests::Tensor grid_outer{
        {1, 1, 7, 2},
        GRID_ET,
        std::vector<GT>{-10.1, -9.7, -7.55, 0.37, -77., 11.56, 0.5, 2.55, 1.7, 1.1, 3., -0.17, 1.301, -1.001}};

    const auto types_str = param_types_str(DATA_ET, GRID_ET);

    params.emplace_back(data_odd_dims,
                        grid_outer,
                        op::v9::GridSample::Attributes{false, GS_NEAREST, GS_ZEROS},
                        reference_tests::Tensor{{1, 1, 1, 7}, DATA_ET, std::vector<DT>{0, 0, 0, 0, 0, 0, 0}},
                        "nearest_zeros_noalign_odd_dims_outer" + types_str);

    params.emplace_back(data_odd_dims,
                        grid_outer,
                        op::v9::GridSample::Attributes{true, GS_NEAREST, GS_ZEROS},
                        reference_tests::Tensor{{1, 1, 1, 7}, GRID_ET, std::vector<GT>{0, 0, 0, 0, 0, 0, 0}},
                        "nearest_zeros_align_odd_dims_outer" + types_str);

    params.emplace_back(data_odd_dims,
                        grid_outer,
                        op::v9::GridSample::Attributes{false, GS_NEAREST, GS_BORDER},
                        reference_tests::Tensor{{1, 1, 1, 7}, DATA_ET, std::vector<DT>{1, 11, 11, 14, 15, 10, 5}},
                        "nearest_border_noalign_odd_dims_outer" + types_str);

    params.emplace_back(data_odd_dims,
                        grid_outer,
                        op::v9::GridSample::Attributes{true, GS_NEAREST, GS_BORDER},
                        reference_tests::Tensor{{1, 1, 1, 7}, DATA_ET, std::vector<DT>{1, 6, 11, 14, 15, 10, 5}},
                        "nearest_border_align_odd_dims_outer" + types_str);

    params.emplace_back(data_odd_dims,
                        grid_outer,
                        op::v9::GridSample::Attributes{false, GS_NEAREST, GS_REFLECTION},
                        reference_tests::Tensor{{1, 1, 1, 7}, DATA_ET, std::vector<DT>{8, 14, 1, 4, 14, 6, 5}},
                        "nearest_reflection_noalign_odd_dims_outer" + types_str);

    params.emplace_back(data_odd_dims,
                        grid_outer,
                        op::v9::GridSample::Attributes{true, GS_NEAREST, GS_REFLECTION},
                        reference_tests::Tensor{{1, 1, 1, 7}, DATA_ET, std::vector<DT>{8, 9, 6, 4, 14, 6, 4}},
                        "nearest_reflection_align_odd_dims_outer" + types_str);

    return params;
}

template <ov::element::Type_t DATA_ET,
          ov::element::Type_t GRID_ET,
          class DT = ov::fundamental_type_for<DATA_ET>,
          class GT = ov::fundamental_type_for<GRID_ET>>
std::vector<GridSampleParams> generateNearestParamsEvenDimensions() {
    std::vector<GridSampleParams> params;
    reference_tests::Tensor data_even_dims{{1, 1, 4, 6}, DATA_ET, std::vector<DT>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                                  9,  10, 11, 12, 13, 14, 15, 16,
                                                                                  17, 18, 19, 20, 21, 22, 23, 24}};
    reference_tests::Tensor grid_inner{
        {1, 1, 8, 2},
        GRID_ET,
        std::vector<GT>{-0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -1, 1, 1, -1, -0.1, -0.1, 0.1, 0.1}};

    reference_tests::Tensor output_align{{1, 1, 1, 8}, DATA_ET, std::vector<DT>{8, 14, 11, 17, 19, 6, 9, 16}};
    reference_tests::Tensor output_noalign{{1, 1, 1, 8}, DATA_ET, std::vector<DT>{2, 14, 5, 17, 19, 6, 9, 16}};
    reference_tests::Tensor output_zeros_noalign{{1, 1, 1, 8}, DATA_ET, std::vector<DT>{2, 14, 5, 17, 0, 0, 9, 16}};

    for (const auto& padding : padding_modes) {
        std::stringstream name1, name2;
        name1 << "nearest_" << padding << "_noalign"
              << "_even_dims_inner" << param_types_str(DATA_ET, GRID_ET);
        params.emplace_back(data_even_dims,
                            grid_inner,
                            op::v9::GridSample::Attributes{false, GS_NEAREST, padding},
                            padding == GS_ZEROS ? output_zeros_noalign : output_noalign,
                            name1.str());

        name2 << "nearest_" << padding << "_align"
              << "_even_dims_inner" << param_types_str(DATA_ET, GRID_ET);
        params.emplace_back(data_even_dims,
                            grid_inner,
                            op::v9::GridSample::Attributes{true, GS_NEAREST, padding},
                            output_align,
                            name2.str());
    }

    return params;
}

template <ov::element::Type_t DATA_ET,
          ov::element::Type_t GRID_ET,
          class DT = ov::fundamental_type_for<DATA_ET>,
          class GT = ov::fundamental_type_for<GRID_ET>>
std::vector<GridSampleParams> generateBilinearParamsOddDimensionsInnerGrids() {
    const auto types_str = param_types_str(DATA_ET, GRID_ET);
    std::vector<GridSampleParams> params;
    reference_tests::Tensor data_odd_dims{{1, 1, 3, 5},
                                          DATA_ET,
                                          std::vector<DT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    reference_tests::Tensor grid_inner{{1, 3, 4, 2}, GRID_ET, std::vector<GT>{-0.1, -0.1, -0.1, 0.1,  0.1,  -0.1,
                                                                              0.1,  0.1,  -0.5, -0.5, -0.5, 0.5,
                                                                              0.5,  -0.5, 0.5,  0.5,  -1.,  -1.,
                                                                              -1.,  1.,   1.,   -1.,  1.,   1.}};

    reference_tests::Tensor output_align{{1, 1, 3, 4},
                                         DATA_ET,
                                         std::vector<DT>{7.3, 8.3, 7.7, 8.7, 4.5, 9.5, 6.5, 11.5, 1, 11, 5, 15}};
    reference_tests::Tensor output_noalign{{1, 1, 3, 4},
                                           DATA_ET,
                                           std::vector<DT>{7, 8.5, 7.5, 9, 3, 10.5, 5.5, 13, 1, 11, 5, 15}};
    reference_tests::Tensor output_zeros_noalign{
        {1, 1, 3, 4},
        DATA_ET,
        std::vector<DT>{7, 8.5, 7.5, 9, 3, 10.5, 5.5, 13, 0.25, 2.75, 1.25, 3.75}};

    params.emplace_back(data_odd_dims,
                        grid_inner,
                        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_ZEROS},
                        output_zeros_noalign,
                        "bilinear_zeros_noalign_odd_dims_inner" + types_str);

    params.emplace_back(data_odd_dims,
                        grid_inner,
                        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_ZEROS},
                        output_align,
                        "bilinear_zeros_align_odd_dims_inner" + types_str);

    params.emplace_back(data_odd_dims,
                        grid_inner,
                        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_BORDER},
                        output_noalign,
                        "bilinear_border_noalign_odd_dims_inner" + types_str);

    params.emplace_back(data_odd_dims,
                        grid_inner,
                        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_BORDER},
                        output_align,
                        "bilinear_border_align_odd_dims_inner" + types_str);

    params.emplace_back(data_odd_dims,
                        grid_inner,
                        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_REFLECTION},
                        output_noalign,
                        "bilinear_reflection_noalign_odd_dims_inner" + types_str);

    params.emplace_back(data_odd_dims,
                        grid_inner,
                        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_REFLECTION},
                        output_align,
                        "bilinear_reflection_align_odd_dims_inner" + types_str);

    return params;
}

template <ov::element::Type_t DATA_ET,
          ov::element::Type_t GRID_ET,
          class DT = ov::fundamental_type_for<DATA_ET>,
          class GT = ov::fundamental_type_for<GRID_ET>>
std::vector<GridSampleParams> generateBilinearParamsOddDimensionsOuterGrids() {
    const auto types_str = param_types_str(DATA_ET, GRID_ET);
    std::vector<GridSampleParams> params;

    reference_tests::Tensor data_odd_dims{{1, 1, 3, 5},
                                          DATA_ET,
                                          std::vector<DT>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};

    reference_tests::Tensor grid_outer{
        {1, 1, 7, 2},
        GRID_ET,
        std::vector<GT>{-10.1, -9.7, -7.55, 0.37, -77., 11.56, 0.5, 2.55, 1.7, 1.1, 3., -0.17, 1.301, -1.001}};

    params.emplace_back(data_odd_dims,
                        grid_outer,
                        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_ZEROS},
                        reference_tests::Tensor{{1, 1, 1, 7}, DATA_ET, std::vector<DT>{0, 0, 0, 0, 0, 0, 0}},
                        "bilinear_zeros_noalign_odd_dims_outer" + types_str);

    params.emplace_back(data_odd_dims,
                        grid_outer,
                        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_ZEROS},
                        reference_tests::Tensor{{1, 1, 1, 7}, DATA_ET, std::vector<DT>{0, 0, 0, 0, 0, 0, 1.9880099}},
                        "bilinear_zeros_align_odd_dims_outer" + types_str);

    params.emplace_back(
        data_odd_dims,
        grid_outer,
        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_BORDER},
        reference_tests::Tensor{{1, 1, 1, 7}, DATA_ET, std::vector<DT>{1, 8.775, 11, 14.25, 15, 8.725, 5}},
        "bilinear_border_noalign_odd_dims_outer" + types_str);

    params.emplace_back(data_odd_dims,
                        grid_outer,
                        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_BORDER},
                        reference_tests::Tensor{{1, 1, 1, 7}, DATA_ET, std::vector<DT>{1, 7.85, 11, 14, 15, 9.15, 5}},
                        "bilinear_border_align_odd_dims_outer" + types_str);

    params.emplace_back(
        data_odd_dims,
        grid_outer,
        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_REFLECTION},
        reference_tests::Tensor{{1, 1, 1, 7},
                                DATA_ET,
                                std::vector<DT>{5.9999995, 11.9, 2.7000031, 5.1250005, 13.75, 4.725, 4.7475}},
        "bilinear_reflection_noalign_odd_dims_outer" + types_str);

    params.emplace_back(
        data_odd_dims,
        grid_outer,
        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_REFLECTION},
        reference_tests::Tensor{{1, 1, 1, 7},
                                DATA_ET,
                                std::vector<DT>{6.7, 10.75, 3.800002, 6.25, 13.099999, 5.15, 4.4030004}},
        "bilinear_reflection_align_odd_dims_outer" + types_str);

    return params;
}

template <ov::element::Type_t DATA_ET,
          ov::element::Type_t GRID_ET,
          class DT = ov::fundamental_type_for<DATA_ET>,
          class GT = ov::fundamental_type_for<GRID_ET>>
std::vector<GridSampleParams> generateBilinearParamsEvenDimensions() {
    const auto types_str = param_types_str(DATA_ET, GRID_ET);
    std::vector<GridSampleParams> params;
    reference_tests::Tensor data_even_dims{{1, 1, 4, 6}, DATA_ET, std::vector<DT>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                                  9,  10, 11, 12, 13, 14, 15, 16,
                                                                                  17, 18, 19, 20, 21, 22, 23, 24}};
    reference_tests::Tensor grid_inner{
        {1, 1, 8, 2},
        GRID_ET,
        std::vector<GT>{-0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -1, 1, 1, -1, -0.1, -0.1, 0.1, 0.1}};

    params.emplace_back(
        data_even_dims,
        grid_inner,
        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_ZEROS},
        reference_tests::Tensor{{1, 1, 1, 8}, DATA_ET, std::vector<DT>{5, 17, 8, 20, 4.75, 1.5, 11, 14}},
        "bilinear_zeros_noalign_even_dims_inner" + types_str);

    params.emplace_back(
        data_even_dims,
        grid_inner,
        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_ZEROS},
        reference_tests::Tensor{{1, 1, 1, 8}, DATA_ET, std::vector<DT>{6.75, 15.75, 9.25, 18.25, 19, 6, 11.35, 13.65}},
        "bilinear_zeros_align_even_dims_inner" + types_str);

    params.emplace_back(data_even_dims,
                        grid_inner,
                        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_BORDER},
                        reference_tests::Tensor{{1, 1, 1, 8}, DATA_ET, std::vector<DT>{5, 17, 8, 20, 19, 6, 11, 14}},
                        "bilinear_border_noalign_even_dims_inner" + types_str);

    params.emplace_back(
        data_even_dims,
        grid_inner,
        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_BORDER},
        reference_tests::Tensor{{1, 1, 1, 8}, DATA_ET, std::vector<DT>{6.75, 15.75, 9.25, 18.25, 19, 6, 11.35, 13.65}},
        "bilinear_border_align_even_dims_inner" + types_str);

    params.emplace_back(data_even_dims,
                        grid_inner,
                        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_REFLECTION},
                        reference_tests::Tensor{{1, 1, 1, 8}, DATA_ET, std::vector<DT>{5, 17, 8, 20, 19, 6, 11, 14}},
                        "bilinear_reflection_noalign_even_dims_inner" + types_str);

    params.emplace_back(
        data_even_dims,
        grid_inner,
        op::v9::GridSample::Attributes{true, GS_BILINEAR, GS_REFLECTION},
        reference_tests::Tensor{{1, 1, 1, 8}, DATA_ET, std::vector<DT>{6.75, 15.75, 9.25, 18.25, 19, 6, 11.35, 13.65}},
        "bilinear_reflection_align_even_dims_inner" + types_str);

    return params;
}

template <ov::element::Type_t DATA_ET, class DT = ov::fundamental_type_for<DATA_ET>>
std::vector<GridSampleParams> generateBicubicParams() {
    constexpr auto GRID_ET = ov::element::Type_t::f32;
    using GT = ov::fundamental_type_for<GRID_ET>;
    const auto types_str = param_types_str(DATA_ET, GRID_ET);
    std::vector<GridSampleParams> params;

    // clang-format off
    reference_tests::Tensor data_even_dims{{1, 1, 4, 7}, DATA_ET,
            std::vector<DT>{1, 1, 1, 1, 1, 1, 1,
                            1, 2, 2, 2, 2, 2, 1,
                            1, 2, 3, 5, 3, 2, 1,
                            1, 2, 5, 9, 5, 2, 1}};
    reference_tests::Tensor grid{
        {1, 4, 4, 2},
        GRID_ET,
            std::vector<GT>{-0.1, -0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1,
                            -0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
                            -0.9, -0.9, -0.9, 0.9, 0.9, -0.9, 0.9, 0.9,
                            -1.75, 0.7, 1.33, -1.11, 0.965, 1.007, 21, 37}};

    params.emplace_back(data_even_dims,
                        grid,
                        op::v9::GridSample::Attributes{false, GS_BICUBIC, GS_ZEROS},
                        reference_tests::Tensor{{1, 1, 4, 4},
                                                DATA_ET,
                                                std::vector<DT>{2.6663566, 3.527928,   2.6663566,  3.527928,
                                                                1.6318359, 2.7156982,  1.6318359,  2.7156982,
                                                                0.6378987, 0.57033366, 0.6378987,  0.57033366,
                                                                0,        -0.01507522, 0.25528803, 0 }},
                        "bicubic_zeros_noalign" + types_str);

    params.emplace_back(data_even_dims,
                        grid,
                        op::v9::GridSample::Attributes{true, GS_BICUBIC, GS_ZEROS},
                        reference_tests::Tensor{{1, 1, 4, 4},
                                                DATA_ET,
                                                std::vector<DT>{2.7887204, 3.4506166,  2.7887204, 3.4506166,
                                                                1.8481445, 2.7364502,  1.8481445, 2.7364502,
                                                                1.2367951, 1.3602872,  1.2367951, 1.3602872,
                                                                0,         0.00650583, 1.1182348, 0 }},
                        "bicubic_zeros_align" + types_str);

    params.emplace_back(data_even_dims,
                        grid,
                        op::v9::GridSample::Attributes{false, GS_BICUBIC, GS_BORDER},
                        reference_tests::Tensor{{1, 1, 4, 4},
                                                DATA_ET,
                                                std::vector<DT>{2.6663566, 3.527928,   2.6663566, 3.527928,
                                                                1.5380859, 2.4677734,  1.5380859, 2.4677734,
                                                                1.0089612, 0.91871876, 1.0089612, 0.91871876,
                                                                1,         1,          0.8902873, 1 }},
                        "bicubic_border_noalign" + types_str);

    params.emplace_back(data_even_dims,
                        grid,
                        op::v9::GridSample::Attributes{true, GS_BICUBIC, GS_BORDER},
                        reference_tests::Tensor{{1, 1, 4, 4},
                                                DATA_ET,
                                                std::vector<DT>{2.7887204, 3.4506166, 2.7887204, 3.4506166,
                                                                1.8129883, 2.623291,  1.8129883, 2.623291,
                                                                1.0363026, 1.1486388, 1.0363026, 1.1486388,
                                                                1,         1.0000064, 1.0641243, 1 }},
                        "bicubic_border_align" + types_str);

    params.emplace_back(data_even_dims,
                        grid,
                        op::v9::GridSample::Attributes{false, GS_BICUBIC, GS_REFLECTION},
                        reference_tests::Tensor{{1, 1, 4, 4},
                                                DATA_ET,
                                                std::vector<DT>{2.6663566, 3.527928,  2.6663566, 3.527928,
                                                                1.5380859, 2.4677734, 1.5380859, 2.4677734,
                                                                1.0150609, 0.904375,  1.0150609, 0.904375,
                                                                5.48851,   0.898316,  0.8237547, 0.8125 }},
                        "bicubic_reflection_noalign" + types_str);

    params.emplace_back(data_even_dims,
                        grid,
                        op::v9::GridSample::Attributes{true, GS_BICUBIC, GS_REFLECTION},
                        reference_tests::Tensor{{1, 1, 4, 4},
                                                DATA_ET,
                                                std::vector<DT>{2.7887204, 3.4506166, 2.7887204, 3.4506166,
                                                                1.7745361, 2.6518555, 1.7745361, 2.6518555,
                                                                1.0085088, 1.0307077, 1.0085088, 1.0307077,
                                                                5.5649586, 1.0553409, 1.0011607, 1 }},
                        "bicubic_reflection_align" + types_str);
    // clang-format on

    return params;
}

template <ov::element::Type_t DATA_ET,
          ov::element::Type_t GRID_ET,
          class DT = ov::fundamental_type_for<DATA_ET>,
          class GT = ov::fundamental_type_for<GRID_ET>>
std::vector<GridSampleParams> generateBicubicBatchesParams() {
    const auto types_str = param_types_str(DATA_ET, GRID_ET);
    std::vector<GridSampleParams> params;

    reference_tests::Tensor data{{2, 2, 4, 3},
                                 DATA_ET,
                                 std::vector<DT>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                                 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48}};
    reference_tests::Tensor grid{
        {2, 2, 4, 2},
        GRID_ET,
        std::vector<GT>{-0.1, -0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.5,  -0.5, -0.5, 0.5,   0.5,   -0.5,  0.5, 0.5,
                        -0.9, -0.9, -0.9, 0.9, 0.9, -0.9, 0.9, 0.9, -1.75, 0.7,  1.33, -1.11, 0.965, 1.007, 21,  37}};

    params.emplace_back(
        data,
        grid,
        op::v9::GridSample::Attributes{true, GS_BICUBIC, GS_BORDER},
        reference_tests::Tensor{
            {2, 2, 2, 4},
            DATA_ET,
            std::vector<DT>{6.0096254, 6.7048755, 6.2951245, 6.9903746, 3.4101562, 8.402344,  4.5976562, 9.589844,
                            18.009624, 18.704876, 18.295124, 18.990376, 15.410156, 20.402344, 16.597656, 21.589844,
                            25.415281, 33.735218, 27.26478,  35.58472,  32.884,    26.852259, 35.996872, 36.,
                            37.41528,  45.735218, 39.264782, 47.58472,  44.884,    38.852257, 47.996872, 48.}},
        "bicubic_border_align_batches" + types_str);

    params.emplace_back(
        data,
        grid,
        op::v9::GridSample::Attributes{false, GS_BICUBIC, GS_REFLECTION},
        reference_tests::Tensor{
            {2, 2, 2, 4},
            DATA_ET,
            std::vector<DT>{5.8170314, 6.7650313, 6.2349687, 7.182969,  2.4101562, 8.972656,  4.0273438, 10.589844,
                            17.81703,  18.765032, 18.234968, 19.18297,  14.410156, 20.972656, 16.027344, 22.589844,
                            24.356874, 34.301876, 26.698126, 36.643124, 34.304035, 26.55013,  36.74749,  36.75,
                            36.356876, 46.301876, 38.698124, 48.643124, 46.304035, 38.55013,  48.74749,  48.75}},
        "bicubic_reflection_noalign_batches" + types_str);

    return params;
}

template <ov::element::Type_t DATA_ET,
          ov::element::Type_t GRID_ET,
          class DT = ov::fundamental_type_for<DATA_ET>,
          class GT = ov::fundamental_type_for<GRID_ET>>
std::vector<GridSampleParams> generateCornerCaseData1x1Params() {
    const auto types_str = param_types_str(DATA_ET, GRID_ET);
    std::vector<GridSampleParams> params;

    const reference_tests::Tensor data{{1, 1, 1, 1}, DATA_ET, std::vector<DT>{7}};
    const reference_tests::Tensor grid{{1, 1, 5, 2}, GRID_ET, std::vector<GT>{1, -1, 0, 0, -1, 0, 0.5, 0.5, 2, -4}};
    const reference_tests::Tensor sevens{{1, 1, 1, 5}, DATA_ET, std::vector<DT>{7, 7, 7, 7, 7}};

    params.emplace_back(data,
                        grid,
                        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_ZEROS},
                        reference_tests::Tensor{{1, 1, 1, 5}, DATA_ET, std::vector<DT>{1.75, 7, 3.5, 3.9375, 0}},
                        "bilinear_zeros_no_align_data1x1" + types_str);

    params.emplace_back(data,
                        grid,
                        op::v9::GridSample::Attributes{false, GS_NEAREST, GS_ZEROS},
                        reference_tests::Tensor{{1, 1, 1, 5}, DATA_ET, std::vector<DT>{7, 7, 7, 7, 0}},
                        "nearest_zeros_no_align_data1x1" + types_str);

    params.emplace_back(
        data,
        grid,
        op::v9::GridSample::Attributes{false, GS_BICUBIC, GS_ZEROS},
        reference_tests::Tensor{{1, 1, 1, 5}, DATA_ET, std::vector<DT>{2.4677734, 7, 4.15625, 5.4073334, 0}},
        "bicubic_zeros_no_align_data1x1" + types_str);

    params.emplace_back(data,
                        grid,
                        op::v9::GridSample::Attributes{true, GS_BICUBIC, GS_ZEROS},
                        sevens,
                        "bicubic_zeros_align_data1x1" + types_str);

    params.emplace_back(data,
                        grid,
                        op::v9::GridSample::Attributes{false, GS_BILINEAR, GS_REFLECTION},
                        sevens,
                        "bilinear_reflection_noalign_data1x1" + types_str);

    params.emplace_back(data,
                        grid,
                        op::v9::GridSample::Attributes{true, GS_NEAREST, GS_BORDER},
                        sevens,
                        "nearest_border_align_data1x1" + types_str);

    return params;
}

std::vector<GridSampleParams> generateGridSampleParams() {
    using namespace ov::element;
    std::vector<std::vector<GridSampleParams>> combo_params{generateNearestParamsOddDimensionsInnerGrids<f32, f32>(),
                                                            generateNearestParamsOddDimensionsInnerGrids<f32, f16>(),
                                                            generateNearestParamsOddDimensionsInnerGrids<f16, f32>(),

                                                            generateNearestParamsEvenDimensions<f32, f32>(),
                                                            generateNearestParamsEvenDimensions<f32, f16>(),
                                                            generateNearestParamsEvenDimensions<f16, f32>(),

                                                            generateBilinearParamsOddDimensionsInnerGrids<f32, f32>(),
                                                            generateBilinearParamsOddDimensionsInnerGrids<f32, f16>(),
                                                            generateBilinearParamsOddDimensionsInnerGrids<f16, f32>(),

                                                            generateBilinearParamsOddDimensionsOuterGrids<f32, f32>(),
                                                            generateBilinearParamsOddDimensionsOuterGrids<f32, f16>(),
                                                            generateBilinearParamsOddDimensionsOuterGrids<f16, f32>(),

                                                            generateBilinearParamsEvenDimensions<f32, f32>(),
                                                            generateBilinearParamsEvenDimensions<f32, f16>(),
                                                            generateBilinearParamsEvenDimensions<f16, f32>(),

                                                            generateBicubicParams<f32>(),
                                                            generateBicubicParams<f64>(),
                                                            generateBicubicParams<f16>(),

                                                            generateBicubicBatchesParams<f32, f32>(),
                                                            generateBicubicBatchesParams<f32, f16>(),
                                                            generateBicubicBatchesParams<f16, f32>(),

                                                            generateCornerCaseData1x1Params<f32, f32>(),
                                                            generateCornerCaseData1x1Params<f32, bf16>(),
                                                            generateCornerCaseData1x1Params<f32, f16>()};
    std::vector<GridSampleParams> test_params;
    for (auto& params : combo_params)
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
