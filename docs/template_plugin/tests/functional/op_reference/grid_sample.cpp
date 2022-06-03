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
        auto params = GetParam();
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

std::array<op::v9::GridSample::PaddingMode, 3> padding_modes{op::v9::GridSample::PaddingMode::ZEROS,
                                                             op::v9::GridSample::PaddingMode::BORDER,
                                                             op::v9::GridSample::PaddingMode::REFLECTION};
std::array<bool, 2> align_modes{false, true};

std::vector<GridSampleParams> generateGridSampleNearestParams() {
    std::vector<GridSampleParams> nearest_params;
    {
        reference_tests::Tensor data{{1, 1, 3, 5},
                                     element::f32,
                                     std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
        reference_tests::Tensor grid{{1, 3, 4, 2}, element::f32, std::vector<float>{-0.1, -0.1, -0.1, 0.1,  0.1,  -0.1,
                                                                                    0.1,  0.1,  -0.5, -0.5, -0.5, 0.5,
                                                                                    0.5,  -0.5, 0.5,  0.5,  -1.,  -1.,
                                                                                    -1.,  1.,   1.,   -1.,  1.,   1.}};
        reference_tests::Tensor output{{1, 1, 3, 4},
                                       element::f32,
                                       std::vector<float>{8, 8, 8, 8, 2, 12, 4, 14, 1, 11, 5, 15}};
        for (const auto& padding : padding_modes) {
            for (const auto align : align_modes) {
                std::stringstream name;
                name << std::boolalpha << "nearest_" << padding << "_" << align << "_odd_dims";
                nearest_params.emplace_back(
                    data,
                    grid,
                    op::v9::GridSample::Attributes{align, op::v9::GridSample::InterpolationMode::NEAREST, padding},
                    output,
                    name.str());
            }
        }
    }
    return nearest_params;
}

std::vector<GridSampleParams> generateGridSampleParams() {
    std::vector<std::vector<GridSampleParams>> all_params{generateGridSampleNearestParams()};
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
