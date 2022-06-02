// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grid_sample.hpp"

#include "base_reference_test.hpp"
#include "gtest/gtest.h"

using namespace ov;
using namespace reference_tests;

namespace reference_tests {
namespace {

struct GridSampleParams {
    GridSampleParams(const reference_tests::Tensor& data,
                     const reference_tests::Tensor& grid,
                     const op::v9::GridSample::Attributes& attrs,
                     const reference_tests::Tensor& expected,
                     const std::string& name,
                     bool dyn_shape = false)
        : data_tensor{data},
          grid_tensor{grid},
          attributes{attrs},
          expected_tensor{expected},
          test_case_name{name},
          dynamic_shape{dyn_shape} {}

    reference_tests::Tensor data_tensor;
    reference_tests::Tensor grid_tensor;
    op::v9::GridSample::Attributes attributes;
    reference_tests::Tensor expected_tensor;
    bool dynamic_shape;
    std::string test_case_name;
};

class ReferenceGridSampleLayerTest : public testing::TestWithParam<GridSampleParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.data_tensor, params.grid_tensor, params.attributes, params.dynamic_shape);
        inputData = {params.data_tensor.data, params.grid_tensor.data};
        refOutData = {params.expected_tensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GridSampleParams>& obj) {
        return obj.param.test_case_name + (obj.param.dynamic_shape ? "_dyn_shape_inputs" : "");
    }

private:
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& data,
                                                 const reference_tests::Tensor& grid,
                                                 const op::v9::GridSample::Attributes& attributes,
                                                 bool dynamic_shape = false) {
        const auto in1 =
            std::make_shared<op::v0::Parameter>(data.type, dynamic_shape ? PartialShape::dynamic() : data.shape);
        const auto in2 =
            std::make_shared<op::v0::Parameter>(grid.type, dynamic_shape ? PartialShape::dynamic() : grid.shape);
        const auto grid_sample = std::make_shared<op::v9::GridSample>(in1, in2, attributes);

        return std::make_shared<Model>(NodeVector{grid_sample}, ParameterVector{in1, in2});
    }
};
}  // namespace

op::v9::GridSample::Attributes makeGridSampleAttributes(bool align_corners,
                                                        std::string mode,
                                                        std::string padding_mode) {
    op::v9::GridSample::Attributes attributes;
    attributes.align_corners = align_corners;
    // attributes.mode = mode;
    // attributes.padding_mode = padding_mode;
    return attributes;
}

std::vector<GridSampleParams> generateGridSampleParams() {
    // op::v9::GridSample::Attributes attributes;
    // attributes.align_corners = true;
    // attributes.mode =
    // attributes.p

    using ::reference_tests::Tensor;
    std::vector<GridSampleParams> test_params{
        GridSampleParams(Tensor{{1, 1, 1, 1}, element::f32, std::vector<float>{7}},
                         Tensor{{1, 2, 2, 2}, element::f32, std::vector<float>{-1, -1, 1, -1, 1, 1, -1, 1}},
                         makeGridSampleAttributes(true, "", ""),
                         Tensor{{1, 1, 2, 2}, element::f32, std::vector<float>{7, 7, 7, 7}},
                         "tj_test_1",
                         false),
        GridSampleParams(Tensor{{1, 1, 1, 1}, element::u32, std::vector<uint32_t>{7}},
                         Tensor{{1, 2, 3, 2}, element::f32, std::vector<float>{-1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1}},
                         makeGridSampleAttributes(true, "", ""),
                         Tensor{{1, 1, 3, 2}, element::u32, std::vector<uint32_t>{7, 7, 7, 7, 7, 7}},
                         "tj_test_2",
                         false),
        GridSampleParams(Tensor{{1, 1, 1, 1}, element::i64, std::vector<int64_t>{8}},
                         Tensor{{1, 1, 1, 2}, element::f32, std::vector<float>{0, 0}},
                         makeGridSampleAttributes(false, "", ""),
                         Tensor{{1, 1, 1, 1}, element::i64, std::vector<int64_t>{4}},
                         "tj_test_3",
                         false)};

    return test_params;
}
}  // namespace reference_tests

TEST_P(ReferenceGridSampleLayerTest, GridSample) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke_GridSample,
                         ReferenceGridSampleLayerTest,
                         ::testing::ValuesIn(generateGridSampleParams()),
                         ReferenceGridSampleLayerTest::getTestCaseName);
