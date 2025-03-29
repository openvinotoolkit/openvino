// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/region_yolo.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "base_reference_test.hpp"
#include "openvino/util/file_util.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct RegionYoloParams {
    template <class IT>
    RegionYoloParams(const size_t num,
                     const size_t coords,
                     const size_t classes,
                     const bool do_softmax,
                     const int axis,
                     const int end_axis,
                     const size_t batch,
                     const size_t channels,
                     const size_t width,
                     const size_t height,
                     const std::vector<int64_t>& mask,
                     const ov::element::Type& iType,
                     const std::vector<IT>& iValues,
                     const Shape& output_shape,
                     const std::vector<IT>& oValues,
                     const std::string& testcaseName = "")
        : num(num),
          coords(coords),
          classes(classes),
          do_softmax(do_softmax),
          axis(axis),
          end_axis(end_axis),
          batch(batch),
          channels(channels),
          width(width),
          height(height),
          mask(mask),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(output_shape, iType, oValues)),
          testcaseName(testcaseName) {
        inputShape = Shape{batch, channels, height, width};
    }

    size_t num;
    size_t coords;
    size_t classes;
    bool do_softmax;
    int axis;
    int end_axis;
    size_t batch;
    size_t channels;
    size_t width;
    size_t height;
    std::vector<int64_t> mask;
    ov::PartialShape inputShape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
    std::string testcaseName;
};

class ReferenceRegionYoloLayerTest : public testing::TestWithParam<RegionYoloParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RegionYoloParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "inputShape=" << param.inputShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "coords=" << param.coords << "_";
        result << "classes=" << param.classes << "_";
        result << "num=" << param.num << "_";
        result << "do_softmax=" << param.do_softmax << "_";
        result << "mask=" << param.mask << "_";
        result << "axis=" << param.axis << "_";
        result << "end_axis=" << param.end_axis;
        if (param.testcaseName != "")
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    std::shared_ptr<Model> CreateFunction(const RegionYoloParams& params) {
        const auto p = std::make_shared<op::v0::Parameter>(params.inType, params.inputShape);
        const auto RegionYolo = std::make_shared<op::v0::RegionYolo>(p,
                                                                     params.coords,
                                                                     params.classes,
                                                                     params.num,
                                                                     params.do_softmax,
                                                                     params.mask,
                                                                     params.axis,
                                                                     params.end_axis);
        return std::make_shared<ov::Model>(NodeVector{RegionYolo}, ParameterVector{p});
    }
};

TEST_P(ReferenceRegionYoloLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<RegionYoloParams> generateRegionYoloParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RegionYoloParams> regionYoloParams{
        RegionYoloParams(
            1,
            4,
            1,
            false,
            1,
            3,
            1,
            8,
            2,
            2,
            std::vector<int64_t>{0},
            IN_ET,
            std::vector<T>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.1f, 0.2f, 0.3f,
                           0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                           0.7f, 0.8f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f},
            Shape{1, 6, 2, 2},
            std::vector<T>{0.52497f, 0.54983f, 0.57444f, 0.59868f, 0.62245f, 0.64565f, 0.66818f, 0.68997f,
                           0.1f,     0.2f,     0.3f,     0.4f,     0.5f,     0.6f,     0.7f,     0.8f,
                           0.52497f, 0.54983f, 0.57444f, 0.59868f, 0.62245f, 0.64565f, 0.66818f, 0.68997f}),
    };
    return regionYoloParams;
}

std::vector<RegionYoloParams> generateRegionYoloCombinedParams() {
    const std::vector<std::vector<RegionYoloParams>> regionYoloTypeParams{
        generateRegionYoloParams<element::Type_t::f64>(),
        generateRegionYoloParams<element::Type_t::f32>(),
        generateRegionYoloParams<element::Type_t::f16>(),
        generateRegionYoloParams<element::Type_t::bf16>()};
    std::vector<RegionYoloParams> combinedParams;

    for (const auto& params : regionYoloTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_RegionYolo_With_Hardcoded_Refs,
                         ReferenceRegionYoloLayerTest,
                         testing::ValuesIn(generateRegionYoloCombinedParams()),
                         ReferenceRegionYoloLayerTest::getTestCaseName);

}  // namespace
