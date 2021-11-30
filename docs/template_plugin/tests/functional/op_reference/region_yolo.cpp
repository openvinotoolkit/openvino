// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "openvino/op/region_yolo.hpp"
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
          refData(CreateTensor(iType, oValues)),
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
    ov::runtime::Tensor inputData;
    ov::runtime::Tensor refData;
    std::string testcaseName;

public:
    template<typename T>
    inline static std::vector<T> read_binary_file(const std::string& filename) {
        std::string path = ov::util::path_join({TEST_FILES, filename});
        std::vector<float> file_content;
        std::vector<T> type_converted_data;
        std::ifstream inputs_fs{path, std::ios::in | std::ios::binary};
        if (!inputs_fs) {
            throw std::runtime_error("Failed to open the file: " + path);
        }

        inputs_fs.seekg(0, std::ios::end);
        auto size = inputs_fs.tellg();
        inputs_fs.seekg(0, std::ios::beg);
        if (size % sizeof(float) != 0) {
            throw std::runtime_error("Error reading binary file content: Input file size (in bytes) "
                                    "is not a multiple of requested data type size.");
        }
        file_content.resize(size / sizeof(float));
        type_converted_data.resize(size / sizeof(float));
        inputs_fs.read(reinterpret_cast<char*>(file_content.data()), size);

        std::transform(file_content.begin(), file_content.end(), type_converted_data.begin(), [](float x) { return (T)x;});
        return type_converted_data;
    }
};

class ReferenceRegionYoloLayerTest : public testing::TestWithParam<RegionYoloParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RegionYoloParams>& obj) {
        auto param = obj.param;
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
    static std::shared_ptr<Function> CreateFunction(const RegionYoloParams& params) {
        const auto p = std::make_shared<op::v0::Parameter>(params.inType, params.inputShape);
        const auto RegionYolo = std::make_shared<op::v0::RegionYolo>(p,
                                                                     params.coords,
                                                                     params.classes,
                                                                     params.num,
                                                                     params.do_softmax,
                                                                     params.mask,
                                                                     params.axis,
                                                                     params.end_axis);
        return std::make_shared<ov::Function>(NodeVector {RegionYolo}, ParameterVector {p});
    }
};

TEST_P(ReferenceRegionYoloLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<RegionYoloParams> generateRegionYoloParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RegionYoloParams> regionYoloParams {
        RegionYoloParams(5, 4, 20, true, 1, 3,
                        1, 125, 13, 13,
                        std::vector<int64_t>{0, 1, 2},
                        IN_ET,
                        RegionYoloParams::read_binary_file<T>("region_in_yolov2_caffe.data"),
                        RegionYoloParams::read_binary_file<T>("region_out_yolov2_caffe.data"),
                        "region_yolo_v2_caffe"),
        RegionYoloParams(9, 4, 20, false, 1, 3,
                        1, 75, 32, 32,
                        std::vector<int64_t>{0, 1, 2},
                        IN_ET,
                        RegionYoloParams::read_binary_file<T>("region_in_yolov3_mxnet.data"),
                        RegionYoloParams::read_binary_file<T>("region_out_yolov3_mxnet.data"),
                        "region_yolo_v3_mxnet"),
        RegionYoloParams(1, 4, 1, false, 1, 3,
                        1, 8, 2, 2,
                        std::vector<int64_t>{0},
                        IN_ET,
                        std::vector<T>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.1f, 0.2f, 0.3f,
                                       0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                                       0.7f, 0.8f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f},
                        std::vector<T>{0.52497f, 0.54983f, 0.57444f, 0.59868f, 0.62245f, 0.64565f, 0.66818f, 0.68997f,
                                       0.1f,     0.2f,     0.3f,     0.4f,     0.5f,     0.6f,     0.7f,     0.8f,
                                       0.52497f, 0.54983f, 0.57444f, 0.59868f, 0.62245f, 0.64565f, 0.66818f, 0.68997f}),
    };
    return regionYoloParams;
}

std::vector<RegionYoloParams> generateRegionYoloCombinedParams() {
    const std::vector<std::vector<RegionYoloParams>> regionYoloTypeParams {
        generateRegionYoloParams<element::Type_t::f64>(),
        generateRegionYoloParams<element::Type_t::f32>(),
        generateRegionYoloParams<element::Type_t::f16>(),
        generateRegionYoloParams<element::Type_t::bf16>()
        };
    std::vector<RegionYoloParams> combinedParams;

    for (const auto& params : regionYoloTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_RegionYolo_With_Hardcoded_Refs, ReferenceRegionYoloLayerTest,
    testing::ValuesIn(generateRegionYoloCombinedParams()), ReferenceRegionYoloLayerTest::getTestCaseName);

} // namespace