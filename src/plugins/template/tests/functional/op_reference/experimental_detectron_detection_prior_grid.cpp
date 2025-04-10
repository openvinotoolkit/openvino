// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/experimental_detectron_prior_grid_generator.hpp"

using namespace ov;

using reference_tests::CommonReferenceTest;
using reference_tests::CreateTensor;
using Attrs = op::v6::ExperimentalDetectronPriorGridGenerator::Attributes;

namespace {
struct ExperimentalPGGParams {
    template <class IT>
    ExperimentalPGGParams(const Attrs& attrs,
                          const PartialShape& priorsShape,
                          const PartialShape& featureMapShape,
                          const PartialShape& imageSizeInfoShape,
                          const Shape& outRefShape,
                          const element::Type& iType,
                          const std::vector<IT>& priorsValues,
                          const std::vector<IT>& refValues,
                          const std::string& testcaseName = "")
        : attrs(attrs),
          priorsShape(priorsShape),
          featureMapShape(featureMapShape),
          imageSizeInfoShape(imageSizeInfoShape),
          outRefShape(outRefShape),
          inType(iType),
          priorsData(CreateTensor(iType, priorsValues)),
          refData(CreateTensor(outRefShape, iType, refValues)),
          testcaseName(testcaseName) {
        std::vector<IT> featureMapValues(shape_size(featureMapShape.get_shape()));
        std::iota(featureMapValues.begin(), featureMapValues.end(), 0.f);
        featureMapData = CreateTensor(iType, featureMapValues);

        std::vector<IT> imageSizeInfoValues(shape_size(imageSizeInfoShape.get_shape()));
        std::iota(imageSizeInfoValues.begin(), imageSizeInfoValues.end(), 0.f);
        imageSizeInfoData = CreateTensor(iType, imageSizeInfoValues);

        if (shape_size(outRefShape) > refValues.size())
            actualComparisonSize = refValues.size();
        else
            actualComparisonSize = 0;
    }

    Attrs attrs;
    PartialShape priorsShape;
    PartialShape featureMapShape;
    PartialShape imageSizeInfoShape;
    Shape outRefShape;
    size_t actualComparisonSize;
    element::Type inType;
    Tensor priorsData;
    Tensor featureMapData;
    Tensor imageSizeInfoData;
    Tensor refData;
    std::string testcaseName;
};

class ReferenceExperimentalPGGLayerTest : public testing::TestWithParam<ExperimentalPGGParams>,
                                          public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.priorsData, params.featureMapData, params.imageSizeInfoData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalPGGParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "priorsShape=" << param.priorsShape << "_";
        result << "featureMapShape=" << param.featureMapShape << "_";
        result << "imageSizeInfoShape=" << param.imageSizeInfoShape << "_";
        result << "iType=" << param.inType << "_";
        result << "flatten=" << param.attrs.flatten << "_";
        result << "h=" << param.attrs.h << "_";
        result << "w=" << param.attrs.w << "_";
        result << "stride_x=" << param.attrs.stride_x << "_";
        result << "stride_y=" << param.attrs.stride_y;
        if (param.testcaseName != "")
            result << "_" << param.testcaseName;
        return result.str();
    }

    virtual void Validate() override {
        if (const auto comparison_size = GetParam().actualComparisonSize) {
            ASSERT_EQ(executableNetwork.outputs().size(), refOutData.size());

            actualOutData.clear();
            actualOutData.emplace_back(inferRequest.get_tensor(executableNetwork.output(0)));

            // Shape matters: the trick is that hard-coded expected data is "shorter" than runtime inferred data. This
            // is due to huge size of the tensor and in such case the test provides a part of reference values for
            // comparison to avoid huge file size.
            ASSERT_EQ(refOutData[0].get_shape(), actualOutData[0].get_shape());
            const auto shape = Shape{comparison_size};
            const auto expected = Tensor{refOutData[0].get_element_type(), shape, refOutData[0].data()};
            const auto inferred = Tensor{actualOutData[0].get_element_type(), shape, actualOutData[0].data()};
            ValidateBlobs(expected, inferred, 0, threshold, abs_threshold, false);
        } else {
            CommonReferenceTest::Validate();
        }
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ExperimentalPGGParams& params) {
        const auto priors = std::make_shared<op::v0::Parameter>(params.inType, params.priorsShape);
        const auto featureMap = std::make_shared<op::v0::Parameter>(params.inType, params.featureMapShape);
        const auto im_info = std::make_shared<op::v0::Parameter>(params.inType, params.imageSizeInfoShape);
        const auto ExperimentalPGG = std::make_shared<op::v6::ExperimentalDetectronPriorGridGenerator>(priors,
                                                                                                       featureMap,
                                                                                                       im_info,
                                                                                                       params.attrs);
        return std::make_shared<Model>(NodeVector{ExperimentalPGG}, ParameterVector{priors, featureMap, im_info});
    }
};

TEST_P(ReferenceExperimentalPGGLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ExperimentalPGGParams> generateExperimentalPGGFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ExperimentalPGGParams> experimentalPGGParams{
        ExperimentalPGGParams(
            Attrs{true, 0, 0, 4.0f, 4.0f},
            {3, 4},
            {1, 16, 4, 5},
            {1, 3, 100, 200},
            {60, 4},
            IN_ET,
            std::vector<T>{-24.5, -12.5, 24.5, 12.5, -16.5, -16.5, 16.5, 16.5, -12.5, -24.5, 12.5, 24.5},
            std::vector<T>{-22.5, -10.5, 26.5,  14.5,  -14.5, -14.5, 18.5,  18.5,  -10.5, -22.5, 14.5,  26.5,  -18.5,
                           -10.5, 30.5,  14.5,  -10.5, -14.5, 22.5,  18.5,  -6.5,  -22.5, 18.5,  26.5,  -14.5, -10.5,
                           34.5,  14.5,  -6.5,  -14.5, 26.5,  18.5,  -2.5,  -22.5, 22.5,  26.5,  -10.5, -10.5, 38.5,
                           14.5,  -2.5,  -14.5, 30.5,  18.5,  1.5,   -22.5, 26.5,  26.5,  -6.5,  -10.5, 42.5,  14.5,
                           1.5,   -14.5, 34.5,  18.5,  5.5,   -22.5, 30.5,  26.5,  -22.5, -6.5,  26.5,  18.5,  -14.5,
                           -10.5, 18.5,  22.5,  -10.5, -18.5, 14.5,  30.5,  -18.5, -6.5,  30.5,  18.5,  -10.5, -10.5,
                           22.5,  22.5,  -6.5,  -18.5, 18.5,  30.5,  -14.5, -6.5,  34.5,  18.5,  -6.5,  -10.5, 26.5,
                           22.5,  -2.5,  -18.5, 22.5,  30.5,  -10.5, -6.5,  38.5,  18.5,  -2.5,  -10.5, 30.5,  22.5,
                           1.5,   -18.5, 26.5,  30.5,  -6.5,  -6.5,  42.5,  18.5,  1.5,   -10.5, 34.5,  22.5,  5.5,
                           -18.5, 30.5,  30.5,  -22.5, -2.5,  26.5,  22.5,  -14.5, -6.5,  18.5,  26.5,  -10.5, -14.5,
                           14.5,  34.5,  -18.5, -2.5,  30.5,  22.5,  -10.5, -6.5,  22.5,  26.5,  -6.5,  -14.5, 18.5,
                           34.5,  -14.5, -2.5,  34.5,  22.5,  -6.5,  -6.5,  26.5,  26.5,  -2.5,  -14.5, 22.5,  34.5,
                           -10.5, -2.5,  38.5,  22.5,  -2.5,  -6.5,  30.5,  26.5,  1.5,   -14.5, 26.5,  34.5,  -6.5,
                           -2.5,  42.5,  22.5,  1.5,   -6.5,  34.5,  26.5,  5.5,   -14.5, 30.5,  34.5,  -22.5, 1.5,
                           26.5,  26.5,  -14.5, -2.5,  18.5,  30.5,  -10.5, -10.5, 14.5,  38.5,  -18.5, 1.5,   30.5,
                           26.5,  -10.5, -2.5,  22.5,  30.5,  -6.5,  -10.5, 18.5,  38.5,  -14.5, 1.5,   34.5,  26.5,
                           -6.5,  -2.5,  26.5,  30.5,  -2.5,  -10.5, 22.5,  38.5,  -10.5, 1.5,   38.5,  26.5,  -2.5,
                           -2.5,  30.5,  30.5,  1.5,   -10.5, 26.5,  38.5,  -6.5,  1.5,   42.5,  26.5,  1.5,   -2.5,
                           34.5,  30.5,  5.5,   -10.5, 30.5,  38.5}),
        ExperimentalPGGParams(
            Attrs{false, 0, 0, 8.0f, 8.0f},
            {3, 4},
            {1, 16, 3, 7},
            {1, 3, 100, 200},
            {3, 7, 3, 4},
            IN_ET,
            std::vector<T>{-44.5, -24.5, 44.5, 24.5, -32.5, -32.5, 32.5, 32.5, -24.5, -44.5, 24.5, 44.5},
            std::vector<T>{
                -40.5, -20.5, 48.5, 28.5, -28.5, -28.5, 36.5, 36.5, -20.5, -40.5, 28.5, 48.5, -32.5, -20.5, 56.5, 28.5,
                -20.5, -28.5, 44.5, 36.5, -12.5, -40.5, 36.5, 48.5, -24.5, -20.5, 64.5, 28.5, -12.5, -28.5, 52.5, 36.5,
                -4.5,  -40.5, 44.5, 48.5, -16.5, -20.5, 72.5, 28.5, -4.5,  -28.5, 60.5, 36.5, 3.5,   -40.5, 52.5, 48.5,
                -8.5,  -20.5, 80.5, 28.5, 3.5,   -28.5, 68.5, 36.5, 11.5,  -40.5, 60.5, 48.5, -0.5,  -20.5, 88.5, 28.5,
                11.5,  -28.5, 76.5, 36.5, 19.5,  -40.5, 68.5, 48.5, 7.5,   -20.5, 96.5, 28.5, 19.5,  -28.5, 84.5, 36.5,
                27.5,  -40.5, 76.5, 48.5, -40.5, -12.5, 48.5, 36.5, -28.5, -20.5, 36.5, 44.5, -20.5, -32.5, 28.5, 56.5,
                -32.5, -12.5, 56.5, 36.5, -20.5, -20.5, 44.5, 44.5, -12.5, -32.5, 36.5, 56.5, -24.5, -12.5, 64.5, 36.5,
                -12.5, -20.5, 52.5, 44.5, -4.5,  -32.5, 44.5, 56.5, -16.5, -12.5, 72.5, 36.5, -4.5,  -20.5, 60.5, 44.5,
                3.5,   -32.5, 52.5, 56.5, -8.5,  -12.5, 80.5, 36.5, 3.5,   -20.5, 68.5, 44.5, 11.5,  -32.5, 60.5, 56.5,
                -0.5,  -12.5, 88.5, 36.5, 11.5,  -20.5, 76.5, 44.5, 19.5,  -32.5, 68.5, 56.5, 7.5,   -12.5, 96.5, 36.5,
                19.5,  -20.5, 84.5, 44.5, 27.5,  -32.5, 76.5, 56.5, -40.5, -4.5,  48.5, 44.5, -28.5, -12.5, 36.5, 52.5,
                -20.5, -24.5, 28.5, 64.5, -32.5, -4.5,  56.5, 44.5, -20.5, -12.5, 44.5, 52.5, -12.5, -24.5, 36.5, 64.5,
                -24.5, -4.5,  64.5, 44.5, -12.5, -12.5, 52.5, 52.5, -4.5,  -24.5, 44.5, 64.5, -16.5, -4.5,  72.5, 44.5,
                -4.5,  -12.5, 60.5, 52.5, 3.5,   -24.5, 52.5, 64.5, -8.5,  -4.5,  80.5, 44.5, 3.5,   -12.5, 68.5, 52.5,
                11.5,  -24.5, 60.5, 64.5, -0.5,  -4.5,  88.5, 44.5, 11.5,  -12.5, 76.5, 52.5, 19.5,  -24.5, 68.5, 64.5,
                7.5,   -4.5,  96.5, 44.5, 19.5,  -12.5, 84.5, 52.5, 27.5,  -24.5, 76.5, 64.5}),
        ExperimentalPGGParams(
            Attrs{true, 3, 6, 64.0f, 64.0f},
            {3, 4},
            {1, 16, 100, 100},
            {1, 3, 100, 200},
            {30000, 4},
            IN_ET,
            std::vector<T>{-364.5, -184.5, 364.5, 184.5, -256.5, -256.5, 256.5, 256.5, -180.5, -360.5, 180.5, 360.5},
            std::vector<T>{-332.5, -152.5, 396.5, 216.5, -224.5, -224.5, 288.5, 288.5, -148.5, -328.5, 212.5, 392.5,
                           -268.5, -152.5, 460.5, 216.5, -160.5, -224.5, 352.5, 288.5, -84.5,  -328.5, 276.5, 392.5,
                           -204.5, -152.5, 524.5, 216.5, -96.5,  -224.5, 416.5, 288.5, -20.5,  -328.5, 340.5, 392.5,
                           -140.5, -152.5, 588.5, 216.5, -32.5,  -224.5, 480.5, 288.5, 43.5,   -328.5, 404.5, 392.5,
                           -76.5,  -152.5, 652.5, 216.5, 31.5,   -224.5, 544.5, 288.5, 107.5,  -328.5, 468.5, 392.5,
                           -12.5,  -152.5, 716.5, 216.5, 95.5,   -224.5, 608.5, 288.5, 171.5,  -328.5, 532.5, 392.5,
                           -332.5, -88.5,  396.5, 280.5, -224.5, -160.5, 288.5, 352.5, -148.5, -264.5, 212.5, 456.5,
                           -268.5, -88.5,  460.5, 280.5, -160.5, -160.5, 352.5, 352.5, -84.5,  -264.5, 276.5, 456.5,
                           -204.5, -88.5,  524.5, 280.5, -96.5,  -160.5, 416.5, 352.5, -20.5,  -264.5, 340.5, 456.5,
                           -140.5, -88.5,  588.5, 280.5, -32.5,  -160.5, 480.5, 352.5, 43.5,   -264.5, 404.5, 456.5,
                           -76.5,  -88.5,  652.5, 280.5, 31.5,   -160.5, 544.5, 352.5, 107.5,  -264.5, 468.5, 456.5,
                           -12.5,  -88.5,  716.5, 280.5, 95.5,   -160.5, 608.5, 352.5, 171.5,  -264.5, 532.5, 456.5,
                           -332.5, -24.5,  396.5, 344.5, -224.5, -96.5,  288.5, 416.5, -148.5, -200.5, 212.5, 520.5,
                           -268.5, -24.5,  460.5, 344.5, -160.5, -96.5,  352.5, 416.5, -84.5,  -200.5, 276.5, 520.5,
                           -204.5, -24.5,  524.5, 344.5, -96.5,  -96.5,  416.5, 416.5, -20.5,  -200.5, 340.5, 520.5,
                           -140.5, -24.5,  588.5, 344.5, -32.5,  -96.5,  480.5, 416.5, 43.5,   -200.5, 404.5, 520.5,
                           -76.5,  -24.5,  652.5, 344.5, 31.5,   -96.5,  544.5, 416.5, 107.5,  -200.5, 468.5, 520.5,
                           -12.5,  -24.5,  716.5, 344.5, 95.5,   -96.5,  608.5, 416.5, 171.5,  -200.5, 532.5, 520.5}),
        ExperimentalPGGParams(
            Attrs{false, 5, 3, 32.0f, 32.0f},
            {3, 4},
            {1, 16, 100, 100},
            {1, 3, 100, 200},
            {100, 100, 3, 4},
            IN_ET,
            std::vector<T>{-180.5, -88.5, 180.5, 88.5, -128.5, -128.5, 128.5, 128.5, -92.5, -184.5, 92.5, 184.5},
            std::vector<T>{-164.5, -72.5, 196.5, 104.5, -112.5, -112.5, 144.5, 144.5, -76.5, -168.5, 108.5, 200.5,
                           -132.5, -72.5, 228.5, 104.5, -80.5,  -112.5, 176.5, 144.5, -44.5, -168.5, 140.5, 200.5,
                           -100.5, -72.5, 260.5, 104.5, -48.5,  -112.5, 208.5, 144.5, -12.5, -168.5, 172.5, 200.5,
                           -164.5, -40.5, 196.5, 136.5, -112.5, -80.5,  144.5, 176.5, -76.5, -136.5, 108.5, 232.5,
                           -132.5, -40.5, 228.5, 136.5, -80.5,  -80.5,  176.5, 176.5, -44.5, -136.5, 140.5, 232.5,
                           -100.5, -40.5, 260.5, 136.5, -48.5,  -80.5,  208.5, 176.5, -12.5, -136.5, 172.5, 232.5,
                           -164.5, -8.5,  196.5, 168.5, -112.5, -48.5,  144.5, 208.5, -76.5, -104.5, 108.5, 264.5,
                           -132.5, -8.5,  228.5, 168.5, -80.5,  -48.5,  176.5, 208.5, -44.5, -104.5, 140.5, 264.5,
                           -100.5, -8.5,  260.5, 168.5, -48.5,  -48.5,  208.5, 208.5, -12.5, -104.5, 172.5, 264.5,
                           -164.5, 23.5,  196.5, 200.5, -112.5, -16.5,  144.5, 240.5, -76.5, -72.5,  108.5, 296.5,
                           -132.5, 23.5,  228.5, 200.5, -80.5,  -16.5,  176.5, 240.5, -44.5, -72.5,  140.5, 296.5,
                           -100.5, 23.5,  260.5, 200.5, -48.5,  -16.5,  208.5, 240.5, -12.5, -72.5,  172.5, 296.5,
                           -164.5, 55.5,  196.5, 232.5, -112.5, 15.5,   144.5, 272.5, -76.5, -40.5,  108.5, 328.5,
                           -132.5, 55.5,  228.5, 232.5, -80.5,  15.5,   176.5, 272.5, -44.5, -40.5,  140.5, 328.5,
                           -100.5, 55.5,  260.5, 232.5, -48.5,  15.5,   208.5, 272.5, -12.5, -40.5,  172.5, 328.5}),
    };
    return experimentalPGGParams;
}

std::vector<ExperimentalPGGParams> generateExperimentalPGGCombinedParams() {
    const std::vector<std::vector<ExperimentalPGGParams>> experimentalPGGTypeParams{
        generateExperimentalPGGFloatParams<element::Type_t::f64>(),
        generateExperimentalPGGFloatParams<element::Type_t::f32>(),
        generateExperimentalPGGFloatParams<element::Type_t::f16>(),
        generateExperimentalPGGFloatParams<element::Type_t::bf16>(),
    };
    std::vector<ExperimentalPGGParams> combinedParams;

    for (const auto& params : experimentalPGGTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronPriorGridGenerator_With_Hardcoded_Refs,
                         ReferenceExperimentalPGGLayerTest,
                         testing::ValuesIn(generateExperimentalPGGCombinedParams()),
                         ReferenceExperimentalPGGLayerTest::getTestCaseName);
}  // namespace
