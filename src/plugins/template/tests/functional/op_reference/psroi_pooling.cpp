// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/psroi_pooling.hpp"

#include <gtest/gtest.h>

#include <random>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct PSROIPoolingParams {
    template <class IT>
    PSROIPoolingParams(const size_t num_channels,
                       const size_t group_size,
                       const size_t spatial_bins_x,
                       const size_t spatial_bins_y,
                       const size_t num_boxes,
                       const float spatial_scale,
                       const std::string& mode,
                       const ov::element::Type& iType,
                       const std::vector<IT>& coordsValues,
                       const std::vector<IT>& oValues,
                       const std::string& test_name = "")
        : groupSize(group_size),
          spatialBinsX(spatial_bins_x),
          spatialBinsY(spatial_bins_y),
          spatialScale(spatial_scale),
          mode(mode),
          imageInputType(iType),
          coordsInputType(iType),
          outType(iType),
          coordsData(CreateTensor(iType, coordsValues)),
          refData(CreateTensor(iType, oValues)),
          testcaseName(test_name) {
        if (mode == "bilinear")
            outputDim = num_channels / (spatial_bins_x * spatial_bins_y);
        else
            outputDim = num_channels / (group_size * group_size);
        imageShape = Shape{2, num_channels, 20, 20};
        coordsShape = Shape{num_boxes, 5};
        std::vector<IT> imageValues(shape_size(imageShape.get_shape()));
        float val = 0;
        std::generate(imageValues.begin(), imageValues.end(), [val]() mutable -> float {
            return val += 0.1;
        });
        imageData = CreateTensor(iType, imageValues);
    }

    size_t groupSize;
    size_t spatialBinsX;
    size_t spatialBinsY;
    size_t outputDim;
    float spatialScale;
    std::string mode;
    ov::PartialShape imageShape;
    ov::PartialShape coordsShape;
    ov::element::Type imageInputType;
    ov::element::Type coordsInputType;
    ov::element::Type outType;
    ov::Tensor imageData;
    ov::Tensor coordsData;
    ov::Tensor refData;
    std::string testcaseName;
};

class ReferencePSROIPoolingLayerTest : public testing::TestWithParam<PSROIPoolingParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        legacy_compare = true;
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.imageData, params.coordsData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<PSROIPoolingParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "imageInputShape=" << param.imageShape << "_";
        result << "coordsInputShape=" << param.coordsShape << "_";
        result << "outputDim=" << param.outputDim << "_";
        result << "iType=" << param.imageInputType << "_";
        if (param.testcaseName != "") {
            result << "mode=" << param.mode << "_";
            result << param.testcaseName;
        } else {
            result << "mode=" << param.mode;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PSROIPoolingParams& params) {
        const auto image = std::make_shared<op::v0::Parameter>(params.imageInputType, params.imageShape);
        const auto coords = std::make_shared<op::v0::Parameter>(params.coordsInputType, params.coordsShape);
        const auto PSROIPooling = std::make_shared<op::v0::PSROIPooling>(image,
                                                                         coords,
                                                                         params.outputDim,
                                                                         params.groupSize,
                                                                         params.spatialScale,
                                                                         static_cast<int>(params.spatialBinsX),
                                                                         static_cast<int>(params.spatialBinsY),
                                                                         params.mode);
        return std::make_shared<ov::Model>(NodeVector{PSROIPooling}, ParameterVector{image, coords});
    }
};

TEST_P(ReferencePSROIPoolingLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<PSROIPoolingParams> generatePSROIPoolingFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<PSROIPoolingParams> pSROIPoolingParams{
        PSROIPoolingParams(
            8,
            2,
            1,
            1,
            3,
            1,
            "average",
            IN_ET,
            std::vector<T>{// batch_id, x1, y1, x2, y2
                           0,
                           1,
                           2,
                           4,
                           6,
                           1,
                           0,
                           3,
                           10,
                           4,
                           0,
                           10,
                           7,
                           11,
                           13},
            std::vector<T>{6.2499962, 46.44986,  90.249184, 130.44876, 166.25095, 206.45341, 250.25606, 290.45853,
                           326.36069, 366.86316, 408.36572, 448.86816, 486.37045, 526.86841, 568.35828, 608.84839,
                           18.100033, 58.199684, 104.09898, 144.1996,  178.10167, 218.20412, 264.1069,  304.20935}),
        PSROIPoolingParams(
            8,
            2,
            1,
            1,
            4,
            0.2,
            "average",
            IN_ET,
            std::vector<T>{// batch_id, x1, y1, x2, y2
                           0, 5, 10, 20, 30, 0, 0, 15, 50, 20, 1, 50, 35, 55, 65, 1, 0, 60, 5, 70},
            std::vector<T>{6.24999619, 46.399868,  90.2491837, 130.398758, 166.250946, 206.403397, 250.256058,
                           290.408508, 6.34999657, 46.8498573, 87.3492432, 127.848656, 166.350952, 206.853409,
                           247.355896, 287.858368, 338.11142,  378.163879, 424.116669, 464.169128, 498.121185,
                           538.165649, 584.104431, 624.144653, 345.111847, 385.164307, 427.116852, 467.169312,
                           505.121613, 545.16394,  587.103699, 627.143921}),
        PSROIPoolingParams(
            12,
            3,
            2,
            3,
            5,
            1,
            "bilinear",
            IN_ET,
            std::vector<T>{0,   0.1, 0.2, 0.7,  0.4, 1,    0.4,  0.1, 0.9, 0.3, 0,   0.5, 0.7,
                           0.7, 0.9, 1,   0.15, 0.3, 0.65, 0.35, 0,   0.0, 0.2, 0.7, 0.8},
            std::vector<T>{
                210.71394, 210.99896, 211.28398, 211.98065, 212.26567, 212.55066, 213.24738, 213.53239, 213.8174,
                250.71545, 251.00047, 251.28548, 251.98218, 252.2672,  252.5522,  253.2489,  253.53392, 253.81892,
                687.40869, 687.64606, 687.88354, 688.67511, 688.91254, 689.14996, 689.94147, 690.17896, 690.41644,
                727.40021, 727.6377,  727.87518, 728.66669, 728.90405, 729.14154, 729.93292, 730.17041, 730.4079,
                230.28471, 230.3797,  230.47472, 231.55144, 231.64642, 231.74141, 232.81813, 232.91313, 233.00813,
                270.28638, 270.38141, 270.47641, 271.5531,  271.64813, 271.74313, 272.81985, 272.91486, 273.00986,
                692.63281, 692.87018, 693.1076,  692.94928, 693.18683, 693.42426, 693.26593, 693.50342, 693.74078,
                732.62402, 732.86139, 733.09888, 732.94049, 733.17804, 733.41547, 733.25714, 733.49463, 733.73199,
                215.63843, 215.97093, 216.30345, 219.43855, 219.77106, 220.10358, 223.23871, 223.57123, 223.90375,
                255.63994, 255.97246, 256.30496, 259.44009, 259.77261, 260.10513, 263.2403,  263.57281, 263.9053}),
        PSROIPoolingParams(
            12,
            4,
            2,
            3,
            6,
            0.5,
            "bilinear",
            IN_ET,
            std::vector<T>{// batch_id, x1, y1, x2, y2
                           0, 0.1, 0.2, 0.7, 0.4,  0, 0.5, 0.7, 1.2, 1.3, 0, 1.0,  1.3, 1.2,  1.8,
                           1, 0.5, 1.1, 0.7, 1.44, 1, 0.2, 1.1, 0.5, 1.2, 1, 0.34, 1.3, 1.15, 1.35},
            std::vector<T>{
                205.40955, 205.50456, 205.59955, 205.69453, 205.83179, 205.9268,  206.0218,  206.11681, 206.25403,
                206.34901, 206.44403, 206.53905, 206.67627, 206.77126, 206.86627, 206.96129, 245.41107, 245.50606,
                245.60106, 245.69604, 245.8333,  245.9283,  246.02327, 246.1183,  246.25554, 246.35052, 246.44556,
                246.54054, 246.67778, 246.77277, 246.86775, 246.96278, 217.84717, 217.95801, 218.06885, 218.17969,
                219.11389, 219.22473, 219.33557, 219.44641, 220.3806,  220.49144, 220.60228, 220.71312, 221.64732,
                221.75816, 221.86897, 221.97981, 257.84872, 257.95956, 258.0704,  258.18124, 259.11545, 259.22629,
                259.33713, 259.44797, 260.38217, 260.49301, 260.60385, 260.71469, 261.6489,  261.75974, 261.87057,
                261.98141, 228.9705,  229.00215, 229.03383, 229.06549, 230.02608, 230.05774, 230.08943, 230.12109,
                231.08168, 231.11334, 231.14502, 231.1767,  232.13728, 232.16895, 232.20062, 232.23228, 268.97217,
                269.00385, 269.03549, 269.06717, 270.02777, 270.05945, 270.09109, 270.12277, 271.08337, 271.11502,
                271.1467,  271.17838, 272.13901, 272.17065, 272.2023,  272.23398, 703.65057, 703.68219, 703.71387,
                703.74554, 704.36816, 704.39984, 704.43146, 704.4632,  705.08575, 705.11749, 705.14911, 705.18085,
                705.80347, 705.83514, 705.86676, 705.89844, 743.64136, 743.67291, 743.70459, 743.73633, 744.35889,
                744.39056, 744.42218, 744.45392, 745.07648, 745.10815, 745.13983, 745.17157, 745.79413, 745.82574,
                745.85742, 745.8891,  701.86963, 701.91724, 701.9646,  702.01221, 702.08081, 702.12823, 702.17578,
                702.22321, 702.29181, 702.33936, 702.38678, 702.43433, 702.50293, 702.55035, 702.5979,  702.64545,
                741.86041, 741.90796, 741.95538, 742.00293, 742.07153, 742.11896, 742.1665,  742.21405, 742.28253,
                742.33008, 742.3775,  742.42505, 742.49365, 742.54108, 742.58862, 742.63617, 705.60645, 705.73468,
                705.86298, 705.99115, 705.71198, 705.84027, 705.96844, 706.09668, 705.81757, 705.94574, 706.07397,
                706.20215, 705.9231,  706.05127, 706.1795,  706.3078,  745.59698, 745.72534, 745.85352, 745.98169,
                745.70264, 745.83081, 745.95898, 746.08722, 745.80811, 745.93628, 746.06451, 746.19269, 745.91364,
                746.04181, 746.1701,  746.29834}),
    };
    return pSROIPoolingParams;
}

std::vector<PSROIPoolingParams> generatePSROIPoolingCombinedParams() {
    const std::vector<std::vector<PSROIPoolingParams>> pSROIPoolingTypeParams{
        generatePSROIPoolingFloatParams<element::Type_t::f64>(),
        generatePSROIPoolingFloatParams<element::Type_t::f32>(),
        generatePSROIPoolingFloatParams<element::Type_t::f16>(),
        generatePSROIPoolingFloatParams<element::Type_t::bf16>()};
    std::vector<PSROIPoolingParams> combinedParams;

    for (const auto& params : pSROIPoolingTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_PSROIPooling_With_Hardcoded_Refs,
                         ReferencePSROIPoolingLayerTest,
                         testing::ValuesIn(generatePSROIPoolingCombinedParams()),
                         ReferencePSROIPoolingLayerTest::getTestCaseName);

}  // namespace
