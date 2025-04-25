// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_generate_proposals.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

using Attrs = op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes;

namespace {
struct ExperimentalGPParams {
    template <class IT>
    ExperimentalGPParams(const Attrs& attrs,
                         const size_t number_of_channels,
                         const size_t height,
                         const size_t width,
                         const element::Type& iType,
                         const std::vector<IT>& imageSizeInfoValues,
                         const std::vector<IT>& anchorsValues,
                         const std::vector<IT>& deltasValues,
                         const std::vector<IT>& scoresValues,
                         const std::vector<IT>& refRoisValues,
                         const std::vector<IT>& refScoresValues,
                         const std::string& testcaseName = "")
        : attrs(attrs),
          inType(iType),
          outType(iType),
          imageSizeInfoData(CreateTensor(iType, imageSizeInfoValues)),
          anchorsData(CreateTensor(iType, anchorsValues)),
          deltasData(CreateTensor(iType, deltasValues)),
          scoresData(CreateTensor(iType, scoresValues)),
          testcaseName(testcaseName) {
        imageSizeInfoShape = Shape{3};
        anchorsShape = Shape{height * width * number_of_channels, 4};
        deltasShape = Shape{number_of_channels * 4, height, width};
        scoresShape = Shape{number_of_channels, height, width};

        const auto post_nms = static_cast<size_t>(attrs.post_nms_count);
        refRoisData = CreateTensor(Shape{post_nms, 4}, iType, refRoisValues);
        refScoresData = CreateTensor(Shape{post_nms}, iType, refScoresValues);
    }

    Attrs attrs;
    PartialShape imageSizeInfoShape;
    PartialShape anchorsShape;
    PartialShape deltasShape;
    PartialShape scoresShape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor imageSizeInfoData;
    ov::Tensor anchorsData;
    ov::Tensor deltasData;
    ov::Tensor scoresData;
    ov::Tensor refRoisData;
    ov::Tensor refScoresData;
    std::string testcaseName;
};

class ReferenceExperimentalGPLayerTest : public testing::TestWithParam<ExperimentalGPParams>,
                                         public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.imageSizeInfoData, params.anchorsData, params.deltasData, params.scoresData};
        refOutData = {params.refRoisData, params.refScoresData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ExperimentalGPParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "imageSizeInfoShape=" << param.imageSizeInfoShape << "_";
        result << "anchorsShape=" << param.anchorsShape << "_";
        result << "deltasShape=" << param.deltasShape << "_";
        result << "scoresShape=" << param.scoresShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        if (param.testcaseName != "")
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ExperimentalGPParams& params) {
        const auto im_info = std::make_shared<op::v0::Parameter>(params.inType, params.imageSizeInfoShape);
        const auto anchors = std::make_shared<op::v0::Parameter>(params.inType, params.anchorsShape);
        const auto deltas = std::make_shared<op::v0::Parameter>(params.inType, params.deltasShape);
        const auto scores = std::make_shared<op::v0::Parameter>(params.inType, params.scoresShape);
        const auto ExperimentalGP =
            std::make_shared<op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(im_info,
                                                                                        anchors,
                                                                                        deltas,
                                                                                        scores,
                                                                                        params.attrs);
        return std::make_shared<ov::Model>(ExperimentalGP->outputs(),
                                           ParameterVector{im_info, anchors, deltas, scores});
    }
};

TEST_P(ReferenceExperimentalGPLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ExperimentalGPParams> generateExperimentalGPFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ExperimentalGPParams> experimentalGPParams{
        ExperimentalGPParams(
            Attrs{
                0,                  // min_size
                0.699999988079071,  // nms_threshold
                6,                  // post_nms_count
                1000                // pre_nms_count
            },
            3,
            2,
            6,
            IN_ET,
            std::vector<T>{1.0f, 1.0f, 1.0f},
            std::vector<T>{
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
            std::vector<T>{
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
            std::vector<T>{5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                           1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f,
                           1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f},
            std::vector<T>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
            std::vector<T>{8.0f, 5.0f, 4.0f, 1.0f, 1.0f, 1.0f},
            "eval"),
        ExperimentalGPParams(
            Attrs{
                0,                  // min_size
                0.699999988079071,  // nms_threshold
                6,                  // post_nms_count
                1000                // pre_nms_count
            },
            3,
            2,
            6,
            IN_ET,
            std::vector<T>{150.0, 150.0, 1.0},
            std::vector<T>{
                12.0,  68.0,  102.0, 123.0, 46.0,  80.0,  79.0,  128.0, 33.0,  71.0,  127.0, 86.0,  33.0,  56.0,  150.0,
                73.0,  5.0,   41.0,  93.0,  150.0, 74.0,  66.0,  106.0, 115.0, 17.0,  37.0,  87.0,  150.0, 31.0,  27.0,
                150.0, 39.0,  29.0,  23.0,  112.0, 123.0, 41.0,  37.0,  103.0, 150.0, 8.0,   46.0,  98.0,  111.0, 7.0,
                69.0,  114.0, 150.0, 70.0,  21.0,  150.0, 125.0, 54.0,  19.0,  132.0, 68.0,  62.0,  8.0,   150.0, 101.0,
                57.0,  81.0,  150.0, 97.0,  79.0,  29.0,  109.0, 130.0, 12.0,  63.0,  100.0, 150.0, 17.0,  33.0,  113.0,
                150.0, 90.0,  78.0,  150.0, 111.0, 47.0,  68.0,  150.0, 71.0,  66.0,  103.0, 111.0, 150.0, 4.0,   17.0,
                112.0, 94.0,  12.0,  8.0,   119.0, 98.0,  54.0,  56.0,  120.0, 150.0, 56.0,  29.0,  150.0, 31.0,  42.0,
                3.0,   139.0, 92.0,  41.0,  65.0,  150.0, 130.0, 49.0,  13.0,  143.0, 30.0,  40.0,  60.0,  150.0, 150.0,
                23.0,  73.0,  24.0,  115.0, 56.0,  84.0,  107.0, 108.0, 63.0,  8.0,   142.0, 125.0, 78.0,  37.0,  93.0,
                144.0, 40.0,  34.0,  150.0, 46.0,  30.0,  21.0,  150.0, 120.0},
            std::vector<T>{
                9.062256,   10.883133,  9.8441105,   12.694285,  0.41781136,  8.749107,    14.990341,     6.587644,
                1.4206103,  13.299262,  12.432549,   2.736371,   0.22732796,  6.3361835,   12.268727,     2.1009045,
                4.771589,   2.5131326,  5.610736,    9.3604145,  4.27379,     8.317948,    0.60510135,    6.7446275,
                1.0207708,  1.1352817,  1.5785321,   1.718335,   1.8093798,   0.99247587,  1.3233583,     1.7432803,
                1.8534478,  1.2593061,  1.7394226,   1.7686696,  1.647999,    1.7611449,   1.3119122,     0.03007332,
                1.1106564,  0.55669737, 0.2546148,   1.9181818,  0.7134989,   2.0407224,   1.7211134,     1.8565536,
                14.562747,  2.8786168,  0.5927796,   0.2064463,  7.6794515,   8.672126,    10.139171,     8.002429,
                7.002932,   12.6314945, 10.550842,   0.15784842, 0.3194304,   10.752157,   3.709805,      11.628928,
                0.7136225,  14.619964,  15.177284,   2.2824087,  15.381494,   0.16618137,  7.507227,      11.173228,
                0.4923559,  1.8227729,  1.4749299,   1.7833921,  1.2363617,   -0.23659119, 1.5737582,     1.779316,
                1.9828427,  1.0482665,  1.4900246,   1.3563544,  1.5341306,   0.7634312,   4.6216766e-05, 1.6161222,
                1.7512476,  1.9363779,  0.9195784,   1.4906164,  -0.03244795, 0.681073,    0.6192401,     1.8033613,
                14.146055,  3.4043705,  15.292292,   3.5295358,  11.138999,   9.952057,    5.633434,      12.114562,
                9.427372,   12.384038,  9.583308,    8.427233,   15.293704,   3.288159,    11.64898,      9.350885,
                2.0037227,  13.523184,  4.4176426,   6.1057625,  14.400079,   8.248259,    11.815807,     15.713364,
                1.0023532,  1.3203261,  1.7100681,   0.7407832,  1.09448,     1.7188418,   1.4412547,     1.4862992,
                0.74790007, 0.31571656, 0.6398838,   2.0236106,  1.1869069,   1.7265586,   1.2624544,     0.09934269,
                1.3508598,  0.85212964, -0.38968498, 1.7059708,  1.6533034,   1.7400402,   1.8123854,     -0.43063712},
            std::vector<T>{0.7719922,  0.35906568,  0.29054508, 0.18124384, 0.5604661,  0.84750974,
                           0.98948747, 0.009793862, 0.7184191,  0.5560748,  0.6952493,  0.6732593,
                           0.3306898,  0.6790913,   0.41128764, 0.34593266, 0.94296855, 0.7348507,
                           0.24478768, 0.94024557,  0.05405676, 0.06466125, 0.36244348, 0.07942984,
                           0.10619422, 0.09412837,  0.9053611,  0.22870538, 0.9237487,  0.20986171,
                           0.5067282,  0.29709867,  0.53138554, 0.189101,   0.4786443,  0.88421875},
            std::vector<T>{
                149, 149, 149, 149, 149, 0,   149, 149, 149, 60.87443542480469, 149, 149, 149, 61.89498901367188, 149,
                149, 149, 149, 149, 149, 149, 149, 149, 149},
            std::vector<T>{0.9894874691963196,
                           0.9429685473442078,
                           0.9402455687522888,
                           0.9237486720085144,
                           0.9053611159324646,
                           0.8842187523841858},
            "eval_2"),
    };
    return experimentalGPParams;
}

std::vector<ExperimentalGPParams> generateExperimentalGPCombinedParams() {
    const std::vector<std::vector<ExperimentalGPParams>> experimentalGPTypeParams{
        generateExperimentalGPFloatParams<element::Type_t::f32>(),
        generateExperimentalGPFloatParams<element::Type_t::f16>(),
        generateExperimentalGPFloatParams<element::Type_t::bf16>(),
    };
    std::vector<ExperimentalGPParams> combinedParams;

    for (const auto& params : experimentalGPTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronGenerateProposalsSingleImage_With_Hardcoded_Refs,
                         ReferenceExperimentalGPLayerTest,
                         testing::ValuesIn(generateExperimentalGPCombinedParams()),
                         ReferenceExperimentalGPLayerTest::getTestCaseName);
}  // namespace
