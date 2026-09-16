// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_delta_net.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct GatedDeltaNetParams {
    PartialShape qShape;
    PartialShape kShape;
    PartialShape vShape;
    PartialShape stateShape;
    PartialShape gateShape;
    PartialShape betaShape;
    bool fuseQkL2Norm;
    float qL2NormEps;
    float kL2NormEps;
    std::string testcaseName;
    reference_tests::Tensor qData;
    reference_tests::Tensor kData;
    reference_tests::Tensor vData;
    reference_tests::Tensor stateData;
    reference_tests::Tensor gateData;
    reference_tests::Tensor betaData;
    reference_tests::Tensor expectedOutput;
    reference_tests::Tensor expectedState;
};

template <typename T>
GatedDeltaNetParams PrepareTestCaseParams(const PartialShape& qShape,
                                          const PartialShape& kShape,
                                          const PartialShape& vShape,
                                          const PartialShape& stateShape,
                                          const PartialShape& gateShape,
                                          const PartialShape& betaShape,
                                          bool fuseQkL2Norm,
                                          float qL2NormEps,
                                          float kL2NormEps,
                                          const std::vector<T>& qData,
                                          const std::vector<T>& kData,
                                          const std::vector<T>& vData,
                                          const std::vector<T>& stateData,
                                          const std::vector<T>& gateData,
                                          const std::vector<T>& betaData,
                                          const std::vector<T>& expectedOutput,
                                          const std::vector<T>& expectedState,
                                          const std::string& description) {
    GatedDeltaNetParams ret;
    const auto elementType = element::from<T>();
    ret.qShape = qShape;
    ret.kShape = kShape;
    ret.vShape = vShape;
    ret.stateShape = stateShape;
    ret.gateShape = gateShape;
    ret.betaShape = betaShape;
    ret.fuseQkL2Norm = fuseQkL2Norm;
    ret.qL2NormEps = qL2NormEps;
    ret.kL2NormEps = kL2NormEps;
    ret.testcaseName = description;
    ret.qData = reference_tests::Tensor(elementType, qShape.to_shape(), qData);
    ret.kData = reference_tests::Tensor(elementType, kShape.to_shape(), kData);
    ret.vData = reference_tests::Tensor(elementType, vShape.to_shape(), vData);
    ret.stateData = reference_tests::Tensor(elementType, stateShape.to_shape(), stateData);
    ret.gateData = reference_tests::Tensor(elementType, gateShape.to_shape(), gateData);
    ret.betaData = reference_tests::Tensor(elementType, betaShape.to_shape(), betaData);
    ret.expectedOutput = reference_tests::Tensor(elementType, vShape.to_shape(), expectedOutput);
    ret.expectedState = reference_tests::Tensor(elementType, stateShape.to_shape(), expectedState);
    return ret;
}

class ReferenceGatedDeltaNetTest : public testing::TestWithParam<GatedDeltaNetParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.qData.data,
                     params.kData.data,
                     params.vData.data,
                     params.stateData.data,
                     params.gateData.data,
                     params.betaData.data};
        refOutData = {params.expectedOutput.data, params.expectedState.data};
        threshold = 1e-5f;
        abs_threshold = 1e-5f;
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GatedDeltaNetParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "type=" << param.qData.data.get_element_type();
        result << "_qShape=" << param.qShape;
        result << "_vShape=" << param.vShape;
        result << "_fuse=" << param.fuseQkL2Norm;
        result << "_=" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const GatedDeltaNetParams& params) {
        const auto elementType = params.qData.data.get_element_type();
        const auto query = std::make_shared<op::v0::Parameter>(elementType, params.qShape);
        const auto key = std::make_shared<op::v0::Parameter>(elementType, params.kShape);
        const auto value = std::make_shared<op::v0::Parameter>(elementType, params.vShape);
        const auto state = std::make_shared<op::v0::Parameter>(elementType, params.stateShape);
        const auto gate = std::make_shared<op::v0::Parameter>(elementType, params.gateShape);
        const auto beta = std::make_shared<op::v0::Parameter>(elementType, params.betaShape);
        const auto gdn = std::make_shared<op::internal::GatedDeltaNet>(query,
                                                                       key,
                                                                       value,
                                                                       state,
                                                                       gate,
                                                                       beta,
                                                                       params.fuseQkL2Norm,
                                                                       params.qL2NormEps,
                                                                       params.kL2NormEps);
        return std::make_shared<Model>(OutputVector{gdn->output(0), gdn->output(1)},
                                       ParameterVector{query, key, value, state, gate, beta});
    }
};

TEST_P(ReferenceGatedDeltaNetTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<GatedDeltaNetParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<GatedDeltaNetParams> params{
        PrepareTestCaseParams<T>(PartialShape{1, 1, 1, 2},
                                 PartialShape{1, 1, 1, 2},
                                 PartialShape{1, 1, 1, 1},
                                 PartialShape{1, 1, 2, 1},
                                 PartialShape{1, 1, 1},
                                 PartialShape{1, 1, 1},
                                 false,
                                 1e-06f,
                                 1e-06f,
                                 std::vector<T>{1.0000000f, 2.0000000f},
                                 std::vector<T>{3.0000000f, 4.0000000f},
                                 std::vector<T>{3.0000000f},
                                 std::vector<T>{0.0000000f, 0.0000000f},
                                 std::vector<T>{0.0000000f},
                                 std::vector<T>{1.0000000f},
                                 std::vector<T>{23.3345242f},
                                 std::vector<T>{9.0000000f, 12.0000000f},
                                 "single_step"),
        PrepareTestCaseParams<T>(PartialShape{1, 1, 1, 2},
                                 PartialShape{1, 1, 1, 2},
                                 PartialShape{1, 1, 1, 1},
                                 PartialShape{1, 1, 2, 1},
                                 PartialShape{1, 1, 1},
                                 PartialShape{1, 1, 1},
                                 true,
                                 0.0f,
                                 0.0f,
                                 std::vector<T>{3.0000000f, 4.0000000f},
                                 std::vector<T>{3.0000000f, 4.0000000f},
                                 std::vector<T>{3.0000000f},
                                 std::vector<T>{0.0000000f, 0.0000000f},
                                 std::vector<T>{0.0000000f},
                                 std::vector<T>{1.0000000f},
                                 std::vector<T>{2.1213202f},
                                 std::vector<T>{1.8000000f, 2.4000001f},
                                 "fused_qk_l2norm"),
        PrepareTestCaseParams<T>(
            PartialShape{2, 3, 2, 4},
            PartialShape{2, 3, 2, 4},
            PartialShape{2, 3, 2, 4},
            PartialShape{2, 2, 4, 4},
            PartialShape{2, 3, 2},
            PartialShape{2, 3, 2},
            false,
            1e-06f,
            1e-06f,
            std::vector<T>{0.0980000f,  0.4300000f,  0.2060000f,  0.0900000f,  -0.1530000f, 0.2920000f,  -0.1250000f,
                           0.7840000f,  0.9270000f,  -0.2330000f, 0.5830000f,  0.0580000f,  0.1360000f,  0.8510000f,
                           -0.8580000f, -0.8260000f, -0.9600000f, 0.6650000f,  0.5560000f,  0.7400000f,  0.9570000f,
                           0.5980000f,  -0.0770000f, 0.5610000f,  -0.7630000f, 0.2800000f,  -0.7130000f, 0.8890000f,
                           0.0440000f,  -0.1710000f, -0.4710000f, 0.5480000f,  -0.0880000f, 0.1370000f,  -0.9620000f,
                           0.2350000f,  0.2240000f,  0.2340000f,  0.8870000f,  0.3640000f,  -0.2810000f, -0.1260000f,
                           0.3950000f,  -0.8800000f, 0.3340000f,  0.3410000f,  -0.5790000f, -0.7420000f},
            std::vector<T>{-0.3690000f, -0.2730000f, 0.1400000f,  -0.1230000f, 0.9770000f,  -0.7960000f, -0.5820000f,
                           -0.6770000f, 0.3060000f,  -0.4930000f, -0.0670000f, -0.5110000f, -0.6820000f, -0.7790000f,
                           0.3130000f,  -0.7240000f, -0.6070000f, -0.2630000f, 0.6420000f,  -0.8060000f, 0.6760000f,
                           -0.8080000f, 0.9530000f,  -0.0630000f, 0.9540000f,  0.2100000f,  0.4790000f,  -0.9220000f,
                           -0.4340000f, -0.7600000f, -0.4080000f, -0.7630000f, -0.3640000f, -0.1710000f, -0.8720000f,
                           0.3850000f,  0.1330000f,  -0.4690000f, 0.0460000f,  -0.8120000f, 0.1520000f,  0.8590000f,
                           -0.3630000f, 0.3350000f,  -0.7360000f, 0.4330000f,  -0.4210000f, -0.6340000f},
            std::vector<T>{0.1730000f,  -0.9600000f, 0.6580000f,  -0.9910000f, 0.3560000f,  -0.4600000f, 0.4700000f,
                           0.9240000f,  -0.5020000f, 0.1520000f,  0.1840000f,  0.1450000f,  -0.5540000f, 0.9050000f,
                           -0.1060000f, 0.6930000f,  0.3990000f,  -0.4050000f, 0.6280000f,  -0.2070000f, 0.7620000f,
                           0.1630000f,  0.7630000f,  0.3850000f,  0.4510000f,  0.0030000f,  0.9120000f,  0.2880000f,
                           -0.1520000f, 0.2130000f,  -0.9620000f, -0.3970000f, 0.3200000f,  -0.4200000f, 0.2360000f,
                           -0.1420000f, -0.7290000f, -0.4030000f, 0.1400000f,  0.1820000f,  0.1490000f,  0.3060000f,
                           0.3040000f,  -0.1370000f, 0.7930000f,  -0.2650000f, -0.1280000f, 0.7840000f},
            std::vector<T>{
                0.6120000f,  0.4080000f,  -0.8000000f, 0.8390000f,  0.4280000f,  0.9980000f,  -0.7010000f, 0.7360000f,
                -0.6750000f, 0.2310000f,  -0.7520000f, 0.6960000f,  0.6150000f,  0.1380000f,  -0.1860000f, -0.8620000f,
                0.3950000f,  -0.0930000f, 0.4440000f,  0.7330000f,  0.9510000f,  0.7120000f,  -0.9770000f, -0.2800000f,
                0.4600000f,  -0.6570000f, 0.0420000f,  -0.8910000f, -0.6000000f, -0.9630000f, 0.5870000f,  -0.5520000f,
                -0.3090000f, 0.8560000f,  0.4090000f,  -0.9360000f, -0.6710000f, 0.2430000f,  0.1540000f,  -0.5240000f,
                0.8680000f,  0.2280000f,  0.0710000f,  0.1800000f,  0.4600000f,  -0.3760000f, -0.2040000f, -0.5800000f,
                -0.6280000f, 0.8890000f,  0.4790000f,  -0.0190000f, -0.5450000f, -0.4910000f, -0.8840000f, -0.1310000f,
                -0.3760000f, 0.3930000f,  -0.2440000f, -0.6410000f, -0.9510000f, -0.8660000f, 0.3590000f,  -0.0930000f},
            std::vector<T>{-0.4630000f,
                           -0.1030000f,
                           -0.0100000f,
                           -0.7830000f,
                           -0.3370000f,
                           -0.7370000f,
                           -0.9790000f,
                           -0.2420000f,
                           -0.6800000f,
                           -0.6170000f,
                           -0.4120000f,
                           -0.1690000f},
            std::vector<T>{0.6290000f,
                           0.8730000f,
                           0.2740000f,
                           0.7980000f,
                           0.1860000f,
                           0.9530000f,
                           0.6870000f,
                           0.2160000f,
                           0.9470000f,
                           0.7310000f,
                           0.2540000f,
                           0.2130000f},
            std::vector<T>{0.0292801f,  0.1965543f,  -0.1907852f, 0.1802422f,  -0.3475720f, 0.0899363f,  0.1347459f,
                           0.0344922f,  -0.0127816f, 0.1556332f,  -0.3452297f, 0.3505386f,  0.3307252f,  0.1523854f,
                           -0.2742097f, 0.0062912f,  0.0174631f,  0.0383229f,  -0.0372409f, -0.1582982f, 0.2121457f,
                           -0.3082414f, 0.0589287f,  -0.2175981f, -0.4188678f, 0.1074970f,  -0.5152162f, -0.2856665f,
                           -0.0921463f, -0.2085163f, 0.2047262f,  0.1161998f,  0.1281915f,  -0.1862048f, 0.1532833f,
                           -0.0736611f, -0.0076235f, 0.1365270f,  0.0008625f,  -0.1217498f, -0.0673063f, 0.0286443f,
                           -0.0572516f, 0.0899307f,  -0.0291833f, -0.0237945f, -0.0286928f, 0.1317744f},
            std::vector<T>{
                0.0640817f,  0.3329480f,  -0.4654701f, 0.5303040f,  0.1172151f,  0.4877794f,  -0.3810537f, 0.4352176f,
                -0.1524748f, 0.0443132f,  -0.2629542f, 0.2324124f,  0.1395575f,  0.0638709f,  -0.1644997f, -0.3056021f,
                0.7906327f,  -0.2136607f, 0.4572420f,  0.0166299f,  -0.3592100f, -0.2845609f, -0.5910100f, -0.4625308f,
                0.6720783f,  0.3752943f,  0.5714668f,  0.3227212f,  -0.1172666f, -0.3795801f, 0.1385095f,  -0.2667879f,
                -0.0415166f, 0.1235775f,  0.0786008f,  -0.0288009f, -0.0850454f, 0.1343191f,  0.0412224f,  -0.0513090f,
                -0.1859950f, 0.2082087f,  -0.2475475f, 0.0598664f,  0.0949164f,  -0.0303361f, -0.0119988f, -0.1609728f,
                -0.3469469f, 0.2534083f,  0.2013349f,  -0.0675972f, 0.2506652f,  0.0815753f,  -0.2609223f, 0.0111625f,
                -0.1646039f, 0.1218667f,  -0.0549655f, -0.2564011f, 0.1661306f,  0.1205980f,  0.0909460f,  -0.1804092f},
            "multihead_multibatch_seq"),
        PrepareTestCaseParams<T>(
            PartialShape{1, 3, 2, 4},
            PartialShape{1, 3, 2, 4},
            PartialShape{1, 3, 4, 3},
            PartialShape{1, 4, 4, 3},
            PartialShape{1, 3, 4},
            PartialShape{1, 3, 4},
            true,
            1e-06f,
            1e-06f,
            std::vector<T>{-0.1660000f, 0.4410000f,  -1.0000000f, -0.3950000f, -0.7060000f, -0.8150000f,
                           -0.6270000f, -0.3090000f, -0.2060000f, 0.0780000f,  -0.1620000f, 0.3700000f,
                           -0.5910000f, 0.7560000f,  -0.9450000f, 0.3410000f,  -0.1650000f, 0.1170000f,
                           -0.7190000f, -0.6040000f, 0.6010000f,  0.9370000f,  -0.3730000f, 0.3850000f},
            std::vector<T>{0.7530000f,  0.7890000f,  -0.8300000f, -0.9220000f, -0.6600000f, 0.7560000f,
                           -0.8030000f, -0.1580000f, 0.9160000f,  0.0660000f,  0.3840000f,  -0.3690000f,
                           0.3730000f,  0.6690000f,  -0.9630000f, 0.5000000f,  0.9780000f,  0.4960000f,
                           -0.4390000f, 0.5790000f,  -0.7940000f, -0.1040000f, 0.8170000f,  -0.4130000f},
            std::vector<T>{-0.4240000f, -0.7400000f, -0.9610000f, 0.3580000f,  -0.5770000f, -0.4690000f,
                           -0.0170000f, -0.8930000f, 0.1480000f,  -0.7070000f, 0.1790000f,  0.4000000f,
                           -0.7950000f, -0.1720000f, 0.3890000f,  -0.1720000f, -0.9000000f, 0.0720000f,
                           0.3280000f,  0.0300000f,  0.8890000f,  0.1730000f,  0.8070000f,  -0.7250000f,
                           -0.7210000f, 0.6150000f,  -0.2050000f, -0.6690000f, 0.8550000f,  -0.3040000f,
                           0.5020000f,  0.4520000f,  0.7670000f,  0.2470000f,  0.5020000f,  -0.3020000f},
            std::vector<T>{-0.4600000f, 0.7920000f,  -0.1440000f, 0.9300000f,  0.3270000f,  0.2430000f,  -0.7710000f,
                           0.8990000f,  -0.1000000f, 0.1570000f,  -0.1840000f, -0.5260000f, 0.8070000f,  0.1470000f,
                           -0.9940000f, 0.2340000f,  -0.3470000f, 0.0540000f,  0.7720000f,  -0.2850000f, 0.8170000f,
                           0.2470000f,  -0.9680000f, 0.8590000f,  0.3820000f,  0.9950000f,  -0.6550000f, -0.7260000f,
                           0.8650000f,  0.3940000f,  -0.8680000f, 0.5110000f,  0.5080000f,  0.8460000f,  0.4230000f,
                           -0.7510000f, -0.9600000f, -0.9480000f, -0.9430000f, -0.5080000f, 0.7200000f,  0.0780000f,
                           0.1060000f,  0.6840000f,  -0.7520000f, -0.4420000f, 0.1720000f,  0.9390000f},
            std::vector<T>{-0.4390000f,
                           -0.9810000f,
                           -0.1990000f,
                           -0.7670000f,
                           -0.1930000f,
                           -0.6120000f,
                           -0.1360000f,
                           -0.2530000f,
                           -0.4440000f,
                           -0.8640000f,
                           -0.9400000f,
                           -0.8790000f},
            std::vector<T>{0.0450000f,
                           0.1070000f,
                           0.2260000f,
                           0.7130000f,
                           0.5600000f,
                           0.0130000f,
                           0.0720000f,
                           0.9670000f,
                           0.5680000f,
                           0.2030000f,
                           0.2520000f,
                           0.7440000f},
            std::vector<T>{0.3156259f,  -0.2373515f, 0.1005170f,  -0.1291053f, 0.0471067f,  -0.1535292f,
                           0.1965116f,  -0.6087571f, 0.0139169f,  0.1418082f,  -0.0751124f, 0.1409220f,
                           0.3011487f,  -0.0347568f, -0.1512573f, -0.0425837f, -0.0676554f, 0.0856228f,
                           0.0934182f,  -0.1311470f, 0.0110565f,  0.0364411f,  0.3941338f,  -0.1337204f,
                           0.1462962f,  -0.1119237f, 0.0600990f,  -0.0247636f, 0.0160566f,  -0.0388770f,
                           -0.0470632f, 0.0939437f,  -0.0877036f, -0.0738933f, -0.0188734f, -0.0037616f},
            std::vector<T>{-0.6210173f, 0.2913882f,  -0.0153076f, 0.1345025f,  0.2170283f,  0.0404436f,  -0.1634557f,
                           0.1040684f,  0.0639312f,  -0.0750427f, 0.1945509f,  -0.2552438f, -0.0368244f, 0.1344352f,
                           -0.1218083f, -0.0306820f, 0.0292087f,  -0.0143055f, 0.1094918f,  -0.0772062f, 0.0872796f,
                           -0.0474569f, 0.0097682f,  0.0502628f,  -0.0305745f, 0.2023586f,  -0.2529472f, -0.2165265f,
                           0.1987633f,  0.1122109f,  -0.1108112f, 0.2758878f,  0.2087887f,  0.1652790f,  0.0706161f,
                           -0.2388630f, -0.1336043f, -0.3598680f, 0.0403608f,  -0.0630931f, 0.2500483f,  -0.1466690f,
                           0.0922092f,  0.1713165f,  -0.1283414f, -0.0228204f, -0.0019501f, 0.1454688f},
            "gqa_fused_l2norm"),
    };
    return params;
}

std::vector<GatedDeltaNetParams> generateCombinedParams() {
    // GatedDeltaNet evaluate currently dispatches f32 only.
    const std::vector<std::vector<GatedDeltaNetParams>> generatedParams{
        generateParams<element::Type_t::f32>(),
    };
    std::vector<GatedDeltaNetParams> combinedParams;
    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_GatedDeltaNet_With_Hardcoded_Refs,
                         ReferenceGatedDeltaNetTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceGatedDeltaNetTest::getTestCaseName);

}  // namespace
