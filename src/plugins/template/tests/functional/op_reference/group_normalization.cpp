// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_normalization.hpp"

#include "base_reference_test.hpp"
#include "gtest/gtest.h"
#include "openvino/op/group_conv.hpp"

using namespace std;
using namespace ov;
using namespace reference_tests;

namespace {
struct GroupNormalizationParams {
    GroupNormalizationParams(const reference_tests::Tensor& data,
                             const reference_tests::Tensor& scale,
                             const reference_tests::Tensor& bias,
                             const reference_tests::Tensor& expected,
                             int64_t num,
                             double eps,
                             string name)
        : data_tensor{data},
          scale_tensor{scale},
          bias_tensor{bias},
          expected_tensor{expected},
          num_groups{num},
          epsilon{eps},
          test_case_name{std::move(name)} {}

    reference_tests::Tensor data_tensor;
    reference_tests::Tensor scale_tensor;
    reference_tests::Tensor bias_tensor;
    reference_tests::Tensor expected_tensor;
    int64_t num_groups;
    double epsilon;
    string test_case_name;
};

class ReferenceGroupNormalization : public testing::TestWithParam<GroupNormalizationParams>,
                                    public CommonReferenceTest {
public:
    void SetUp() override {
        legacy_compare = true;
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.data_tensor.data, params.scale_tensor.data, params.bias_tensor.data};
        refOutData = {params.expected_tensor.data};
    }

    static string getTestCaseName(const testing::TestParamInfo<GroupNormalizationParams>& obj) {
        std::ostringstream name;
        name << obj.param.test_case_name;
        name << "_";
        name << obj.param.data_tensor.type;
        return name.str();
    }

private:
    static shared_ptr<Model> CreateFunction(const GroupNormalizationParams& params) {
        const auto in_data = make_shared<op::v0::Parameter>(params.data_tensor.type, params.data_tensor.shape);
        const auto in_scale = make_shared<op::v0::Parameter>(params.scale_tensor.type, params.scale_tensor.shape);
        const auto in_bias = make_shared<op::v0::Parameter>(params.bias_tensor.type, params.bias_tensor.shape);
        const auto group_norm =
            make_shared<op::v12::GroupNormalization>(in_data, in_scale, in_bias, params.num_groups, params.epsilon);
        return make_shared<Model>(NodeVector{group_norm}, ParameterVector{in_data, in_scale, in_bias});
    }
};

vector<GroupNormalizationParams> generateBasicParams() {
    constexpr auto et = element::f32;
    using vt = typename element_type_traits<et>::value_type;

    const Shape data_shape{1, 4, 2, 2};
    reference_tests::Tensor scale{{4}, et, vector<vt>{1, 1, 1, 1}};
    reference_tests::Tensor bias{{4}, et, vector<vt>{0, 0, 0, 0}};

    // clang-format off
    reference_tests::Tensor data{data_shape, et,
                                 vector<vt>{0.001, 0.002, 0.003, 0.004,
                                            1.01, 1.02, 1.03, 1.04,
                                            2.1, 2.2, 2.3, 2.4,
                                            11, 12, 13, 14}};
    reference_tests::Tensor output_4_groups{data_shape, et,
                                            vector<vt>{-0.4472, -0.1491, 0.1491, 0.4472,
                                                       -1.2910, -0.4303, 0.4303, 1.2910,
                                                       -1.3411, -0.4470, 0.4470, 1.3411,
                                                       -1.3416, -0.4472, 0.4472, 1.3416}};
    reference_tests::Tensor output_2_groups{data_shape, et,
                                            vector<vt>{-1.0028, -1.0008, -0.9989, -0.9969,
                                                        0.9705,  0.9901,  1.0096,  1.0292,
                                                       -1.0171, -0.9978, -0.9786, -0.9593,
                                                        0.6990,  0.8918,  1.0846,  1.2774}};
    reference_tests::Tensor output_1_group{data_shape, et,
                                           vector<vt>{-0.7832, -0.7830, -0.7828, -0.7826,
                                                      -0.5828, -0.5808, -0.5789, -0.5769,
                                                      -0.3663, -0.3465, -0.3266, -0.3067,
                                                       1.4014,  1.6000,  1.7986,  1.9973}};
    // clang-format on

    vector<GroupNormalizationParams> params;
    params.emplace_back(data, scale, bias, output_4_groups, 4, 1e-5, "basic_4groups");
    params.emplace_back(data, scale, bias, output_2_groups, 2, 1e-5, "basic_2groups");
    params.emplace_back(data, scale, bias, output_1_group, 1, 1e-5, "basic_1group");
    return params;
}

template <element::Type_t et>
vector<GroupNormalizationParams> generateVariousScaleBiasParams() {
    using vt = typename element_type_traits<et>::value_type;

    const Shape data_shape{1, 2, 2, 2};
    reference_tests::Tensor scale{{2}, et, vector<vt>{0.7, 1.2}};
    reference_tests::Tensor bias{{2}, et, vector<vt>{0.2, -0.3}};

    reference_tests::Tensor data{data_shape, et, vector<vt>{0.001, 0.002, 0.003, 0.004, 2.1, 2.2, 2.3, 2.4}};
    reference_tests::Tensor output_2_groups{
        data_shape,
        et,
        vector<vt>{-0.5937, -0.0646, 0.4646, 0.9937, -1.9099, -0.8366, 0.2366, 1.3099}};
    reference_tests::Tensor output_1_group{
        data_shape,
        et,
        vector<vt>{-0.4992, -0.4986, -0.4980, -0.4973, 0.7373, 0.8438, 0.9503, 1.0568}};

    vector<GroupNormalizationParams> params;
    params.emplace_back(data, scale, bias, output_2_groups, 2, 5e-7, "scale_bias_2groups");
    params.emplace_back(data, scale, bias, output_1_group, 1, 5e-7, "scale_bias_1group");
    return params;
}

vector<GroupNormalizationParams> generateComplexParams() {
    constexpr auto et = element::f32;
    using vt = typename element_type_traits<et>::value_type;

    vector<GroupNormalizationParams> params;
    {
        const Shape data_shape{2, 2, 1, 2};
        reference_tests::Tensor scale{{2}, et, vector<vt>{2, 3}};
        reference_tests::Tensor bias{{2}, et, vector<vt>{-1, 10}};

        reference_tests::Tensor data{data_shape, et, vector<vt>{1, 11, 2, 22, 33, 3, 44, 4}};

        reference_tests::Tensor output_2_groups{data_shape, et, vector<vt>{-3, 1, 7, 13, 1, -3, 13, 7}};
        reference_tests::Tensor output_1_group{
            data_shape,
            et,
            vector<vt>{-2.8922, -0.5270, 7.5165, 14.6122, 0.3385, -3.0078, 13.8482, 7.1557}};
        params.emplace_back(data, scale, bias, output_2_groups, 2, 1e-5, "2batches_2groups");
        params.emplace_back(data, scale, bias, output_1_group, 1, 1e-5, "2batches_1group");
    }

    {
        const Shape data_shape{3, 3, 7, 5};
        reference_tests::Tensor scale{{3}, et, vector<vt>{0.77, 0.99, 1.11}};
        reference_tests::Tensor bias{{3}, et, vector<vt>{0.21, 0.07, -0.37}};
        reference_tests::Tensor data{
            data_shape,
            et,
            vector<vt>{0.1648, 0.2497, 0.1195, 0.1584, 0.2611, 0.3091, 0.1737, 0.0421, 0.9458, 0.3296, 0.8453, 0.1896,
                       0.1267, 0.2665, 0.2749, 0.2594, 0.4134, 0.7526, 0.6499, 0.7584, 0.8176, 0.9045, 0.8179, 0.9137,
                       0.5692, 0.6913, 0.1530, 0.8242, 0.1903, 0.2448, 0.9984, 0.9623, 0.5121, 0.2807, 0.5177,

                       0.3564, 0.6398, 0.5228, 0.3958, 0.1639, 0.5372, 0.5363, 0.5431, 0.5481, 0.7458, 0.5475, 0.6672,
                       0.8525, 0.7553, 0.1790, 0.5535, 0.4606, 0.9114, 0.9227, 0.1119, 0.5071, 0.2426, 0.2356, 0.7997,
                       0.8595, 0.5270, 0.4007, 0.7677, 0.1941, 0.1273, 0.8571, 0.5119, 0.8057, 0.8024, 0.2653,

                       0.7026, 0.0503, 0.4928, 0.1142, 0.2521, 0.5274, 0.9813, 0.1118, 0.6919, 0.9727, 0.4152, 0.7918,
                       0.2413, 0.3088, 0.3133, 0.0472, 0.1392, 0.1585, 0.7208, 0.3988, 0.9662, 0.9470, 0.4756, 0.7606,
                       0.9366, 0.8194, 0.6172, 0.8799, 0.9407, 0.4477, 0.4529, 0.8176, 0.1426, 0.4921, 0.1570,

                       0.9935, 0.0889, 0.3966, 0.9366, 0.0620, 0.2797, 0.6732, 0.0448, 0.4008, 0.5622, 0.2622, 0.1329,
                       0.9235, 0.3959, 0.2991, 0.7200, 0.8702, 0.9584, 0.4113, 0.5059, 0.4774, 0.6776, 0.6542, 0.1194,
                       0.1748, 0.6760, 0.0938, 0.7508, 0.6926, 0.6281, 0.8262, 0.2689, 0.0053, 0.3767, 0.9926,

                       0.0568, 0.2568, 0.8414, 0.0434, 0.5292, 0.0985, 0.2029, 0.5948, 0.3747, 0.5926, 0.2130, 0.0519,
                       0.9413, 0.2119, 0.2858, 0.2752, 0.9895, 0.1642, 0.9411, 0.8528, 0.5139, 0.3473, 0.8534, 0.3050,
                       0.1338, 0.7669, 0.2127, 0.7881, 0.5265, 0.3529, 0.5342, 0.5154, 0.4277, 0.1556, 0.2562,

                       0.9280, 0.6243, 0.4608, 0.3487, 0.8103, 0.8956, 0.8416, 0.3014, 0.5215, 0.9945, 0.3381, 0.7508,
                       0.6548, 0.8600, 0.5781, 0.7989, 0.0082, 0.9144, 0.4307, 0.1504, 0.3410, 0.7372, 0.6724, 0.4242,
                       0.0205, 0.9671, 0.5956, 0.1512, 0.5713, 0.6840, 0.2899, 0.3118, 0.6995, 0.8726, 0.0138,

                       0.2557, 0.5081, 0.6399, 0.0994, 0.8553, 0.8695, 0.0975, 0.1074, 0.2329, 0.4659, 0.4810, 0.3403,
                       0.6434, 0.1760, 0.0869, 0.7490, 0.4383, 0.2731, 0.3274, 0.0870, 0.2499, 0.6138, 0.2206, 0.6845,
                       0.6387, 0.7274, 0.5181, 0.8421, 0.4487, 0.2840, 0.9888, 0.9848, 0.4048, 0.1550, 0.5963,

                       0.5346, 0.8775, 0.6872, 0.6386, 0.4324, 0.7045, 0.0228, 0.7691, 0.8856, 0.7495, 0.6414, 0.6308,
                       0.2641, 0.5891, 0.1852, 0.0247, 0.0910, 0.8219, 0.0344, 0.6146, 0.1786, 0.1071, 0.8895, 0.4652,
                       0.3224, 0.2132, 0.0159, 0.6958, 0.2965, 0.6573, 0.6049, 0.4310, 0.6706, 0.6910, 0.8486,

                       0.2045, 0.8496, 0.1481, 0.7015, 0.1985, 0.3038, 0.9332, 0.4544, 0.2214, 0.5375, 0.1587, 0.0834,
                       0.8334, 0.1958, 0.2984, 0.8155, 0.6043, 0.8328, 0.5758, 0.2238, 0.9458, 0.9503, 0.1430, 0.5495,
                       0.1429, 0.4320, 0.1902, 0.6482, 0.1235, 0.8223, 0.8314, 0.6919, 0.8652, 0.1149, 0.3165}};
        reference_tests::Tensor output{
            data_shape,
            et,
            vector<vt>{-0.7244, -0.4964, -0.8461, -0.7416, -0.4657, -0.3368, -0.7005, -1.0541, 1.3736,
                       -0.2817, 1.1036,  -0.6578, -0.8268, -0.4512, -0.4287, -0.4703, -0.0566, 0.8546,
                       0.5787,  0.8702,  1.0292,  1.2627,  1.0300,  1.2874,  0.3619,  0.6899,  -0.7561,
                       1.0469,  -0.6559, -0.5095, 1.5149,  1.4179,  0.2085,  -0.4131, 0.2236,

                       -0.4697, 0.5092,  0.1051,  -0.3336, -1.1345, 0.1548,  0.1517,  0.1752,  0.1925,
                       0.8753,  0.1904,  0.6038,  1.2438,  0.9081,  -1.0824, 0.2111,  -0.1098, 1.4473,
                       1.4863,  -1.3141, 0.0508,  -0.8627, -0.8869, 1.0615,  1.2680,  0.1196,  -0.3166,
                       0.9509,  -1.0302, -1.2609, 1.2597,  0.0674,  1.0822,  1.0708,  -0.7843,

                       0.3656,  -2.1605, -0.4469, -1.9130, -1.3790, -0.3129, 1.4449,  -1.9223, 0.3242,
                       1.4116,  -0.7474, 0.7110,  -1.4208, -1.1594, -1.1420, -2.1725, -1.8162, -1.7414,
                       0.4361,  -0.8109, 1.3864,  1.3121,  -0.5135, 0.5902,  1.2718,  0.8179,  0.0349,
                       1.0522,  1.2877,  -0.6215, -0.6014, 0.8109,  -1.8030, -0.4496, -1.7473,

                       1.5069,  -0.8523, -0.0498, 1.3585,  -0.9224, -0.3547, 0.6716,  -0.9673, -0.0388,
                       0.3821,  -0.4003, -0.7375, 1.3244,  -0.0516, -0.3041, 0.7936,  1.1854,  1.4154,
                       -0.0115, 0.2353,  0.1609,  0.6830,  0.6220,  -0.7727, -0.6283, 0.6789,  -0.8395,
                       0.8740,  0.7222,  0.5540,  1.0706,  -0.3828, -1.0703, -0.1017, 1.5046,

                       -1.4034, -0.7328, 1.2275,  -1.4484, 0.1806,  -1.2636, -0.9135, 0.4006,  -0.3375,
                       0.3932,  -0.8797, -1.4199, 1.5624,  -0.8834, -0.6356, -0.6711, 1.7241,  -1.0433,
                       1.5618,  1.2657,  0.1293,  -0.4293, 1.2677,  -0.5712, -1.1452, 0.9776,  -0.8807,
                       1.0487,  0.1715,  -0.4106, 0.1974,  0.1343,  -0.1597, -1.0721, -0.7348,

                       1.2533,  0.1115,  -0.5032, -0.9246, 0.8108,  1.1315,  0.9285,  -1.1024, -0.2749,
                       1.5033,  -0.9645, 0.5871,  0.2262,  0.9977,  -0.0622, 0.7680,  -2.2047, 1.2022,
                       -0.6163, -1.6701, -0.9535, 0.5360,  0.2924,  -0.6408, -2.1585, 1.4003,  0.0036,
                       -1.6671, -0.0877, 0.3360,  -1.1457, -1.0633, 0.3943,  1.0450,  -2.1837,

                       -0.3997, 0.2887,  0.6482,  -0.8260, 1.2357,  1.2744,  -0.8311, -0.8041, -0.4619,
                       0.1736,  0.2148,  -0.1689, 0.6577,  -0.6170, -0.8600, 0.9457,  0.0983,  -0.3522,
                       -0.2041, -0.8598, -0.4155, 0.5770,  -0.4954, 0.7698,  0.6449,  0.8868,  0.3160,
                       1.1997,  0.1267,  -0.3225, 1.5998,  1.5888,  0.0070,  -0.6743, 0.5293,

                       0.2641,  1.4665,  0.7992,  0.6288,  -0.0942, 0.8599,  -1.5305, 1.0864,  1.4949,
                       1.0177,  0.6386,  0.6015,  -0.6844, 0.4552,  -0.9611, -1.5239, -1.2914, 1.2716,
                       -1.4899, 0.5447,  -0.9842, -1.2349, 1.5086,  0.0208,  -0.4800, -0.8629, -1.5547,
                       0.8294,  -0.5708, 0.6944,  0.5106,  -0.0992, 0.7410,  0.8126,  1.3652,

                       -1.4502, 1.0861,  -1.6719, 0.5038,  -1.4738, -1.0598, 1.4148,  -0.4677, -1.3837,
                       -0.1409, -1.6302, -1.9263, 1.0224,  -1.4844, -1.0810, 0.9521,  0.1217,  1.0201,
                       0.0096,  -1.3743, 1.4643,  1.4820,  -1.6920, -0.0938, -1.6924, -0.5557, -1.5064,
                       0.2943,  -1.7686, 0.9788,  1.0146,  0.4661,  1.1475,  -1.8025, -1.0098}};

        params.emplace_back(data, scale, bias, output, 1, 1e-5, "3batches_1group");
    }
    return params;
}

vector<GroupNormalizationParams> generateGroupNormalizationParams() {
    vector<vector<GroupNormalizationParams>> combo_params{generateBasicParams(),
                                                          generateVariousScaleBiasParams<element::f16>(),
                                                          generateVariousScaleBiasParams<element::f32>(),
                                                          generateVariousScaleBiasParams<element::f64>(),
                                                          generateComplexParams()};
    vector<GroupNormalizationParams> test_params;
    for (auto& params : combo_params)
        move(params.begin(), params.end(), back_inserter(test_params));
    return test_params;
}
}  // namespace

TEST_P(ReferenceGroupNormalization, LayerTest) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         ReferenceGroupNormalization,
                         ::testing::ValuesIn(generateGroupNormalizationParams()),
                         ReferenceGroupNormalization::getTestCaseName);
