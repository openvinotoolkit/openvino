// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <iostream>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/rdft.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct RDFTParams {
    template <class T>
    RDFTParams(const Shape& input_shape,
               const Shape& expected_shape,
               const element::Type_t& input_type,
               const element::Type_t& expected_type,
               const std::vector<T>& input_value,
               const std::vector<T>& expected_value,
               const std::shared_ptr<op::v0::Constant>& axes,
               const std::shared_ptr<op::v0::Constant>& signal) {
        std::cout << "number of elements in input data is " << input_value.size() << "\n";
        m_input_shape = input_shape;
        m_expected_shape = expected_shape;
        m_input_type = input_type;
        m_expected_type = expected_type;
        m_input_value = CreateTensor(input_type, input_value);
        m_expected_value = CreateTensor(expected_type, expected_value);
        m_axes = axes;
        m_signal = signal;
    }

    Shape m_input_shape;
    Shape m_expected_shape;
    element::Type_t m_input_type;
    element::Type_t m_expected_type;
    ov::Tensor m_input_value;
    ov::Tensor m_expected_value;
    std::shared_ptr<op::v0::Constant> m_axes;
    std::shared_ptr<op::v0::Constant> m_signal;
};

class ReferenceRDFTLayerTest : public testing::TestWithParam<RDFTParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        if (params.m_signal != NULL) {
            function = CreateFunctionWithSignal(params);
        } else {
            function = CreateFunction(params);
        }

        inputData = {params.m_input_value};
        refOutData = {params.m_expected_value};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<RDFTParams>& obj) {
        const auto param = obj.param;
        std::ostringstream result;

        result << "input_shape1=" << param.m_input_shape << "; ";
        result << "output_shape=" << param.m_expected_shape << "; ";
        result << "input_type1=" << param.m_input_type << "; ";
        result << "output_type=" << param.m_expected_type << "; ";
        result << "transpose1=" << param.m_axes;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(RDFTParams& p) {
        auto in = std::make_shared<op::v0::Parameter>(p.m_input_type, p.m_input_shape);
        auto rdft = std::make_shared<op::v9::RDFT>(in, p.m_axes);

        return std::make_shared<ov::Model>(rdft, ParameterVector{in});
    }

    static std::shared_ptr<Model> CreateFunctionWithSignal(RDFTParams& p) {
        auto in = std::make_shared<op::v0::Parameter>(p.m_input_type, p.m_input_shape);
        auto rdft = std::make_shared<op::v9::RDFT>(in, p.m_axes, p.m_signal);

        return std::make_shared<ov::Model>(rdft, ParameterVector{in});
    }
};

TEST_P(ReferenceRDFTLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

static const std::vector<float> input_data = {
    0.10606491,  0.7454715,   0.57231355,  0.4582412,   0.3847059,   0.27398932, 0.66796243, 0.395475,
    0.2815729,   0.7799197,   0.59909415,  0.12294636,  0.38957402,  0.97498834, 0.46759892, 0.14017141,
    0.04206858,  0.7279963,   0.61560553,  0.9027321,   0.6226334,   0.2601217,  0.5555177,  0.40498647,
    0.14175586,  0.57774633,  0.52652127,  0.9385691,   0.9588788,   0.9844318,  0.23095612, 0.09707925,
    0.24574867,  0.6907577,   0.1974319,   0.8295272,   0.34612727,  0.51401484, 0.66115797, 0.9336245,
    0.06690067,  0.7468897,   0.39028263,  0.53575844,  0.060429193, 0.8913558,  0.77787375, 0.6701197,
    0.7350527,   0.6636995,   0.18176624,  0.8629976,   0.45142895,  0.6497297,  0.159372,   0.40598175,
    0.7988516,   0.7291543,   0.07090418,  0.7697132,   0.4972157,   0.7669217,  0.67975855, 0.13026066,
    0.6587437,   0.24532892,  0.24545169,  0.83795583,  0.105490535, 0.7264323,  0.94568557, 0.7216649,
    0.14389831,  0.7930531,   0.70895344,  0.9724701,   0.9775157,   0.49999878, 0.65569246, 0.26876843,
    0.63248956,  0.85201293,  0.5689624,   0.023386303, 0.5546464,   0.36860028, 0.9603114,  0.39123482,
    0.0380728,   0.89212376,  0.14387614,  0.63858676,  0.10003748,  0.8906635,  0.06681054, 0.7458642,
    0.45452347,  0.54724604,  0.6496482,   0.7818356,   0.6608355,   0.77711326, 0.24588613, 0.013456763,
    0.355845,    0.80388206,  0.027993264, 0.73677206,  0.52755004,  0.9052324,  0.54311025, 0.5367805,
    0.4131242,   0.7752338,   0.109669454, 0.13664648,  0.7828739,   0.9083969,  0.5247593,  0.7493595,
    0.19275227,  0.007190853, 0.6087981,   0.344136,    0.46909887,  0.41924855, 0.7072913,  0.19932869,
    0.5303847,   0.651384,    0.06686331,  0.9717932,   0.65702224,  0.11786682, 0.3154073,  0.88923013,
    0.5564087,   0.91047823,  0.28466642,  0.0934668,   0.88953066,  0.9919338,  0.18322521, 0.8185455,
    0.566391,    0.014207997, 0.29673064,  0.6347744,   0.6801958,   0.39601147, 0.34374171, 0.7216888,
    0.6152569,   0.76679546,  0.5860851,   0.4276813,   0.79339284,  0.13130653, 0.68764234, 0.053128112,
    0.02611321,  0.2982243,   0.7618372,   0.3331729,   0.5468192,   0.15707079, 0.28592056, 0.15286565,
    0.9368963,   0.350671,    0.4336494,   0.08934934,  0.41172776,  0.5850259,  0.70730376, 0.8598349,
    0.088788144, 0.26711187,  0.8002491,   0.19422275,  0.8312039,   0.5198718,  0.40111357, 0.98375803,
    0.77703434,  0.037818834, 0.704231,    0.689808,    0.17102319,  0.42153922, 0.7278252,  0.8030207,
    0.9101717,   0.0199644,   0.13768466,  0.55669,     0.17991355,  0.6720098,  0.7733328,  0.20881335};

static const std::vector<float> expected_rdft1d_results_1 = {
    4.6657147,   -1.1622906e-06, 0.21456887,    -0.14946258, -0.20476034,  -0.37063062,
    -0.31414136, 0.5099413,      -1.1779613,    0.07057127,  -0.64047664,  -1.0058284e-07,
    4.982774,    -1.1771917e-06, 0.6607505,     0.18829148,  -0.9772357,   1.4243596,
    0.8640026,   0.34923682,     0.33401352,    0.25859502,  -0.7548928,   8.940697e-08,
    5.9711604,   -1.4901161e-06, 0.5638976,     1.5429841,   -0.52065414,  0.24638398,
    -0.27140495, 0.5040715,      0.5360231,     0.3234269,   -0.36054826,  1.7508864e-07,
    4.7464237,   -1.2218952e-06, -0.29650804,   0.80609477,  -0.161426,    1.0022418,
    -0.50812817, 0.7967348,      0.4394225,     -0.1588624,  -1.3835809,   -7.4505806e-08,
    5.53836,     -1.7136335e-06, -0.38635445,   0.8284859,   -0.23278837,  -0.63777345,
    -0.93614054, 0.3215857,      -0.14075133,   -0.67071164, -1.4772836,   2.0861626e-07,
    5.0798974,   -1.5944242e-06, 0.056767445,   0.03468219,  -0.1497254,   -0.9672509,
    0.2603209,   0.69644475,     -0.9208536,    0.006730467, -1.7552528,   2.682209e-07,
    4.893558,    -1.6242266e-06, 0.6719861,     -0.13982919, 0.064845346,  -0.39896214,
    0.21785057,  -0.5099982,     -0.65526295,   1.4383471,   -0.52023906,  2.5331974e-07,
    6.687699,    -1.5497208e-06, -0.7423769,    0.09968524,  1.052381,     -0.21306956,
    0.5875206,   -0.3038844,     0.3991575,     -1.1895186,  0.17579001,   3.874302e-07,
    5.2818384,   -1.1026859e-06, 0.5087582,     0.106959194, 1.1816688,    -0.87592727,
    0.03740315,  0.5197907,      -1.3198637,    0.6398836,   0.22712436,   2.2351742e-08,
    5.0190897,   -1.5646219e-06, -0.087282926,  0.50819266,  -0.28002462,  0.29240948,
    -0.32303664, 0.38377762,     -0.0051696897, -0.99301195, -2.189299,    2.0861626e-07,
    5.0545654,   -1.5795231e-06, 0.9146397,     0.83839166,  0.870533,     0.17405808,
    -0.56308234, -0.7806684,     0.26397777,    0.6880482,   -1.4183462,   2.682209e-07,
    5.479953,    -1.2665987e-06, 0.49444157,    0.7534672,   -0.76784146,  -0.4507342,
    0.88815784,  0.6985409,      -0.2727425,    -0.25027415, -0.7328796,   2.682209e-07,
    4.1296124,   -5.662441e-07,  -0.46133032,   0.30635798,  -0.18225375,  0.42515472,
    -0.5484285,  0.9704039,      -0.35255045,   0.17549685,  0.8870368,    -3.1292439e-07,
    4.8632016,   -1.8924475e-06, -0.6926452,    0.025076404, -0.039108217, -1.7492937,
    -0.8120377,  -0.85315156,    -0.0022608787, 0.45002514,  -1.1024668,   3.501773e-07,
    5.4715447,   -1.4901161e-06, 1.1176248,     -0.2109062,  -0.27492502,  0.08983741,
    1.1903813,   -1.007312,      -0.20150042,   -0.83919466, -0.23939973,  4.917383e-07,
    5.1267176,   -9.983778e-07,  -0.44803134,   -0.8066604,  -0.3435102,   -0.41692197,
    -0.22457689, -0.1076939,     -0.29129186,   -1.1880502,  0.9255183,    -1.6391277e-07,
    3.8495903,   -5.5134296e-07, 0.09505272,    -0.12751618, -1.1264827,   0.5068884,
    -1.055237,   -0.19516481,    -0.34035242,   -0.15379356, 1.2655814,    -2.6077032e-07,
    4.4372616,   -9.23872e-07,   -0.72962606,   -0.23475963, -0.04278487,  1.1032158,
    -0.558924,   -0.5300043,     1.0578637,     -0.2466627,  0.44617313,   -7.8231096e-08,
    5.5374002,   -1.4156103e-06, 0.016273111,   -0.5989829,  -0.19913958,  0.013256833,
    1.8512837,   0.14526272,     -0.39700353,   -0.07573915, 0.23181,      2.9429793e-07,
    4.989425,    -1.4901161e-06, 1.0391837,     0.16554561,  -0.22647032,  -1.0689808,
    -0.84556,    -0.82779336,    0.9430445,     0.37618563,  0.4684292,    -9.685755e-08};

static const std::vector<float> expected_rdft1d_results_2 = {
    2.266797,  -8.195639e-08,  -0.37842733,  -0.41015846,  -0.48980892,  -0.10356337,
    2.5542018, -2.2351742e-08, -0.3223713,   0.671882,     0.54300576,   -0.35418037,
    1.985015,  -2.2351742e-08, -0.030243821, -0.20105253,  0.59431964,   0.07358998,
    1.4619737, -7.450581e-09,  -0.4356845,   0.35701087,   0.28208786,   -0.36424285,
    1.8002605, -1.1920929e-07, -0.43280697,  -0.56735414,  -0.30007166,  -0.541847,
    2.3052943, -1.2293458e-07, -0.39316025,  -0.5526293,   -0.30507135,  -0.6021758,
    2.7329001, -6.7055225e-08, 0.28245124,   -0.42586988,  -0.40586215,  0.4590181,
    3.3132548, -5.9604645e-08, 0.6297612,    0.3694744,    0.077824846,  -0.6248544,
    2.6314974, -2.9802322e-08, 0.58795106,   -0.60349375,  -0.3224758,   0.34408605,
    1.8399743, -9.685755e-08,  -0.43963802,  -0.079073176, -0.120658875, -1.0880115,
    2.0531366, -4.4703484e-08, 0.80112594,   -0.53726834,  -0.17560546,  -0.026561722,
    2.3779182, -9.685755e-08,  -0.21852754,  -0.19336401,  0.38734403,   -0.5954362,
    1.6219761, 7.450581e-09,   -0.43100592,  0.28373614,   0.101898566,  0.52321124,
    2.128953,  -1.4901161e-07, -0.1622684,   -0.94116735,  -0.7350497,   0.12695336,
    3.449626,  -8.940697e-08,  0.56062996,   -0.031283244, -0.06161648,  -0.8543532,
    3.033568,  -8.195639e-08,  -0.37023768,  -0.03989461,  -0.28719214,  -0.22382751,
    1.9661667, -1.4901161e-08, -0.59863573,  -0.015534669, -0.31916466,  0.55380434,
    2.227056,  -5.2154064e-08, -0.12656188,  0.6895717,    0.097157195,  0.19840825,
    3.5129817, -2.1234155e-07, 0.11158541,   0.5870459,    0.20993343,   -0.40297145,
    2.5986667, 0.0,            0.26602313,   -1.1560227,   0.2542065,    0.45556274};

template<class T>
static std::vector<T> convert(const std::vector<float>& v) {
    if (v.empty()) {
        return std::vector<T>();
    }

    size_t num_of_elems = v.size();
    std::vector<T> converted(num_of_elems);
    for (size_t i = 0; i < num_of_elems; ++i) {
        converted[i] = static_cast<T>(v[i]);
    }
    return converted;
}

template <class T>
static std::vector<T> convert(const std::vector<float16>& v) {
    if (v.empty()) {
        return std::vector<T>();
    }

    size_t num_of_elems = v.size();
    std::vector<T> converted(num_of_elems);
    for (size_t i = 0; i < num_of_elems; ++i) {
        converted[i] = static_cast<T>(v[i]);
    }
    return converted;
}

template <class T>
static std::vector<T> convert(const std::vector<bfloat16>& v) {
    if (v.empty()) {
        return std::vector<T>();
    }

    size_t num_of_elems = v.size();
    std::vector<T> converted(num_of_elems);
    for (size_t i = 0; i < num_of_elems; ++i) {
        converted[i] = static_cast<T>(v[i]);
    }
    return converted;
}
template <element::Type_t ET>
std::vector<RDFTParams> generateParamsForRDFT() {
    std::vector<RDFTParams> params{
        // rdft1d_eval
        RDFTParams(Shape{2, 10, 10},
                   Shape{2, 10, 6, 2},
                   ET,
                   ET,
                   input_data,
                   expected_rdft1d_results_1,
                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2}),
                   NULL),
        // rdft1d_eval_signal_size_1
        RDFTParams(Shape{2, 10, 10},
                   Shape{2, 10, 3, 2},
                   ET,
                   ET,
                   input_data,
                   expected_rdft1d_results_2,
                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2}),
                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {5})),
//        // dft1d_eval_1
//        DFTParams(Shape{4, 6, 8, 2},
//                  Shape{4, 6, 8, 2},
//                  ET,
//                  ET,
//                  input_data_1,
//                  expected_dft1d_results_1,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2}),
//                  NULL),
//        // dft1d_eval_i32
//        DFTParams(Shape{2, 10, 10, 2},
//                  Shape{2, 10, 10, 2},
//                  ET,
//                  ET,
//                  input_data,
//                  expected_dft1d_results,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i32, Shape{1}, {2}),
//                  NULL),
//        // dft2d_eval_1
//        DFTParams(Shape{4, 6, 8, 2},
//                  Shape{4, 6, 8, 2},
//                  ET,
//                  ET,
//                  input_data_1,
//                  expected_dft2d_results_1,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {1, 2}),
//                  NULL),
//        // dft2d_eval
//        DFTParams(Shape{2, 10, 10, 2},
//                  Shape{2, 10, 10, 2},
//                  ET,
//                  ET,
//                  input_data,
//                  expected_dft2d_results,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {1, 2}),
//                  NULL),
//        // dft2d_eval_i32
//        DFTParams(Shape{2, 10, 10, 2},
//                  Shape{2, 10, 10, 2},
//                  ET,
//                  ET,
//                  input_data,
//                  expected_dft2d_results,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i32, Shape{2}, {1, 2}),
//                  NULL),
//        // dft3d_eval_1
//        DFTParams(Shape{4, 6, 8, 2},
//                  Shape{4, 6, 8, 2},
//                  ET,
//                  ET,
//                  input_data_1,
//                  expected_dft3d_results_1,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{3}, {0, 1, 2}),
//                  NULL),
//        // dft3d_eval
//        DFTParams(Shape{2, 10, 10, 2},
//                  Shape{2, 10, 10, 2},
//                  ET,
//                  ET,
//                  input_data,
//                  expected_dft3d_results,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{3}, {0, 1, 2}),
//                  NULL),
//        // dft3d_eval_i32
//        DFTParams(Shape{2, 10, 10, 2},
//                  Shape{2, 10, 10, 2},
//                  ET,
//                  ET,
//                  input_data,
//                  expected_dft3d_results,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i32, Shape{3}, {0, 1, 2}),
//                  NULL),
//        // dft1d_signal_size_eval
//        DFTParams(Shape{2, 10, 10, 2},
//                  Shape{2, 10, 10, 2},
//                  ET,
//                  ET,
//                  input_data,
//                  expected_dft1d_signal_size_results,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {-2}),
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {20})),
//        // dft2d_signal_size_eval_1
//        DFTParams(Shape{4, 6, 8, 2},
//                  Shape{4, 6, 8, 2},
//                  ET,
//                  ET,
//                  input_data_1,
//                  expected_dft2d_signal_size_results_1,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {0, 2}),
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {5, 9})),
//        // dft2d_signal_size_eval_2
//        DFTParams(Shape{4, 6, 8, 2},
//                  Shape{4, 6, 8, 2},
//                  ET,
//                  ET,
//                  input_data_1,
//                  expected_dft2d_signal_size_results_2,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {0, 1}),
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {4, 6})),
//        // dft2d_signal_size_eval_3
//        DFTParams(Shape{4, 6, 8, 2},
//                  Shape{4, 6, 8, 2},
//                  ET,
//                  ET,
//                  input_data_1,
//                  expected_dft2d_signal_size_results_3,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {0, 2}),
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {3, 4})),
//        // dft2d_signal_size_eval_4
//        DFTParams(Shape{4, 6, 8, 2},
//                  Shape{4, 6, 8, 2},
//                  ET,
//                  ET,
//                  input_data_1,
//                  expected_dft2d_signal_size_results_4,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {0, 2}),
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {4, 8})),
//        // dft2d_signal_size_eval_5
//        DFTParams(Shape{4, 6, 8, 2},
//                  Shape{4, 6, 8, 2},
//                  ET,
//                  ET,
//                  input_data_1,
//                  expected_dft2d_signal_size_results_5,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {0, 2}),
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {5, 4})),
//        // dft3d_signal_size_eval
//        DFTParams(Shape{4, 6, 8, 2},
//                  Shape{4, 6, 8, 2},
//                  ET,
//                  ET,
//                  input_data_1,
//                  expected_dft3d_signal_size_results,
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{3}, {0, 1, 2}),
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{3}, {3, 7, 5})),
    };

    return params;
}

//template <element::Type_t ET>
//std::vector<DFTParams> generateParamsForDFT_float16() {
//    using T = typename element_type_traits<ET>::value_type;
//
//    std::vector<DFTParams> params{
//        // dft1d_eval_float16
//        DFTParams(Shape{2, 10, 10, 2},
//                  Shape{2, 10, 10, 2},
//                  ET,
//                  ET,
//                  convert<T>(input_data),
//                  convert<T>(expected_dft1d_float16_results),
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2}),
//                  NULL),
//        // dft2d_eval_float16
//        DFTParams(Shape{2, 10, 10, 2},
//                  Shape{2, 10, 10, 2},
//                  ET,
//                  ET,
//                  convert<T>(input_data),
//                  convert<T>(expected_dft2d_float16_results),
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {1, 2}),
//                  NULL),
//        // dft3d_eval_float16
//        DFTParams(Shape{2, 10, 10, 2},
//                  Shape{2, 10, 10, 2},
//                  ET,
//                  ET,
//                  convert<T>(input_data),
//                  convert<T>(expected_dft3d_float16_results),
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{3}, {0, 1, 2}),
//                  NULL),
//    };
//
//    return params;
//}
//
//template <element::Type_t ET>
//std::vector<DFTParams> generateParamsForDFT_bfloat16() {
//    using T = typename element_type_traits<ET>::value_type;
//
//    std::vector<DFTParams> params{
//        // dft1d_eval_bfloat16
//        DFTParams(Shape{2, 10, 10, 2},
//                  Shape{2, 10, 10, 2},
//                  ET,
//                  ET,
//                  convert<T>(input_data),
//                  convert<T>(expected_dft1d_bfloat16_results),
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2}),
//                  NULL),
//        // dft2d_eval_bfloat16
//        DFTParams(Shape{2, 10, 10, 2},
//                  Shape{2, 10, 10, 2},
//                  ET,
//                  ET,
//                  convert<T>(input_data),
//                  convert<T>(expected_dft2d_bfloat16_results),
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {1, 2}),
//                  NULL),
//        // dft3d_eval_bfloat16
//        DFTParams(Shape{2, 10, 10, 2},
//                  Shape{2, 10, 10, 2},
//                  ET,
//                  ET,
//                  convert<T>(input_data),
//                  convert<T>(expected_dft3d_bfloat16_results),
//                  op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{3}, {0, 1, 2}),
//                  NULL),
//    };
//
//    return params;
//}

std::vector<RDFTParams> generateCombinedParamsForRDFT() {
    const std::vector<std::vector<RDFTParams>> allTypeParams{
        generateParamsForRDFT<element::Type_t::f32>(),
//        generateParamsForDFT_float16<element::Type_t::f16>(),
//        generateParamsForDFT_bfloat16<element::Type_t::bf16>(),
    };

    std::vector<RDFTParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_RDFT_With_Hardcoded_Refs,
    ReferenceRDFTLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForRDFT()),
    ReferenceRDFTLayerTest::getTestCaseName);
}  // namespace
