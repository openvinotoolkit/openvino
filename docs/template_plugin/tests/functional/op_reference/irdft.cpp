// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <iostream>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/irdft.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct IRDFTParams {
    template <class T>
    IRDFTParams(const Shape& input_shape,
                const Shape& expected_shape,
                const element::Type_t& input_type,
                const element::Type_t& expected_type,
                const std::vector<T>& input_value,
                const std::vector<T>& expected_value,
                const std::shared_ptr<op::v0::Constant>& axes,
                const std::shared_ptr<op::v0::Constant>& signal) {
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

class ReferenceIRDFTLayerTest : public testing::TestWithParam<IRDFTParams>, public CommonReferenceTest {
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

    static std::string getTestCaseName(const testing::TestParamInfo<IRDFTParams>& obj) {
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
    static std::shared_ptr<Model> CreateFunction(IRDFTParams& p) {
        auto in = std::make_shared<op::v0::Parameter>(p.m_input_type, p.m_input_shape);
        auto irdft = std::make_shared<op::v9::IRDFT>(in, p.m_axes);

        return std::make_shared<ov::Model>(irdft, ParameterVector{in});
    }

    static std::shared_ptr<Model> CreateFunctionWithSignal(IRDFTParams& p) {
        auto in = std::make_shared<op::v0::Parameter>(p.m_input_type, p.m_input_shape);
        auto irdft = std::make_shared<op::v9::IRDFT>(in, p.m_axes, p.m_signal);

        return std::make_shared<ov::Model>(irdft, ParameterVector{in});
    }
};

TEST_P(ReferenceIRDFTLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

static const std::vector<float> input_data = {
    0.10606491,  0.7454715,   0.57231355,  0.4582412,    0.3847059,   0.27398932,  0.66796243,  0.395475,
    0.2815729,   0.7799197,   0.59909415,  0.12294636,   0.38957402,  0.97498834,  0.46759892,  0.14017141,
    0.04206858,  0.7279963,   0.61560553,  0.9027321,    0.6226334,   0.2601217,   0.5555177,   0.40498647,
    0.14175586,  0.57774633,  0.52652127,  0.9385691,    0.9588788,   0.9844318,   0.23095612,  0.09707925,
    0.24574867,  0.6907577,   0.1974319,   0.8295272,    0.34612727,  0.51401484,  0.66115797,  0.9336245,
    0.06690067,  0.7468897,   0.39028263,  0.53575844,   0.060429193, 0.8913558,   0.77787375,  0.6701197,
    0.7350527,   0.6636995,   0.18176624,  0.8629976,    0.45142895,  0.6497297,   0.159372,    0.40598175,
    0.7988516,   0.7291543,   0.07090418,  0.7697132,    0.4972157,   0.7669217,   0.67975855,  0.13026066,
    0.6587437,   0.24532892,  0.24545169,  0.83795583,   0.105490535, 0.7264323,   0.94568557,  0.7216649,
    0.14389831,  0.7930531,   0.70895344,  0.9724701,    0.9775157,   0.49999878,  0.65569246,  0.26876843,
    0.63248956,  0.85201293,  0.5689624,   0.023386303,  0.5546464,   0.36860028,  0.9603114,   0.39123482,
    0.0380728,   0.89212376,  0.14387614,  0.63858676,   0.10003748,  0.8906635,   0.06681054,  0.7458642,
    0.45452347,  0.54724604,  0.6496482,   0.7818356,    0.6608355,   0.77711326,  0.24588613,  0.013456763,
    0.355845,    0.80388206,  0.027993264, 0.73677206,   0.52755004,  0.9052324,   0.54311025,  0.5367805,
    0.4131242,   0.7752338,   0.109669454, 0.13664648,   0.7828739,   0.9083969,   0.5247593,   0.7493595,
    0.19275227,  0.007190853, 0.6087981,   0.344136,     0.46909887,  0.41924855,  0.7072913,   0.19932869,
    0.5303847,   0.651384,    0.06686331,  0.9717932,    0.65702224,  0.11786682,  0.3154073,   0.88923013,
    0.5564087,   0.91047823,  0.28466642,  0.0934668,    0.88953066,  0.9919338,   0.18322521,  0.8185455,
    0.566391,    0.014207997, 0.29673064,  0.6347744,    0.6801958,   0.39601147,  0.34374171,  0.7216888,
    0.6152569,   0.76679546,  0.5860851,   0.4276813,    0.79339284,  0.13130653,  0.68764234,  0.053128112,
    0.02611321,  0.2982243,   0.7618372,   0.3331729,    0.5468192,   0.15707079,  0.28592056,  0.15286565,
    0.9368963,   0.350671,    0.4336494,   0.08934934,   0.41172776,  0.5850259,   0.70730376,  0.8598349,
    0.088788144, 0.26711187,  0.8002491,   0.19422275,   0.8312039,   0.5198718,   0.40111357,  0.98375803,
    0.77703434,  0.037818834, 0.704231,    0.689808,     0.17102319,  0.42153922,  0.7278252,   0.8030207,
    0.9101717,   0.0199644,   0.13768466,  0.55669,      0.17991355,  0.6720098,   0.7733328,   0.20881335,
    0.037048724, 0.23186281,  0.51947355,  0.048872072,  0.012902894, 0.726808,    0.7399658,   0.6120117,
    0.47996572,  0.044601407, 0.38995656,  0.7136875,    0.54200786,  0.9137477,   0.019450456, 0.87729025,
    0.63369817,  0.94858027,  0.5828617,   0.2594391,    0.5001518,   0.14265054,  0.73658663,  0.52912134,
    0.43768325,  0.96134335,  0.92333144,  0.044408076,  0.7894684,   0.010386336, 0.30753252,  0.18513045,
    0.65942764,  0.46920168,  0.52875453,  0.16858862,   0.009865526, 0.69832677,  0.4512412,   0.75672126,
    0.88412845,  0.9558565,   0.28965947,  0.36676157,   0.21277316,  0.20550993,  0.43206176,  0.6940698,
    0.9557682,   0.72806126,  0.6812339,   0.40266454,   0.22059599,  0.58443034,  0.20210509,  0.06344778,
    0.69618744,  0.12523267,  0.3263705,   0.44827136,   0.62200356,  0.845199,    0.68803686,  0.23659822,
    0.3520896,   0.27610266,  0.4992663,   0.0038073587, 0.7571601,   0.38161016,  0.52645093,  0.67807734,
    0.58921075,  0.9000049,   0.20419674,  0.63863295,   0.7740024,   0.51551616,  0.1006969,   0.7047221,
    0.59895676,  0.73848796,  0.17399798,  0.76440364,   0.52670205,  0.9820852,   0.9655277,   1.3231402e-05,
    0.7719922,   0.35906568,  0.29054508,  0.18124384,   0.5604661,   0.84750974,  0.98948747,  0.009793862,
    0.7184191,   0.5560748,   0.6952493,   0.6732593,    0.3306898,   0.6790913,   0.41128764,  0.34593266,
    0.94296855,  0.7348507,   0.24478768,  0.94024557,   0.05405676,  0.06466125,  0.36244348,  0.07942984,
    0.10619422,  0.09412837,  0.9053611,   0.22870538,   0.9237487,   0.20986171,  0.5067282,   0.29709867,
    0.53138554,  0.189101,    0.4786443,   0.88421875,   0.95664364,  0.46239087,  0.53202313,  0.78942096,
    0.137265,    0.20953515,  0.9436369,   0.9486073,    0.9843343,   0.022179728, 0.5347736,   0.6712253,
    0.10952409,  0.8710814,   0.011709057, 0.24770738,   0.6256841,   0.7619287,   0.15110113,  0.049545255,
    0.39370516,  0.8843956,   0.26512384,  0.93957156,   0.95622736,  0.20436367,  0.8620167,   0.8722808,
    0.01225669,  0.89293826,  0.008706713, 0.027482228,  0.9945729,   0.15698057,  0.5600313,   0.49245444,
    0.875277,    0.97211605,  0.6500988,   0.65309316,   0.41882965,  0.29644313,  0.3097659,   0.597524,
    0.80133235,  0.8590142,   0.34570643,  0.16331969,   0.97009367,  0.6723736,   0.97456884,  0.042129718,
    0.6576646,   0.32067406,  0.6531016,   0.33883506,   0.58958584,  0.376756,    0.2955236,   0.6379718,
    0.8621009,   0.6601814,   0.13407534,  0.4242811,    0.08052258,  0.8271176,   0.51772064,  0.20591077,
    0.5819514,   0.16577412,  0.5938862,   0.5776045,    0.5685978,   0.11109683,  0.59757715,  0.12524122};

static const std::vector<float> expected_irdft1d_results_1 = {
    0.4184138,     -0.27598706,   -0.006450875, -0.07465666,   -0.005245729,  -0.08811711,
    0.002804786,   -0.16338626,   0.15419002,   -0.16264628,   -0.14073074,   0.1105442,
    0.051571753,   0.0052294377,  -0.036425564, 0.059583426,   -0.07729218,   0.33466592,
    0.4272035,     -0.36593032,   -0.03475378,  -0.17882892,   0.22039334,    0.109648176,
    -0.07702944,   -0.10877094,   0.04551638,   0.018091345,   -0.073307976,  -0.05411123,
    0.12121431,    0.044729933,   0.0029542628, 0.0607259,     0.00970519,    0.45518342,
    0.40266195,    -0.45727646,   -0.08459554,  -0.18407981,   0.007835884,   0.012076079,
    0.09952847,    -0.02435577,   -0.13015305,  0.05938517,    -0.16664405,   0.01037844,
    -0.0694556,    0.15436485,    0.026946524,  0.015212444,   -0.017221835,  0.41229296,
    0.5602166,     -0.48106226,   0.20158666,   -0.025649427,  0.09381822,    -0.032855563,
    -0.1539908,    0.09526638,    -0.03560306,  -0.085937515,  0.04775922,    0.122609,
    -0.0842911,    -0.094522804,  -0.05168595,  0.051323075,   -0.0013554362, 0.37159055,
    0.39203426,    -0.29385728,   0.20954835,   -0.12626143,   0.024378777,   -0.021371195,
    0.07726196,    0.07357122,    0.030629426,  -0.06680652,   -0.10689789,   0.04154566,
    0.09784284,    -0.009246837,  -0.05904018,  -0.09654971,   -0.024688551,  0.4903965,
    0.399872,      -0.45473805,   0.0067230994, -0.032742612,  0.19891962,    -0.053130303,
    0.08724912,    -0.0153728165, 0.11990947,   0.13564137,    -0.13145526,   0.09186138,
    -0.14248294,   -0.007876493,  0.00855005,   0.04815747,    0.04551228,    0.35623792,
    0.4611091,     -0.3240845,    0.09889657,   -0.13251431,   0.006920836,   -0.15288353,
    0.07671951,    0.032530483,   -0.14757682,  0.05206634,    -0.037690982,  -0.13693814,
    -0.003487827,  0.1930467,     -0.1226512,   0.048372496,   -0.09352949,   0.37444714,
    0.5392893,     -0.39019704,   0.056647725,  -0.03125781,   -0.07282965,   0.0038826184,
    -0.06079346,   -0.101931386,  -0.025743863, 0.14959985,    0.13141608,    0.12182322,
    0.08836199,    0.06809068,    0.11807177,   0.021481361,   0.014166607,   0.25945252,
    0.50956905,    -0.19826593,   0.06448524,   -0.12395297,   0.005832309,   0.009144372,
    -0.16877219,   -0.14880641,   0.06900328,   -0.065727405,  0.034049492,   -0.1770428,
    0.02947952,    -0.015928954,  0.012717932,  0.10259341,    -0.14318335,   0.23091878,
    0.5345848,     -0.24913073,   -0.042740896, -0.064899355,  0.014478589,   -0.30438924,
    0.09226764,    -0.031547274,  -0.12111721,  0.01069157,    0.18909042,    0.11838305,
    0.17870422,    0.051106248,   -0.03115296,  0.13249981,    -0.011846168,  0.3662214,
    0.40526384,    -0.41233394,   0.10635306,   -0.24066877,   0.0834809,     0.03569686,
    0.19916648,    -0.032840244,  -0.15838136,  -0.030353047,  -0.11691042,   -0.00541403,
    -0.073787585,  -0.053250197,  0.028788418,  0.11406694,    -0.16401787,   0.35218972,
    0.54092723,    -0.1033598,    -0.09457661,  -0.21597207,   -0.09317484,   -0.0020268974,
    0.1819915,     -0.10256999,   0.07521623,   -0.06392271,   -0.094392076,  -0.090536445,
    -0.03676716,   0.109799206,   0.03676027,   0.20007852,    -0.040287875,  0.29296526,
    0.47729248,    -0.27816322,   -0.103723355, 0.0046423557,  0.14799257,    0.071483314,
    -0.069877684,  -0.036883753,  0.02097966,   0.08434933,    -0.031067278,  0.13464029,
    0.011887407,   0.074179865,   0.14406723,   -0.06652063,   0.0076984563,  0.29115114,
    0.5279736,     -0.26480842,   0.13792582,   -0.045278277,  -0.004051553,  -0.0064324946,
    -0.0018670667, 0.009724317,   -0.0623539,   0.09057393,    -0.13936117,   0.010369011,
    -0.042830903,  0.06606974,    0.12217537,   0.07157013,    -0.17625968,   0.3288651,
    0.62713766,    -0.3314877,    -0.065081455, -0.25215414,   -0.19212066,   0.053122636,
    0.105015196,   -0.042499457,  0.20603144,   0.012208419,   -0.11903475,   0.05109081,
    -0.007787534,  0.18078992,    0.065069996,  0.08887646,    0.027873179,   0.19190653,
    0.48550618,    -0.2635743,    -0.013126226, -0.17094027,   -0.05851199,   0.042673897,
    -0.077439696,  0.046145827,   0.0009360272, 0.0016744527,  0.10504647,    0.10277959,
    -0.15150274,   -0.087410055,  -0.1509898,   0.09372833,    0.27879104,    0.14690335,
    0.54982114,    -0.28986064,   -0.11191738,  -0.045548216,  -0.07889767,   -0.26967183,
    0.07283146,    0.029252125,   -0.13779576,  -0.0046080146, 0.20243134,    0.10006317,
    -0.026684487,  0.037547383,   -0.13398017,  0.2875392,     -0.06426069,   0.41512486,
    0.47072962,    -0.4029027,    -0.09929869,  -0.10194954,   0.121289134,   0.11328362,
    0.06673959,    0.036691997,   -0.012112199, 0.12251561,    0.0124647925,  0.13746831,
    -0.2473899,    -0.10970233,   0.24933825,   -0.09391294,   0.031097075,   0.3313342,
    0.6546941,     -0.35889223,   -0.03856469,  0.06294843,    -0.012368825,  0.028103853,
    -0.0611313,    -0.19783688,   0.10685261,   0.07542985,    -0.07086358,   0.033852134,
    0.08778654,    0.11711612,    -0.005260711, 0.11875999,    0.103045076,   0.23160617,
    0.46977314,    -0.32612872,   -0.029793505, -0.048244745,  -0.049801487,  -0.13720272,
    -0.11914676,   0.113162495,   0.100396454,  0.060885735,   0.042446703,   -0.015682518,
    0.08590958,    -0.033345744,  -0.06670459,  0.13531905,    0.16050199,    0.24724133};

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
std::vector<IRDFTParams> generateParamsForIRDFT() {
    std::vector<IRDFTParams> params{
        // irdft1d_eval
        IRDFTParams(Shape{2, 10, 10, 2},
                    Shape{2, 10, 18},
                    ET,
                    ET,
                    input_data,
                    expected_irdft1d_results_1,
                    op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2}),
                    NULL),
//        // rdft1d_eval_signal_size_0
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft1d_results_1,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {10})),
//        // rdft1d_eval_signal_size_0_1
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft1d_results_1,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {-1})),
//        // rdft1d_eval_1
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft1d_results_1,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {-1}),
//                   NULL),
//        // rdft1d_eval_signal_size_1
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 3, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft1d_results_2,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {5})),
//        // rdft1d_eval_signal_size_1_1
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 3, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft1d_results_2,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {-1}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {5})),
//        // rdft1d_eval_signal_size_2
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 7, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft1d_results_3,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {12})),
//        // rdft1d_eval_signal_size_2_1
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 7, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft1d_results_3,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {-1}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {12})),
//        // rdft2d_eval_1
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {1, 2}),
//                   NULL),
//        // rdft2d_eval_1_positive_negative_axes
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {1, -1}),
//                   NULL),
//        // rdft2d_eval_1_negative_positive_axes
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {-2, 2}),
//                   NULL),
//        // rdft2d_eval_1_negative_negative_axes
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {-2, -1}),
//                   NULL),
//        // rdft2d_eval_1_signal_size_0_s10_10
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {1, 2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {10, 10})),
//        // rdft2d_eval_1_signal_size_0_s10_10_positive_negative_axes
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {1, -1}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {10, 10})),
//        // rdft2d_eval_1_signal_size_0_s10_10_negative_positive_axes
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {-2, 2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {10, 10})),
//        // rdft2d_eval_1_signal_size_0_s10_10_negative_negative_axes
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {-2, -1}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {10, 10})),
//        // rdft2d_eval_1_signal_size_0_s10_m1
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {1, 2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {10, -1})),
//        // rdft2d_eval_1_signal_size_0_sm1_10
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {1, 2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {-1, 10})),
//        // rdft2d_eval_1_signal_size_0_sm1_m1
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {1, 2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {-1, -1})),
//        // rdft2d_eval_2_signal_size
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 5, 7, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results_2,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {1, 2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {5, 12})),
//        // rdft2d_eval_2_signal_size_positive_negative_axes
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 5, 7, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results_2,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {1, -1}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {5, 12})),
//        // rdft2d_eval_2_signal_size_negative_positive_axes
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 5, 7, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results_2,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {-2, 2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {5, 12})),
//        // rdft2d_eval_2_signal_size_negative_negative_axes
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 5, 7, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft2d_results_2,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {-2, -1}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {5, 12})),
//        // rdft3d_eval_1
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft3d_results,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{3}, {0, 1, 2}),
//                   NULL),
//        // rdft3d_eval_1_negative_axes_and_signal_size
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{2, 10, 6, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft3d_results,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{3}, {-3, 1, 2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{3}, {-1, 10, -1})),
//        // rdft3d_eval_2
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{4, 5, 7, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft3d_results_2,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{3}, {0, 1, 2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{3}, {4, 5, 12})),
//        // rdft3d_eval_2_negative_axes
//        IRDFTParams(Shape{2, 10, 10},
//                   Shape{4, 5, 7, 2},
//                   ET,
//                   ET,
//                   input_data,
//                   expected_rdft3d_results_2,
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{3}, {-3, -2, 2}),
//                   op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{3}, {4, 5, 12})),
    };

    return params;
}

std::vector<IRDFTParams> generateCombinedParamsForIRDFT() {
    const std::vector<std::vector<IRDFTParams>> allTypeParams{
        generateParamsForIRDFT<element::Type_t::f32>()
    };

    std::vector<IRDFTParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_IRDFT_With_Hardcoded_Refs,
    ReferenceIRDFTLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForIRDFT()),
    ReferenceIRDFTLayerTest::getTestCaseName);
}  // namespace
