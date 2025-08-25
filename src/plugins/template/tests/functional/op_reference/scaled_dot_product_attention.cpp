// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scaled_dot_product_attention.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct SDPAParams {
    PartialShape qShape;
    PartialShape kShape;
    PartialShape vShape;
    PartialShape attentionMaskShape;
    PartialShape outputShape;
    bool isCausal;
    std::string testcaseName;
    reference_tests::Tensor qData;
    reference_tests::Tensor kData;
    reference_tests::Tensor vData;
    reference_tests::Tensor attentionMaskData;
    reference_tests::Tensor expectedOutputData;
    PartialShape sinkShape;
    reference_tests::Tensor sinkData;
};

template <typename T, typename TMask>
SDPAParams PrepareTestCaseParams(const PartialShape& qShape,
                                 const PartialShape& kShape,
                                 const PartialShape& vShape,
                                 const PartialShape& attentionMaskShape,
                                 const PartialShape& outputShape,
                                 bool isCausal,
                                 const std::vector<T>& qData,
                                 const std::vector<T>& kData,
                                 const std::vector<T>& vData,
                                 const std::vector<TMask>& attentionMaskData,
                                 const std::vector<T>& expectedOutputData,
                                 const std::string& description,
                                 const PartialShape& sinkShape = {0},
                                 const std::vector<T>& sinkData = {}) {
    SDPAParams ret;
    const auto elementType = element::from<T>();

    ret.qShape = qShape;
    ret.kShape = kShape;
    ret.vShape = vShape;
    ret.attentionMaskShape = attentionMaskShape;
    ret.outputShape = outputShape;
    ret.isCausal = isCausal;
    ret.testcaseName = description;
    ret.qData = reference_tests::Tensor(elementType, qShape.to_shape(), qData);
    ret.kData = reference_tests::Tensor(elementType, kShape.to_shape(), kData);
    ret.vData = reference_tests::Tensor(elementType, vShape.to_shape(), vData);
    ret.attentionMaskData =
        reference_tests::Tensor(element::from<TMask>(), attentionMaskShape.to_shape(), attentionMaskData);
    ret.expectedOutputData = reference_tests::Tensor(elementType, outputShape.to_shape(), expectedOutputData);
    ret.sinkShape = sinkShape;
    if (!sinkData.empty()) {
        ret.sinkData = reference_tests::Tensor(elementType, sinkShape.to_shape(), sinkData);
    }
    return ret;
}

class ReferenceSDPATest : public testing::TestWithParam<SDPAParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.qData.data, params.kData.data, params.vData.data};
        if (params.attentionMaskShape.size() != 0) {
            inputData.push_back(params.attentionMaskData.data);
        }
        if (shape_size(params.sinkShape.get_shape()) != 0) {
            inputData.push_back(params.sinkData.data);
        }
        refOutData = {params.expectedOutputData.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SDPAParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "type=" << param.qData.data.get_element_type();
        result << "_qShape=" << param.qShape;
        result << "_kShape=" << param.kShape;
        result << "_vShape=" << param.vShape;
        result << "_attentionMaskShape=" << param.attentionMaskShape;
        result << "_outputShape=" << param.outputShape;
        result << "_isCausal=" << param.isCausal;
        if (param.sinkData.data) {
            result << "_sinkShape=" << param.sinkShape;
        }
        result << "_=" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const SDPAParams& params) {
        const auto q = std::make_shared<op::v0::Parameter>(params.qData.data.get_element_type(), params.qShape);
        const auto k = std::make_shared<op::v0::Parameter>(params.kData.data.get_element_type(), params.kShape);
        const auto v = std::make_shared<op::v0::Parameter>(params.vData.data.get_element_type(), params.vShape);

        OutputVector inputs = {q, k, v};
        ParameterVector paramsVec = {q, k, v};

        if (params.attentionMaskShape.size() != 0) {
            const auto attentionMask =
                std::make_shared<op::v0::Parameter>(params.attentionMaskData.data.get_element_type(),
                                                    params.attentionMaskShape);
            inputs.push_back(attentionMask);
            paramsVec.push_back(attentionMask);
        } else {
            const auto attentionMask = std::make_shared<ov::op::v0::Constant>(params.qData.data.get_element_type(),
                                                                              ov::Shape{},
                                                                              std::vector<float>{0});
            inputs.push_back(attentionMask);
        }

        if (shape_size(params.sinkShape.get_shape()) != 0) {
            const auto scale = std::make_shared<ov::op::v0::Constant>(params.qData.data.get_element_type(),
                                                                      ov::Shape{},
                                                                      std::vector<float>{1.0});
            inputs.push_back(scale);

            const auto sink =
                std::make_shared<op::v0::Parameter>(params.sinkData.data.get_element_type(), params.sinkShape);
            inputs.push_back(sink);
            paramsVec.push_back(sink);
        }

        const auto op = std::make_shared<op::v13::ScaledDotProductAttention>(inputs, params.isCausal);

        return std::make_shared<Model>(OutputVector{op}, paramsVec);
    }
};

TEST_P(ReferenceSDPATest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<SDPAParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<SDPAParams> params;

#define TEST_DATA(q_shape,                                                              \
                  k_shape,                                                              \
                  v_shape,                                                              \
                  attention_mask_shape,                                                 \
                  output_shape,                                                         \
                  is_causal,                                                            \
                  is_attention_mask_bool,                                               \
                  q_data,                                                               \
                  k_data,                                                               \
                  v_data,                                                               \
                  attention_mask_data,                                                  \
                  expected_output_data,                                                 \
                  description)                                                          \
    {                                                                                   \
        using TMask = typename std::conditional<is_attention_mask_bool, char, T>::type; \
        std::vector<TMask> attention_mask_data_vec = attention_mask_data;               \
        params.push_back(PrepareTestCaseParams<T, TMask>(q_shape,                       \
                                                         k_shape,                       \
                                                         v_shape,                       \
                                                         attention_mask_shape,          \
                                                         output_shape,                  \
                                                         is_causal,                     \
                                                         q_data,                        \
                                                         k_data,                        \
                                                         v_data,                        \
                                                         attention_mask_data_vec,       \
                                                         expected_output_data,          \
                                                         description));                 \
    }

#include "unit_test_utils/tests_data/scaled_dot_product_attention_data.h"
#undef TEST_DATA

    return params;
}

std::vector<SDPAParams> generateCombinedParams() {
    const std::vector<std::vector<SDPAParams>> generatedParams{generateParams<element::Type_t::f32>(),
                                                               generateParams<element::Type_t::f16>(),
                                                               generateParams<element::Type_t::f64>()};
    std::vector<SDPAParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_SDPA_With_Hardcoded_Refs,
                         ReferenceSDPATest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceSDPATest::getTestCaseName);

template <element::Type_t ET>
std::vector<SDPAParams> generateParamsWithSink() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<SDPAParams> params;

    PartialShape q_shape{2, 3, 4, 5};
    PartialShape k_shape{2, 1, 4, 5};
    PartialShape v_shape{2, 1, 4, 5};
    PartialShape output_shape{2, 3, 4, 5};

    std::vector<T> q_data = {
        -1.1258398294448853,  -1.152360200881958,    -0.2505785822868347,  -0.4338788092136383,   0.8487103581428528,
        0.946298360824585,    -0.843676745891571,    -0.6135830879211426,  0.03159274160861969,   -0.4926769733428955,
        -1.5311843156814575,  -1.2341350317001343,   1.8197252750396729,   -0.5515286922454834,   -0.5692480802536011,
        -1.881689190864563,   -0.049727022647857666, -1.0449786186218262,  -0.9565008282661438,   0.03353185951709747,
        0.6920091509819031,   -0.31601276993751526,  -2.1152193546295166,  0.32227492332458496,   -1.2633347511291504,
        0.2484147548675537,   0.4396958351135254,    0.11241118609905243,  0.6407923698425293,    0.441156268119812,
        0.9199714064598083,   1.1108161211013794,    1.2898740768432617,   -1.4781739711761475,   2.567232847213745,
        0.7100865840911865,   1.6458669900894165,    -1.3601689338684082,  0.34456542134284973,   0.5198677182197571,
        0.34998318552970886,  0.30813392996788025,   0.11984150856733322,  1.237657904624939,     1.1167771816253662,
        -0.10230965167284012, 0.7924439907073975,    -0.28966769576072693, 0.05250748619437218,   0.5228604674339294,
        -0.4731197953224182,  0.3355507552623749,    -1.6293259859085083,  -0.54974365234375,     -0.47983425855636597,
        -2.6133224964141846,  -1.6964744329452515,   -0.22824178636074066, 0.2799549996852875,    -0.7015236020088196,
        -0.2472781538963318,  -1.3526537418365479,   -1.6959311962127686,  0.5666506290435791,    0.7935083508491516,
        2.3022053241729736,   -1.4688938856124878,   -1.586688756942749,   -0.6730899214744568,   0.8728312253952026,
        -0.4996815323829651,  -1.0669803619384766,   1.114939570426941,    -0.14067143201828003,  0.8057535886764526,
        1.0366867780685425,   -0.6036701202392578,   -1.2787652015686035,  0.0929502323269844,    -0.6660997271537781,
        0.5988394618034363,   -1.5550950765609741,   -0.3413603901863098,  1.85300612449646,      0.750189483165741,
        1.0553574562072754,   0.17784371972084045,   -0.23033547401428223, -0.3917543888092041,   0.5432947278022766,
        -0.09334823489189148, 0.6870502233505249,    -0.8383153676986694,  0.0008918217499740422, 0.8418940901756287,
        0.6080471873283386,   -0.7300198674201965,   1.3750379085540771,   0.659631073474884,     0.4765571057796478,
        -0.5854975581169128,  -0.1733967512845993,   0.18347793817520142,  1.3893661499023438,    1.586334228515625,
        -0.3951575458049774,  -0.4462171792984009,   0.7440207004547119,   1.5209795236587524,    3.4105026721954346,
        -0.4000341594219208,  1.0394619703292847,    0.3581531047821045,   -0.24600094556808472,  2.302516460418701,
        -1.0163074731826782,  0.18036697804927826,   0.10833186656236649,  -0.7548232674598694,   0.24431852996349335};
    std::vector<T> k_data = {
        1.1403506994247437,  -0.08988206833600998, 0.7297962307929993,   -1.8453190326690674, -0.025019939988851547,
        -1.1427778005599976, 0.03758539631962776,  2.6962764263153076,   1.235763669013977,   0.5428298115730286,
        -1.6115024089813232, -0.47944778203964233, -0.14335104823112488, -0.3172946274280548, -0.06614404916763306,
        1.5735383033752441,  0.7814299464225769,   -1.0786579847335815,  -0.720909833908081,  1.4707926511764526,
        1.3693809509277344,  2.6570231914520264,   0.9851194024085999,   0.37718191742897034, 1.101234793663025,
        0.5255303382873535,  -0.8293668627738953,  -1.4072566032409668,  1.6268466711044312,  0.1722732037305832,
        -0.3583550751209259, -1.5615617036819458,  -0.3546432852745056,  1.0810725688934326,  0.13147805631160736,
        0.2756350040435791,  0.6667810678482056,   -0.9943896532058716,  -1.189363956451416,  -1.1959494352340698};

    std::vector<T> v_data = {
        -0.5596300959587097,   0.5334718227386475,   0.40688660740852356,  0.3945865333080292,   0.1715109646320343,
        0.392026424407959,     0.5945261120796204,   0.6622740626335144,   -1.2063024044036865,  0.6074396967887878,
        -1.2536547183990479,   0.9868361353874207,   -0.494655042886734,   -1.2830430269241333,  -0.6537995934486389,
        0.531403660774231,     1.2350903749465942,   -1.1070388555526733,  -1.7173612117767334,  1.534561038017273,
        0.876044750213623,     -0.28708741068840027, 1.0216400623321533,   -0.07439491897821426, -1.0922236442565918,
        -0.547156810760498,    1.1710889339447021,   0.0974961370229721,   0.9633745551109314,   0.8403232097625732,
        1.7198240756988525,    -0.9609553813934326,  -0.6375024914741516,  0.07472498714923859,  0.5599694848060608,
        -0.003206211142241955, -1.6034189462661743,  0.058098483830690384, -0.6302464604377747,  0.7466416358947754};

    std::vector<T> out_data_t_no_sink_scale_default = {
        -0.5596300959587097,  0.5334718227386475,   0.40688660740852356,  0.3945865333080292,    0.1715109646320343,
        -0.4044354557991028,  0.5434284210205078,   0.4485347867012024,   0.1335161179304123,    0.24260151386260986,
        -0.04295007884502411, 0.6755574941635132,   0.38903844356536865,  -1.0970888137817383,   0.29717087745666504,
        -0.9095513820648193,  0.9409497976303101,   -0.3887394964694977,  -1.161990761756897,    -0.28604379296302795,
        -0.5596300959587097,  0.5334718227386475,   0.40688660740852356,  0.3945865333080292,    0.1715109646320343,
        0.11010648310184479,  0.5764393210411072,   0.5866177678108215,   -0.7320530414581299,   0.478299617767334,
        -0.3641112446784973,  0.5661339163780212,   0.4304530918598175,   -0.04928196594119072,  0.24057039618492126,
        0.3524087071418762,   1.1537384986877441,   -0.918759286403656,   -1.5207775831222534,   1.2959774732589722,
        -0.5596300959587097,  0.5334718227386475,   0.40688660740852356,  0.3945865333080292,    0.1715109646320343,
        -0.06002488732337952, 0.5655243992805481,   0.5409611463546753,   -0.44585585594177246,  0.4003669321537018,
        -0.9681406021118164,  0.8334401845932007,   -0.17134259641170502, -0.7821053862571716,   -0.3480212390422821,
        -0.8686208128929138,  0.8932577967643738,   -0.2254813015460968,  -1.2397266626358032,   -0.35004693269729614,
        0.876044750213623,    -0.28708741068840027, 1.0216400623321533,   -0.07439491897821426,  -1.0922236442565918,
        -0.4266248047351837,  1.0475949048995972,   0.1757626086473465,   0.8754850625991821,    0.6766543388366699,
        1.1049975156784058,   -0.40238574147224426, -0.21055714786052704, 0.23690593242645264,   0.3474586308002472,
        0.1369362473487854,   -0.13229122757911682, -0.01725679822266102, 0.27280086278915405,   0.662578821182251,
        0.876044750213623,    -0.28708741068840027, 1.0216400623321533,   -0.07439491897821426,  -1.0922236442565918,
        0.4569539725780487,   0.14230245351791382,  0.7495070099830627,   0.2311975359916687,    -0.5231457352638245,
        0.5743083953857422,   0.05502946674823761,  0.4440821409225464,   0.2777979075908661,    -0.1986960619688034,
        0.8122739791870117,   -0.27402305603027344, 0.1377803087234497,   0.1898355633020401,    0.07055333256721497,
        0.876044750213623,    -0.28708741068840027, 1.0216400623321533,   -0.07439491897821426,  -1.0922236442565918,
        0.3447447121143341,   0.2572692036628723,   0.6766449213027954,   0.31301823258399963,   -0.37077826261520386,
        0.8486983776092529,   -0.24997326731681824, 0.8985856771469116,   -0.017985841259360313, -0.9216011762619019,
        0.5427770614624023,   -0.7654033899307251,  0.11973342299461365,  -0.11444665491580963,  0.2868369519710541};

    params.push_back(PrepareTestCaseParams<T, char>(q_shape,
                                                    k_shape,
                                                    v_shape,
                                                    Shape{1},
                                                    output_shape,
                                                    true,
                                                    q_data,
                                                    k_data,
                                                    v_data,
                                                    {0},
                                                    out_data_t_no_sink_scale_default,
                                                    "with_no_sink"));

    PartialShape sink_shape{2, 3, 4, 1};
    std::vector<T> sink_data = {1.989909291267395,   1.989909291267395,   1.989909291267395,   1.989909291267395,
                                0.7182523012161255,  0.7182523012161255,  0.7182523012161255,  0.7182523012161255,
                                -0.4966665208339691, -0.4966665208339691, -0.4966665208339691, -0.4966665208339691,
                                1.3100289106369019,  1.3100289106369019,  1.3100289106369019,  1.3100289106369019,
                                0.39555686712265015, 0.39555686712265015, 0.39555686712265015, 0.39555686712265015,
                                0.08947940915822983, 0.08947940915822983, 0.08947940915822983, 0.08947940915822983};

    std::vector<T> out_data_t_with_sink_scale_one = {
        -0.03965197131037712,   0.03779855743050575,    0.028829501941800117, 0.027957992628216743,
        0.012152220122516155,   -0.11443307250738144,   0.11428776383399963,  0.08829117566347122,
        0.07568749785423279,    0.03898078203201294,    0.26677078008651733,  0.605960488319397,
        0.5677525401115417,     -1.1716898679733276,    0.506074845790863,    -1.0069886445999146,
        0.8096990585327148,     -0.4007461667060852,    -1.0468798875808716,  -0.5102735757827759,
        -0.06627094000577927,   0.06317330151796341,    0.04818318039178848,  0.046726614236831665,
        0.020310189574956894,   0.16821038722991943,    0.36302635073661804,  0.3897629678249359,
        -0.6212533712387085,    0.34174299240112305,    -0.48022228479385376, 0.5281412601470947,
        0.4162833094596863,     0.26764488220214844,    0.19879533350467682,  0.5100196599960327,
        1.2007501125335693,     -1.071186900138855,     -1.6644197702407837,  1.4861838817596436,
        -0.11463763564825058,   0.10927923023700714,    0.0833488404750824,   0.08082922548055649,
        0.035133227705955505,   -0.021249890327453613,  0.3913222551345825,   0.37849581241607666,
        -0.3413900136947632,    0.285353422164917,      -0.9711475372314453,  0.7777552008628845,
        -0.3053199350833893,    -0.8790218234062195,    -0.4489268660545349,  -1.1495906114578247,
        0.9593806266784668,     -0.4228992164134979,    -1.2732564210891724,  -0.5751422047615051,
        0.002577104140073061,   -0.0008445392595604062, 0.003005409147590399, -0.00021885121532250196,
        -0.0032130484469234943, -0.4957696497440338,    1.068182110786438,    0.09358927607536316,
        0.8794413208961487,     0.7625315189361572,     0.8379843235015869,   -0.426442950963974,
        -0.282054603099823,     0.0756443440914154,     0.28236064314842224,  -0.15671619772911072,
        0.26120778918266296,    0.013841899111866951,   0.43579405546188354,  0.6869266033172607,
        0.0653177946805954,     -0.021405203267931938,  0.07617336511611938,  -0.0055468762293457985,
        -0.08143606781959534,   0.6066479086875916,     -0.09200847148895264, 0.7866788506507874,
        0.047215692698955536,   -0.739319384098053,     0.5055353045463562,   0.012736570090055466,
        0.597989022731781,      0.1530308723449707,     -0.45584386587142944, 0.8580659031867981,
        -0.31098097562789917,   0.0922587513923645,     0.1278807520866394,   -0.01665644720196724,
        0.6572203636169434,     -0.21537677943706512,   0.766447901725769,    -0.0558120533823967,
        -0.8194006681442261,    0.5205270051956177,     0.05960403382778168,  0.7777613997459412,
        0.16860418021678925,    -0.6122713088989258,    0.8689383268356323,   -0.2839549481868744,
        1.01072096824646,       -0.0722401961684227,    -1.0794318914413452,  0.31849390268325806,
        -0.7553979158401489,    0.06754793971776962,    -0.21245919167995453, 0.2665559947490692};

    params.push_back(PrepareTestCaseParams<T, char>(q_shape,
                                                    k_shape,
                                                    v_shape,
                                                    Shape{},
                                                    output_shape,
                                                    true,
                                                    q_data,
                                                    k_data,
                                                    v_data,
                                                    {0},
                                                    out_data_t_with_sink_scale_one,
                                                    "with_sink",
                                                    sink_shape,
                                                    sink_data));
    return params;
}

std::vector<SDPAParams> generateCombinedParamsWithSink() {
    const std::vector<std::vector<SDPAParams>> generatedParams{generateParamsWithSink<element::Type_t::f32>(),
                                                               generateParamsWithSink<element::Type_t::f16>(),
                                                               generateParamsWithSink<element::Type_t::f64>()};
    std::vector<SDPAParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_SDPA_With_Sink,
                         ReferenceSDPATest,
                         testing::ValuesIn(generateCombinedParamsWithSink()),
                         ReferenceSDPATest::getTestCaseName);
}  // namespace
