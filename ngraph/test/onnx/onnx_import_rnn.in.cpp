//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lstm_fwd_with_clip)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lstm_fwd_with_clip.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({-0.455351, -0.276391, -0.185934, -0.269585}); // X
    test_case.add_input<float>({-0.494659f,                                   // W
                                0.0453352f,
                                -0.487793f,
                                0.417264f,
                                -0.0175329f,
                                0.489074f,
                                -0.446013f,
                                0.414029f,
                                -0.0091708f,
                                -0.255364f,
                                -0.106952f,
                                -0.266717f,
                                -0.0888852f,
                                -0.428709f,
                                -0.283349f,
                                0.208792f}); // W
    test_case.add_input<float>({0.146626f,
                                -0.0620289f,
                                -0.0815302f,
                                0.100482f,
                                -0.219535f,
                                -0.306635f,
                                -0.28515f,
                                -0.314112f,
                                -0.228172f,
                                0.405972f,
                                0.31576f,
                                0.281487f,
                                -0.394864f,
                                0.42111f,
                                -0.386624f,
                                -0.390225f}); // R

    test_case.add_input<float>({0.381619f,
                                0.0323954f,
                                -0.14449f,
                                0.420804f,
                                -0.258721f,
                                0.45056f,
                                -0.250755f,
                                0.0967895f,
                                0.0f,
                                0.0f,
                                0.0f,
                                0.0f,
                                0.0f,
                                0.0f,
                                0.0f,
                                0.0f});                                                 // B
    test_case.add_input<float>({0.2345f, 0.5235f, 0.4378f, 0.3475f, 0.8927f, 0.3456f}); // P

    test_case.add_expected_output<float>(
        Shape{2, 1, 1, 2}, {-0.02280854f, 0.02744377f, -0.03516197f, 0.03875681f});    // Y_data
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {-0.03516197f, 0.03875681f}); // Y_h_data
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {-0.07415761f, 0.07395997f}); // Y_c_data

    // We have to enlarge tolerance bits to 3 - it's only one bit more than default value.
    // The discrepancies may occur at most on 7th decimal position.
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lstm_fwd_mixed_seq)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lstm_fwd_mixed_seq.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    int hidden_size{3};
    test_case.add_input<float>({1.f, 2.f, 10.f, 11.f}); // X
    test_case.add_input<float>(
        {0.1f, 0.2f, 0.3f, 0.4f, 1.f, 2.f, 3.f, 4.f, 10.f, 11.f, 12.f, 13.f});    // W
    test_case.add_input(std::vector<float>(4 * hidden_size * hidden_size, 0.1f)); // R
    test_case.add_input(std::vector<float>(8 * hidden_size, 0.0f));               // B
    test_case.add_input<int>({1, 2});                                             // seq_lengths

    test_case.add_expected_output<float>(Shape{2, 1, 2, 3},
                                         {0.28828835f,
                                          0.36581863f,
                                          0.45679406f,
                                          0.34526032f,
                                          0.47220859f,
                                          0.55850911f,
                                          0.f,
                                          0.f,
                                          0.f,
                                          0.85882828f,
                                          0.90703777f,
                                          0.92382453f}); // Y_data
    test_case.add_expected_output<float>(
        Shape{1, 2, 3},
        {0.28828835f, 0.36581863f, 0.45679406f, 0.85882828f, 0.90703777f, 0.92382453f}); // Y_h_data
    test_case.add_expected_output<float>(
        Shape{1, 2, 3},
        {0.52497941f, 0.54983425f, 0.5744428f, 1.3249796f, 1.51063104f, 1.61451544f}); // Y_c_data

    // We have to enlarge tolerance bits to 3 - it's only one bit more than default value.
    // The discrepancies may occur at most on 7th decimal position.
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lstm_fwd_hardsigmoid_activation)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lstm_fwd_hardsigmoid_activation.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    // X
    test_case.add_input<float>({-0.455351f, -0.276391f, -0.185934f, -0.269585f});
    // W
    test_case.add_input<float>({-0.494659f,
                                0.0453352f,
                                -0.487793f,
                                0.417264f,
                                -0.0175329f,
                                0.489074f,
                                -0.446013f,
                                0.414029f,
                                -0.0091708f,
                                -0.255364f,
                                -0.106952f,
                                -0.266717f,
                                -0.0888852f,
                                -0.428709f,
                                -0.283349f,
                                0.208792f});
    // R
    test_case.add_input<float>({0.146626f,
                                -0.0620289f,
                                -0.0815302f,
                                0.100482f,
                                -0.219535f,
                                -0.306635f,
                                -0.28515f,
                                -0.314112f,
                                -0.228172f,
                                0.405972f,
                                0.31576f,
                                0.281487f,
                                -0.394864f,
                                0.42111f,
                                -0.386624f,
                                -0.390225f});

    // Y
    test_case.add_expected_output<float>(Shape{2, 1, 1, 2},
                                         {0.09086666f, 0.04378549f, 0.12914555f, 0.00257774f});
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {0.12914555f, 0.00257774f});
    // Y_c
    test_case.add_expected_output<float>(Shape{1, 1, 2}, {0.19017234f, 0.00356848f});

    // The discrepancies occur at most at 18th mantissa bit - 8th decimal position.
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lstm_fwd_large_batch_no_clip)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lstm_fwd_large_batch_no_clip.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    std::size_t seq_length = 2;
    std::size_t batch_size = 32;
    std::size_t input_size = 1;
    std::size_t hidden_size = 3;

    std::vector<float> in_X(seq_length * batch_size * input_size);
    std::iota(std::begin(in_X), std::end(in_X), 1.f);
    std::vector<float> in_R(4 * hidden_size * hidden_size, 0.1f);

    // X
    test_case.add_input<float>(in_X);
    // W
    test_case.add_input<float>(
        {0.1f, 0.2f, 0.3f, 0.4f, 1.f, 2.f, 3.f, 4.f, 10.f, 11.f, 12.f, 13.f});
    // R
    test_case.add_input<float>(in_R);

    // Y_h_data
    test_case.add_expected_output<float>(
        Shape{1, batch_size, hidden_size},
        {0.90387899f, 0.9135572f,  0.91772245f, 0.90897038f, 0.92132433f, 0.92825467f, 0.91365823f,
         0.92815113f, 0.93676105f, 0.91799162f, 0.93406357f, 0.94344562f, 0.92199681f, 0.93912057f,
         0.94859476f, 0.92569357f, 0.94340185f, 0.95250664f, 0.92909964f, 0.94699686f, 0.95545127f,
         0.93223207f, 0.94999634f, 0.95765468f, 0.93510761f, 0.9524867f,  0.95929726f, 0.93774272f,
         0.9545467f,  0.96051891f, 0.9401536f,  0.95624603f, 0.96142619f, 0.94235605f, 0.95764499f,
         0.96209939f, 0.94436539f, 0.95879495f, 0.96259862f, 0.94619635f, 0.95973921f, 0.96296872f,
         0.94786299f, 0.96051397f, 0.96324302f, 0.94937864f, 0.96114929f, 0.96344629f, 0.95075587f,
         0.96167006f, 0.96359692f, 0.95200645f, 0.96209679f, 0.96370852f, 0.95314133f, 0.9624464f,
         0.9637912f,  0.95417069f, 0.96273278f, 0.96385246f, 0.95510395f, 0.96296733f, 0.96389785f,
         0.95594975f, 0.96315942f, 0.96393147f, 0.95671607f, 0.96331673f, 0.96395638f, 0.9574102f,
         0.96344554f, 0.96397483f, 0.9580388f,  0.96355102f, 0.9639885f,  0.95860795f, 0.96363739f,
         0.96399863f, 0.95912322f, 0.96370811f, 0.96400613f, 0.95958963f, 0.96376601f, 0.96401169f,
         0.96001179f, 0.96381342f, 0.96401581f, 0.96039386f, 0.96385224f, 0.96401886f, 0.96073964f,
         0.96388402f, 0.96402112f, 0.96105254f, 0.96391004f, 0.96402279f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lstm_bdir_short_input_seq)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lstm_bdir_short_input_seq.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    // X
    test_case.add_input<float>({-0.455351f, -0.276391f, -0.185934f, -0.269585f});
    // W
    test_case.add_input<float>(
        {-0.494659f,  0.0453352f,  -0.487793f, 0.417264f,   -0.0175329f, 0.489074f,   -0.446013f,
         0.414029f,   -0.0091708f, -0.255364f, -0.106952f,  -0.266717f,  -0.0888852f, -0.428709f,
         -0.283349f,  0.208792f,   -0.494659f, 0.0453352f,  -0.487793f,  0.417264f,   -0.0175329f,
         0.489074f,   -0.446013f,  0.414029f,  -0.0091708f, -0.255364f,  -0.106952f,  -0.266717f,
         -0.0888852f, -0.428709f,  -0.283349f, 0.208792f});
    // R
    test_case.add_input<float>(
        {0.146626f,  -0.0620289f, -0.0815302f, 0.100482f,   -0.219535f,  -0.306635f, -0.28515f,
         -0.314112f, -0.228172f,  0.405972f,   0.31576f,    0.281487f,   -0.394864f, 0.42111f,
         -0.386624f, -0.390225f,  0.146626f,   -0.0620289f, -0.0815302f, 0.100482f,  -0.219535f,
         -0.306635f, -0.28515f,   -0.314112f,  -0.228172f,  0.405972f,   0.31576f,   0.281487f,
         -0.394864f, 0.42111f,    -0.386624f,  -0.390225f});
    // B
    test_case.add_input<float>(
        {0.381619f, 0.0323954f, -0.14449f, 0.420804f, -0.258721f, 0.45056f, -0.250755f, 0.0967895f,
         0.0f,      0.0f,       0.0f,      0.0f,      0.0f,       0.0f,     0.0f,       0.0f,
         0.381619f, 0.0323954f, -0.14449f, 0.420804f, -0.258721f, 0.45056f, -0.250755f, 0.0967895f,
         0.0f,      0.0f,       0.0f,      0.0f,      0.0f,       0.0f,     0.0f,       0.0f});
    // sequence_lens
    test_case.add_input<int>({1});
    // initial_h
    test_case.add_input<float>({0.0f, 0.0f, -0.0306872f, 0.028035f});
    // initial_c
    test_case.add_input<float>({0.0f, 0.0f, -0.07243599f, 0.0467052f});
    // P
    test_case.add_input<float>({0.2345f,
                                0.5235f,
                                0.4378f,
                                0.3475f,
                                0.8927f,
                                0.3456f,
                                0.2345f,
                                0.5235f,
                                0.4378f,
                                0.3475f,
                                0.8927f,
                                0.3456f});

    // Y
    test_case.add_expected_output<float>(
        Shape{2, 2, 1, 2},
        {-0.0251062f, 0.0561262f, -0.0318928f, 0.0762679f, 0.0f, 0.0f, 0.0f, 0.0f});
    // Y_h
    test_case.add_expected_output<float>(Shape{2, 1, 2},
                                         {-0.0251062f, 0.0561262f, -0.0318928f, 0.0762679f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 3);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_lstm_mixed_seq_reverse)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/lstm_mixed_seq_reverse.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    size_t hidden_size = 3;

    // X
    test_case.add_input<float>({1.f, 2.f, 10.f, 11.f});
    // W
    test_case.add_input<float>(
        {0.1f, 0.2f, 0.3f, 0.4f, 1.f, 2.f, 3.f, 4.f, 10.f, 11.f, 12.f, 13.f});
    // R
    test_case.add_input<float>(std::vector<float>(4 * hidden_size * hidden_size, 0.1f));
    // sequence_lens
    test_case.add_input<int>({1, 2});

    // Y
    test_case.add_expected_output<float>(Shape{2, 1, 2, 3},
                                         {0.28828844f,
                                          0.36581877f,
                                          0.45679423f,
                                          0.64046413f,
                                          0.82303363f,
                                          0.91610711f,
                                          0.f,
                                          0.f,
                                          0.f,
                                          0.62759886f,
                                          0.71640738f,
                                          0.74624585f});
    // Y_h
    test_case.add_expected_output<float>(
        Shape{1, 2, 3},
        {0.28828844f, 0.36581877f, 0.45679423f, 0.64046413f, 0.82303363f, 0.91610711f});
    // Y_c
    test_case.add_expected_output<float>(
        Shape{1, 2, 3},
        {0.52497941f, 0.54983425f, 0.5744428f, 1.34960834f, 1.54772296f, 1.65633056f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

// RNNLikeSequenceOp test fixture for test setup reuse
class RNNLikeSequenceOp : public testing::Test
{
public:
    std::vector<float> in_X{0.68172926f,  1.1405563f,   -0.03931177f, -0.03759607f, 0.22778925f,
                            1.2471468f,   0.2785642f,   0.5198979f,   0.3712886f,   -0.3194908f,
                            0.8448233f,   -0.62065625f, 1.2968333f,   -0.20370148f, 0.40204826f,
                            -0.23721986f, 0.3629822f,   -0.3819832f,  -0.7766345f,  0.19374144f,
                            1.1397027f,   0.60444903f,  1.3246384f,   -0.28191715f};

    std::vector<float> in_W{
        0.31403765f,  -0.16793324f, 1.388258f,    -0.6902954f,  -0.3994045f,  -0.7833511f,
        -0.30992958f, 0.3557573f,   -0.4682631f,  1.1741459f,   -2.414789f,   -0.42783254f,
        -0.82199496f, -0.03900861f, -0.43670088f, -0.53810567f, -0.10769883f, 0.75242394f,
        -0.2507971f,  1.0447186f,   -1.4777364f,  0.19993274f,  0.925649f,    -2.282516f,
        0.95039636f,  1.5379831f,   -0.88576007f, 0.28566247f,  0.79292643f,  -0.04261953f,
    };

    std::vector<float> in_R{
        0.8490583f,   0.45121244f,  -1.1799014f,  0.13536449f,  0.81328654f,  0.6017516f,
        0.48475724f,  -1.2136037f,  0.16383322f,  1.5106261f,   1.1177503f,   0.23582461f,
        0.5754652f,   0.43879887f,  0.7399294f,   0.4517558f,   1.3536783f,   -0.4843166f,
        -1.1503736f,  -0.2458678f,  0.54523313f,  -0.08649993f, -0.6936281f,  1.002422f,
        -1.770847f,   -0.94642f,    -1.8135757f,  1.8819852f,   -0.10852333f, -0.26120332f,
        1.0223165f,   -0.7468837f,  0.28566906f,  0.92321056f,  0.22521864f,  1.1123824f,
        -0.9298287f,  1.2141289f,   1.3470556f,   -0.32972014f, -1.6552197f,  -1.0998285f,
        0.71901864f,  0.962846f,    -0.1366851f,  -2.6534476f,  -1.4992771f,  -0.45793465f,
        0.4290477f,   0.9893151f,   0.2511034f,   0.12906462f,  0.7491512f,   0.3316756f,
        1.0576645f,   -0.04618666f, 1.3556088f,   1.2842374f,   0.7103014f,   0.52889013f,
        0.30327162f,  1.5069056f,   0.16591893f,  1.5719851f,   -2.099427f,   -1.010277f,
        -0.52800924f, -0.22292352f, -0.55177474f, 1.3432894f,   0.8731192f,   -0.01055307f,
        -0.01138215f, 0.85698843f,  -1.2615703f,
    };

    std::vector<float> in_B{
        0.5336702f,   1.6593654f,   -1.150011f,  0.00342217f,  0.799371f,    0.43780383f,
        -0.55082625f, 1.0774187f,   -0.6065135f, 0.6434064f,   -1.5693754f,  1.4923384f,
        1.1554348f,   -1.328159f,   0.24995533f, 0.15112682f,  -0.34698758f, -0.10088819f,
        -0.2931625f,  -0.47319615f, 0.66167855f, -1.1646721f,  -0.09588219f, 0.5212928f,
        0.37182367f,  0.27342287f,  1.1613405f,  -0.75196224f, -1.5143642f,  0.20604452f,
    };

    std::vector<int32_t> in_sequence_lens{2, 3, 4};

    std::vector<float> in_initial_h{
        -0.4840693f,
        -1.4054376f,
        0.84533644f,
        -0.1160888f,
        -1.3724717f,
        1.978259f,
        -0.8500094f,
        -2.0120409f,
        0.89959633f,
        -0.5367942f,
        0.21188478f,
        1.7603784f,
        0.38752958f,
        -0.06706902f,
        -1.4290836f,
    };

protected:
    virtual void SetUp() override {}
};

NGRAPH_TEST_F(${BACKEND_NAME}, RNNLikeSequenceOp, onnx_model_gru_defaults_fwd)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gru_defaults_fwd.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.3224981f,  -0.44282594f, 0.7499796f,   -0.12240417f, 0.12079421f,  0.02534253f,
            0.02504562f,  -0.0463777f,  0.01204534f,  -0.01497037f, -0.04651929f, -0.6264307f,
            0.7236632f,   0.06250653f,  0.02594197f,  -0.06868916f, -0.5412897f,  0.49794048f,
            0.22239858f,  -0.11257736f, -0.23071964f, 0.26079988f,  -0.07375772f, -0.21816255f,
            0.18764113f,  -0.5228772f,  0.00575754f,  0.2514028f,   -0.58864325f, 0.49843538f,
            -0.6129046f,  -0.10794663f, 0.6544055f,   -0.70105773f, 0.5397687f,   -0.35791716f,
            0.3885092f,   -0.15291792f, -0.22324723f, 0.11557932f,  -0.42112932f, 0.26772985f,
            -0.38304564f, -0.05039781f, -0.5057976f,  0.5775348f,   -0.6736855f,  -0.20032284f,
            0.03698462f,  -0.7693824f,  -0.5831348f,  0.25767964f,  0.7121098f,   -0.35951245f,
            0.39223647f,  -0.6645166f,  0.37950075f,  0.59931314f,  -0.4741001f,  0.21156166f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             0.5775348f,
                                             -0.6736855f,
                                             -0.20032284f,
                                             0.03698462f,
                                             -0.7693824f,
                                             -0.5831348f,
                                             0.25767964f,
                                             0.7121098f,
                                             -0.35951245f,
                                             0.39223647f,
                                             -0.6645166f,
                                             0.37950075f,
                                             0.59931314f,
                                             -0.4741001f,
                                             0.21156166f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 6);
}

NGRAPH_TEST_F(${BACKEND_NAME}, RNNLikeSequenceOp, onnx_model_gru_fwd_activations)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gru_fwd_activations.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            0.33636323f,  0.08874974f, 0.9804139f,   0.35797057f, -0.01193848f, 0.51011515f,
            0.50988495f,  0.4592467f,  0.5048162f,   0.4940862f,  0.4825466f,   0.f,
            0.9269162f,   0.3328298f,  -0.18795171f, 0.69541144f, 0.7612694f,   0.937299f,
            0.3463983f,   0.38764104f, 0.49957055f,  0.27359068f, 0.38423678f,  0.3618936f,
            0.55977404f,  0.5223568f,  0.46266305f,  1.016379f,   0.22654215f,  0.6347567f,
            0.53541327f,  0.46684968f, 1.0639775f,   0.21325049f, 0.70507824f,  0.48425108f,
            -0.05370265f, 0.3055008f,  0.38166368f,  0.5645658f,  0.5998517f,   0.42573926f,
            1.4539189f,   0.31789488f, 0.5567502f,   1.f,         0.92153484f,  1.4015231f,
            0.24147032f,  0.5783859f,  0.42785907f,  -0.5690068f, 0.69624555f,  0.32291538f,
            0.68179333f,  0.50179297f, 0.0067991f,   2.043301f,   0.12669492f,  0.7062868f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             1.f,
                                             0.92153484f,
                                             1.4015231f,
                                             0.24147032f,
                                             0.5783859f,
                                             0.42785907f,
                                             -0.5690068f,
                                             0.69624555f,
                                             0.32291538f,
                                             0.68179333f,
                                             0.50179297f,
                                             0.0067991f,
                                             2.043301f,
                                             0.12669492f,
                                             0.7062868f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 5);
}

NGRAPH_TEST_F(${BACKEND_NAME}, RNNLikeSequenceOp, onnx_model_gru_fwd_mixed_seq_len)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gru_fwd_mixed_seq_len.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);
    test_case.add_input<float>(in_B);
    test_case.add_input<int>(in_sequence_lens);
    test_case.add_input<float>(in_initial_h);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.9559332f,  0.4372494f,   0.9967716f,   -0.9079381f,  -1.2538278f,  1.9265908f,
            -0.8437393f,  -1.2057271f,  -0.25887525f, -0.52679026f, -0.3619178f,  0.67928517f,
            0.9486744f,   -0.12006134f, -1.3862017f,  -0.98941356f, 0.80389524f,  0.97586197f,
            -0.9343586f,  -0.74858856f, 1.797039f,    -0.7873732f,  -0.72469383f, -0.5866635f,
            -0.42103744f, -0.8406298f,  0.85877097f,  0.6349921f,   -0.55897295f, -0.6168443f,
            0.f,          0.f,          0.f,          0.f,          0.f,          1.577129f,
            -0.6935871f,  -0.304804f,   -0.75392795f, -0.20703818f, -0.93796504f, 0.9220495f,
            0.36017662f,  -0.7007159f,  0.06962098f,  0.f,          0.f,          0.f,
            0.f,          0.f,          0.f,          0.f,          0.f,          0.f,
            0.f,          -0.96323603f, 0.9265786f,   0.54976916f,  -0.8037839f,  0.73501444f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.98941356f,
                                             0.80389524f,
                                             0.97586197f,
                                             -0.9343586f,
                                             -0.74858856f,
                                             1.577129f,
                                             -0.6935871f,
                                             -0.304804f,
                                             -0.75392795f,
                                             -0.20703818f,
                                             -0.96323603f,
                                             0.9265786f,
                                             0.54976916f,
                                             -0.8037839f,
                                             0.73501444f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 3);
}

NGRAPH_TEST_F(${BACKEND_NAME}, RNNLikeSequenceOp, onnx_model_gru_rev_clip)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gru_rev_clip.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.50679326f, -0.8251296f,  0.7804218f,   -0.1813852f,  0.00147036f,  -0.18647355f,
            0.38888037f,  -0.07898733f, -0.05150563f, -0.23335457f, -0.21705005f, -0.2966391f,
            0.67461425f,  -0.1695634f,  -0.09241624f, -0.10538863f, -0.6444952f,  -0.01815936f,
            -0.09695458f, -0.15107796f, -0.5036379f,  0.56125206f,  0.12785181f,  -0.22290717f,
            0.08662428f,  -0.5849108f,  0.4789885f,   -0.03569929f, -0.42043984f, 0.33464667f,
            -0.01091215f, -0.42090097f, 0.24428985f,  -0.6002675f,  0.27305228f,  -0.35063627f,
            0.3717615f,   -0.00495788f, -0.00491725f, -0.27061304f, -0.3190831f,  0.3542383f,
            -0.17784928f, -0.12995736f, -0.30778408f, 0.47168806f,  -0.6330014f,  -0.1905269f,
            0.26708886f,  -0.19741398f, -0.3995853f,  -0.07459997f, 0.6749513f,   -0.36566192f,
            0.32173023f,  -0.36364347f, 0.13916425f,  0.3908174f,   -0.53085154f, 0.56740737f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.50679326f,
                                             -0.8251296f,
                                             0.7804218f,
                                             -0.1813852f,
                                             0.00147036f,
                                             -0.18647355f,
                                             0.38888037f,
                                             -0.07898733f,
                                             -0.05150563f,
                                             -0.23335457f,
                                             -0.21705005f,
                                             -0.2966391f,
                                             0.67461425f,
                                             -0.1695634f,
                                             -0.09241624f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 8);
}

NGRAPH_TEST_F(${BACKEND_NAME}, RNNLikeSequenceOp, onnx_model_gru_reverse)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gru_reverse.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.51097775f, -0.85767376f, 0.8065842f,   -0.1832461f,  -0.00109532f, -0.18766233f,
            0.3910985f,   -0.0617601f,  -0.05733761f, -0.23259571f, -0.22787738f, -0.3715533f,
            0.70320934f,  -0.17635077f, -0.0972611f,  -0.11218601f, -0.660165f,   -0.03494868f,
            -0.07503931f, -0.15422714f, -0.5053969f,  0.5710621f,   0.1448728f,   -0.225453f,
            0.07250313f,  -0.5988957f,  0.48768237f,  0.00665835f,  -0.42196327f, 0.2749501f,
            -0.02106231f, -0.44533628f, 0.24044508f,  -0.5907899f,  0.26883256f,  -0.3462156f,
            0.3782666f,   0.00699124f,  -0.00378288f, -0.2990779f,  -0.32031405f, 0.3363319f,
            -0.1877775f,  -0.10781199f, -0.40970552f, 0.47168806f,  -0.6330014f,  -0.1905269f,
            0.26708886f,  -0.19741398f, -0.3995853f,  -0.07459997f, 0.691666f,    -0.36566192f,
            0.32173023f,  -0.37267625f, 0.1103513f,   0.3908174f,   -0.53085154f, 0.56740737f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.51097775f,
                                             -0.85767376f,
                                             0.8065842f,
                                             -0.1832461f,
                                             -0.00109532f,
                                             -0.18766233f,
                                             0.3910985f,
                                             -0.0617601f,
                                             -0.05733761f,
                                             -0.23259571f,
                                             -0.22787738f,
                                             -0.3715533f,
                                             0.70320934f,
                                             -0.17635077f,
                                             -0.0972611f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 7);
}

NGRAPH_TEST_F(${BACKEND_NAME}, RNNLikeSequenceOp, onnx_model_gru_fwd_bias_initial_h)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/gru_fwd_bias_initial_h.prototxt"));

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    test_case.add_input<float>(in_X);
    test_case.add_input<float>(in_W);
    test_case.add_input<float>(in_R);
    test_case.add_input<float>(in_B);
    test_case.add_input<float>(in_initial_h);

    // Y
    test_case.add_expected_output<float>(
        Shape{4, 1, 3, 5},
        std::vector<float>{
            -0.9559332f,  0.4372494f,   0.9967716f,   -0.9079381f,  -1.2538278f,  1.9265908f,
            -0.8437393f,  -1.2057271f,  -0.25887525f, -0.52679026f, -0.3619178f,  0.67928517f,
            0.9486744f,   -0.12006134f, -1.3862017f,  -0.98941356f, 0.80389524f,  0.97586197f,
            -0.9343586f,  -0.74858856f, 1.797039f,    -0.7873732f,  -0.72469383f, -0.5866635f,
            -0.42103744f, -0.8406298f,  0.85877097f,  0.6349921f,   -0.55897295f, -0.6168443f,
            -0.99686503f, 0.87408733f,  0.87070423f,  -0.9564345f,  0.52932394f,  1.577129f,
            -0.6935871f,  -0.304804f,   -0.75392795f, -0.20703818f, -0.93796504f, 0.9220495f,
            0.36017662f,  -0.7007159f,  0.06962098f,  -0.22581682f, 0.9119905f,   -0.64628327f,
            -0.79374063f, -0.82321495f, 1.2853851f,   -0.6176347f,  0.6865668f,   -0.85147655f,
            0.0379298f,   -0.96323603f, 0.9265786f,   0.54976916f,  -0.8037839f,  0.73501444f,
        });
    // Y_h
    test_case.add_expected_output<float>(Shape{1, 3, 5},
                                         std::vector<float>{
                                             -0.22581682f,
                                             0.9119905f,
                                             -0.64628327f,
                                             -0.79374063f,
                                             -0.82321495f,
                                             1.2853851f,
                                             -0.6176347f,
                                             0.6865668f,
                                             -0.85147655f,
                                             0.0379298f,
                                             -0.96323603f,
                                             0.9265786f,
                                             0.54976916f,
                                             -0.8037839f,
                                             0.73501444f,
                                         });
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 4);
}
