// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on
#include <numeric>
#include "gtest/gtest.h"
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

using Attrs = op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes;
using ExperimentalGP = op::v6::ExperimentalDetectronGenerateProposalsSingleImage;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, experimental_detectron_generate_proposals_eval)
{
    Attrs attrs;
    attrs.min_size = 0;
    attrs.nms_threshold = 0.699999988079071;
    attrs.post_nms_count = 6;
    attrs.pre_nms_count = 1000;

    auto im_info = std::make_shared<op::Parameter>(element::f32, Shape{3});
    auto anchors = std::make_shared<op::Parameter>(element::f32, Shape{36, 4});
    auto deltas = std::make_shared<op::Parameter>(element::f32, Shape{12, 2, 6});
    auto scores = std::make_shared<op::Parameter>(element::f32, Shape{3, 2, 6});

    auto proposals = std::make_shared<ExperimentalGP>(im_info, anchors, deltas, scores, attrs);

    auto f0 = make_shared<Function>(OutputVector{proposals->output(0)},
                                    ParameterVector{im_info, anchors, deltas, scores});
    auto f1 = make_shared<Function>(OutputVector{proposals->output(1)},
                                    ParameterVector{im_info, anchors, deltas, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    std::vector<float> im_info_data = {1.0f, 1.0f, 1.0f};
    std::vector<float> anchors_data = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f};
    std::vector<float> deltas_data = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f};
    std::vector<float> scores_data = {
        5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f};

    const auto output_rois_shape = Shape{6, 4};
    const auto output_scores_shape = Shape{6};

    std::vector<float> expected_output_rois = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    std::vector<float> expected_output_scores = {8.0f, 5.0f, 4.0f, 1.0f, 1.0f, 1.0f};

    auto output_rois = backend->create_tensor(element::f32, output_rois_shape);
    auto output_scores = backend->create_tensor(element::f32, output_scores_shape);

    auto backend_im_info = backend->create_tensor(element::f32, Shape{3});
    auto backend_anchors = backend->create_tensor(element::f32, Shape{36, 4});
    auto backend_deltas = backend->create_tensor(element::f32, Shape{12, 2, 6});
    auto backend_scores = backend->create_tensor(element::f32, Shape{3, 2, 6});

    copy_data(backend_im_info, im_info_data);
    copy_data(backend_anchors, anchors_data);
    copy_data(backend_deltas, deltas_data);
    copy_data(backend_scores, scores_data);

    auto handle0 = backend->compile(f0);
    auto handle1 = backend->compile(f1);

    handle0->call_with_validate({output_rois},
                                {backend_im_info, backend_anchors, backend_deltas, backend_scores});
    handle1->call_with_validate({output_scores},
                                {backend_im_info, backend_anchors, backend_deltas, backend_scores});

    EXPECT_TRUE(test::all_close_f(
        expected_output_rois, read_vector<float>(output_rois), MIN_FLOAT_TOLERANCE_BITS));
    EXPECT_TRUE(test::all_close_f(
        expected_output_scores, read_vector<float>(output_scores), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, experimental_detectron_generate_proposals_eval_2)
{
    Attrs attrs;
    attrs.min_size = 0;
    attrs.nms_threshold = 0.699999988079071;
    attrs.post_nms_count = 6;
    attrs.pre_nms_count = 1000;

    auto im_info = std::make_shared<op::Parameter>(element::f32, Shape{3});
    auto anchors = std::make_shared<op::Parameter>(element::f32, Shape{36, 4});
    auto deltas = std::make_shared<op::Parameter>(element::f32, Shape{12, 2, 6});
    auto scores = std::make_shared<op::Parameter>(element::f32, Shape{3, 2, 6});

    auto proposals = std::make_shared<ExperimentalGP>(im_info, anchors, deltas, scores, attrs);

    auto f0 = make_shared<Function>(OutputVector{proposals->output(0)},
                                    ParameterVector{im_info, anchors, deltas, scores});
    auto f1 = make_shared<Function>(OutputVector{proposals->output(1)},
                                    ParameterVector{im_info, anchors, deltas, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    std::vector<float> im_info_data = {150.0, 150.0, 1.0};
    std::vector<float> anchors_data = {
        12.0,  68.0,  102.0, 123.0, 46.0,  80.0,  79.0,  128.0, 33.0,  71.0,  127.0, 86.0,  33.0,
        56.0,  150.0, 73.0,  5.0,   41.0,  93.0,  150.0, 74.0,  66.0,  106.0, 115.0, 17.0,  37.0,
        87.0,  150.0, 31.0,  27.0,  150.0, 39.0,  29.0,  23.0,  112.0, 123.0, 41.0,  37.0,  103.0,
        150.0, 8.0,   46.0,  98.0,  111.0, 7.0,   69.0,  114.0, 150.0, 70.0,  21.0,  150.0, 125.0,
        54.0,  19.0,  132.0, 68.0,  62.0,  8.0,   150.0, 101.0, 57.0,  81.0,  150.0, 97.0,  79.0,
        29.0,  109.0, 130.0, 12.0,  63.0,  100.0, 150.0, 17.0,  33.0,  113.0, 150.0, 90.0,  78.0,
        150.0, 111.0, 47.0,  68.0,  150.0, 71.0,  66.0,  103.0, 111.0, 150.0, 4.0,   17.0,  112.0,
        94.0,  12.0,  8.0,   119.0, 98.0,  54.0,  56.0,  120.0, 150.0, 56.0,  29.0,  150.0, 31.0,
        42.0,  3.0,   139.0, 92.0,  41.0,  65.0,  150.0, 130.0, 49.0,  13.0,  143.0, 30.0,  40.0,
        60.0,  150.0, 150.0, 23.0,  73.0,  24.0,  115.0, 56.0,  84.0,  107.0, 108.0, 63.0,  8.0,
        142.0, 125.0, 78.0,  37.0,  93.0,  144.0, 40.0,  34.0,  150.0, 46.0,  30.0,  21.0,  150.0,
        120.0};
    std::vector<float> deltas_data = {
        9.062256,    10.883133,  9.8441105,     12.694285,  0.41781136, 8.749107,    14.990341,
        6.587644,    1.4206103,  13.299262,     12.432549,  2.736371,   0.22732796,  6.3361835,
        12.268727,   2.1009045,  4.771589,      2.5131326,  5.610736,   9.3604145,   4.27379,
        8.317948,    0.60510135, 6.7446275,     1.0207708,  1.1352817,  1.5785321,   1.718335,
        1.8093798,   0.99247587, 1.3233583,     1.7432803,  1.8534478,  1.2593061,   1.7394226,
        1.7686696,   1.647999,   1.7611449,     1.3119122,  0.03007332, 1.1106564,   0.55669737,
        0.2546148,   1.9181818,  0.7134989,     2.0407224,  1.7211134,  1.8565536,   14.562747,
        2.8786168,   0.5927796,  0.2064463,     7.6794515,  8.672126,   10.139171,   8.002429,
        7.002932,    12.6314945, 10.550842,     0.15784842, 0.3194304,  10.752157,   3.709805,
        11.628928,   0.7136225,  14.619964,     15.177284,  2.2824087,  15.381494,   0.16618137,
        7.507227,    11.173228,  0.4923559,     1.8227729,  1.4749299,  1.7833921,   1.2363617,
        -0.23659119, 1.5737582,  1.779316,      1.9828427,  1.0482665,  1.4900246,   1.3563544,
        1.5341306,   0.7634312,  4.6216766e-05, 1.6161222,  1.7512476,  1.9363779,   0.9195784,
        1.4906164,  -0.03244795, 0.681073,      0.6192401,  1.8033613,  14.146055,   3.4043705,
        15.292292,  3.5295358,   11.138999,     9.952057,   5.633434,   12.114562,   9.427372,
        12.384038,  9.583308,    8.427233,      15.293704,  3.288159,   11.64898,    9.350885,
        2.0037227,  13.523184,   4.4176426,     6.1057625,  14.400079,  8.248259,    11.815807,
        15.713364,  1.0023532,   1.3203261,     1.7100681,  0.7407832,  1.09448,     1.7188418,
        1.4412547,  1.4862992,   0.74790007,    0.31571656, 0.6398838,  2.0236106,   1.1869069,
        1.7265586,  1.2624544,   0.09934269,    1.3508598,  0.85212964, -0.38968498, 1.7059708,
        1.6533034,  1.7400402,   1.8123854,     -0.43063712};
    std::vector<float> scores_data = {
        0.7719922,   0.35906568, 0.29054508, 0.18124384, 0.5604661,  0.84750974, 0.98948747,
        0.009793862, 0.7184191,  0.5560748,  0.6952493,  0.6732593,  0.3306898,  0.6790913,
        0.41128764,  0.34593266, 0.94296855, 0.7348507,  0.24478768, 0.94024557, 0.05405676,
        0.06466125,  0.36244348, 0.07942984, 0.10619422, 0.09412837, 0.9053611,  0.22870538,
        0.9237487,   0.20986171, 0.5067282,  0.29709867, 0.53138554, 0.189101,   0.4786443,
        0.88421875};

    const auto output_rois_shape = Shape{6, 4};
    const auto output_scores_shape = Shape{6};

    std::vector<float> expected_output_rois = {
        149, 149,               149, 149, 149, 0,   149, 149, 149, 60.87443542480469, 149, 149,
        149, 61.89498901367188, 149, 149, 149, 149, 149, 149, 149, 149,               149, 149};

    std::vector<float> expected_output_scores = {
        0.9894874691963196, 0.9429685473442078, 0.9402455687522888,
        0.9237486720085144, 0.9053611159324646, 0.8842187523841858};

    auto output_rois = backend->create_tensor(element::f32, output_rois_shape);
    auto output_scores = backend->create_tensor(element::f32, output_scores_shape);

    auto backend_im_info = backend->create_tensor(element::f32, Shape{3});
    auto backend_anchors = backend->create_tensor(element::f32, Shape{36, 4});
    auto backend_deltas = backend->create_tensor(element::f32, Shape{12, 2, 6});
    auto backend_scores = backend->create_tensor(element::f32, Shape{3, 2, 6});

    copy_data(backend_im_info, im_info_data);
    copy_data(backend_anchors, anchors_data);
    copy_data(backend_deltas, deltas_data);
    copy_data(backend_scores, scores_data);

    auto handle0 = backend->compile(f0);
    auto handle1 = backend->compile(f1);

    handle0->call_with_validate({output_rois},
                                {backend_im_info, backend_anchors, backend_deltas, backend_scores});
    handle1->call_with_validate({output_scores},
                                {backend_im_info, backend_anchors, backend_deltas, backend_scores});

    const auto calculated_rois = read_vector<float>(output_rois);
    const auto calculated_scores = read_vector<float>(output_scores);

    EXPECT_TRUE(test::all_close_f(
        expected_output_rois, read_vector<float>(output_rois), MIN_FLOAT_TOLERANCE_BITS));

    EXPECT_TRUE(test::all_close_f(
        expected_output_scores, read_vector<float>(output_scores), MIN_FLOAT_TOLERANCE_BITS));
}
