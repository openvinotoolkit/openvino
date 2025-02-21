// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

#include "common_test_utils/all_close.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ndarray.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"
#include "openvino/op/max_pool.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_conv2d_strides_padding) {
    // Convolution with strides=2 and padding=1
    auto model = convert_model("conv_with_strides_padding.onnx");

    Inputs inputs;
    // data (1, 1, 7, 5) input tensor
    inputs.emplace_back(ov::test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f, 4.f},
                                                       {5.f, 6.f, 7.f, 8.f, 9.f},
                                                       {10.f, 11.f, 12.f, 13.f, 14.f},
                                                       {15.f, 16.f, 17.f, 18.f, 19.f},
                                                       {20.f, 21.f, 22.f, 23.f, 24.f},
                                                       {25.f, 26.f, 27.f, 28.f, 29.f},
                                                       {30.f, 31.f, 32.f, 33.f, 34.f}}}}}
                            .get_vector());

    // filters (1, 1, 3, 3) aka convolution weights
    inputs.emplace_back(
        ov::test::NDArray<float, 4>{{{{{1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}}}}}.get_vector());

    // (1, 1, 4, 3)
    auto expected_output =
        ov::test::NDArray<float, 4>(
            {{{{12.f, 27.f, 24.f}, {63.f, 108.f, 81.f}, {123.f, 198.f, 141.f}, {112.f, 177.f, 124.f}}}})
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_conv2d_strides_no_padding) {
    // Convolution with strides=2 and padding=1
    auto model = convert_model("conv_with_strides_no_padding.onnx");

    Inputs inputs;
    // data (1, 1, 7, 5) input tensor
    inputs.emplace_back(ov::test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f, 4.f},
                                                       {5.f, 6.f, 7.f, 8.f, 9.f},
                                                       {10.f, 11.f, 12.f, 13.f, 14.f},
                                                       {15.f, 16.f, 17.f, 18.f, 19.f},
                                                       {20.f, 21.f, 22.f, 23.f, 24.f},
                                                       {25.f, 26.f, 27.f, 28.f, 29.f},
                                                       {30.f, 31.f, 32.f, 33.f, 34.f}}}}}
                            .get_vector());

    // filters (1, 1, 3, 3) aka convolution weights
    inputs.emplace_back(
        ov::test::NDArray<float, 4>{{{{{1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}}}}}.get_vector());

    // (1, 1, 3, 2)
    auto expected_output = ov::test::NDArray<float, 4>({{{{54.f, 72.f}, {144.f, 162.f}, {234.f, 252.f}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_conv2d_strides_assymetric_padding) {
    // Convolution with strides=2 and padding=1
    auto model = convert_model("conv_with_strides_and_asymmetric_padding.onnx");

    Inputs inputs;
    // data (1, 1, 7, 5) input tensor
    inputs.emplace_back(ov::test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f, 4.f},
                                                       {5.f, 6.f, 7.f, 8.f, 9.f},
                                                       {10.f, 11.f, 12.f, 13.f, 14.f},
                                                       {15.f, 16.f, 17.f, 18.f, 19.f},
                                                       {20.f, 21.f, 22.f, 23.f, 24.f},
                                                       {25.f, 26.f, 27.f, 28.f, 29.f},
                                                       {30.f, 31.f, 32.f, 33.f, 34.f}}}}}
                            .get_vector());

    // filters (1, 1, 3, 3) aka convolution weights
    inputs.emplace_back(
        ov::test::NDArray<float, 4>{{{{{1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}}}}}.get_vector());

    // (1, 1, 4, 2)
    auto expected_output =
        ov::test::NDArray<float, 4>({{{{21.f, 33.f}, {99.f, 117.f}, {189.f, 207.f}, {171.f, 183.f}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_conv2d_dilation_assymetric_pads_strides) {
    auto model = convert_model("conv2d_dilation_assym_pads_strides.onnx");

    //   "",                           // auto_pad
    //   vector<int64_t>{1, 1},        // dilations
    //   1,                            // group
    //   vector<int64_t>{3, 3},        // kernel_shape
    //   vector<int64_t>{1, 1, 1, 2},  // pads
    //   vector<int64_t>{3, 1}         // strides

    Inputs inputs;
    // {2, 1, 1, 1}
    inputs.emplace_back(
        ov::test::NDArray<float, 4>({{{{-0.09103918075561523f}}}, {{{-0.32513630390167236f}}}}).get_vector());
    // {2, 1, 3, 3}
    inputs.emplace_back(
        ov::test::NDArray<float, 4>({{{{0.4312484860420227f, -0.12559029459953308f, 0.44889551401138306f},
                                       {-0.3100617825984955f, 0.13522827625274658f, -0.06791308522224426f},
                                       {0.22671669721603394f, -0.17391827702522278f, -0.31299442052841187f}}},
                                     {{{-0.31545522809028625f, 0.06560015678405762f, 0.2656586766242981f},
                                       {0.41363757848739624f, 0.31231558322906494f, -0.376018226146698f},
                                       {-0.005708813667297363f, 0.34922850131988525f, 0.45095211267471313f}}}})
            .get_vector());

    // {2, 2, 1, 2}
    auto expected_output =
        ov::test::NDArray<float, 4>(
            {{{{-0.012311071157455444f, 0.02822777070105076f}}, {{-0.028432954102754593f, -0.037657227367162704f}}},
             {{{-0.04396762326359749f, 0.10081233829259872f}}, {{-0.10154513269662857f, -0.13448859751224518f}}}})
            .get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_conv3d_bias) {
    auto model = convert_model("conv3d_bias.onnx");

    // "",                                 // auto_pad
    // vector<int64_t>{2, 2, 2},           // dilations
    // 1,                                  // group
    // vector<int64_t>{2, 2, 2},           // kernel_shape
    // vector<int64_t>{2, 2, 2, 2, 2, 2},  // pads
    // vector<int64_t>{2, 2, 2}            // strides

    Inputs inputs;
    // X: {2, 1, 4, 4, 4}
    inputs.emplace_back(std::vector<float>{
        0.46796226501464844f,  -0.4613912105560303f,   0.33512794971466064f,   -0.4010460674762726f,
        0.41722816228866577f,  -0.048133403062820435f, 0.20415884256362915f,   0.03189706802368164f,
        -0.04779183864593506f, -0.0795503556728363f,   0.4987630844116211f,    0.3506373167037964f,
        0.48065757751464844f,  0.269855260848999f,     -0.2463444471359253f,   0.19044137001037598f,
        -0.11830493807792664f, -0.2576887905597687f,   -0.33940935134887695f,  -0.257951021194458f,
        -0.08279827237129211f, 0.3513314127922058f,    -0.29122066497802734f,  -0.43358397483825684f,
        -0.13429927825927734f, 0.44032156467437744f,   0.05308258533477783f,   -0.3499870300292969f,
        -0.28474611043930054f, -0.44209951162338257f,  -0.07418054342269897f,  -0.10919415950775146f,
        0.2845439314842224f,   0.3498746156692505f,    -0.19313520193099976f,  0.32609254121780396f,
        0.4880145788192749f,   0.05574071407318115f,   -0.46457427740097046f,  -0.02524462342262268f,
        -0.18780940771102905f, -0.14720159769058228f,  0.207585871219635f,     0.47157740592956543f,
        -0.05567386746406555f, -0.49871665239334106f,  0.2274145483970642f,    0.4589425325393677f,
        -0.4725189805030823f,  -0.4358765780925751f,   0.2841453552246094f,    -0.27037882804870605f,
        0.34227508306503296f,  0.33575427532196045f,   -0.19485199451446533f,  -0.27679920196533203f,
        -0.4238079786300659f,  -0.4385119676589966f,   0.43724071979522705f,   0.3065117597579956f,
        0.45696544647216797f,  0.05291992425918579f,   -0.023618370294570923f, -0.1860884726047516f,
        0.08669537305831909f,  0.32541000843048096f,   0.1846179962158203f,    -0.1984834372997284f,
        -0.2754465937614441f,  0.32004624605178833f,   -0.34846532344818115f,  0.0999596118927002f,
        -0.11374691128730774f, 0.21225297451019287f,   -0.02315312623977661f,  0.1671370267868042f,
        0.22319108247756958f,  0.03609824180603027f,   -0.1587022840976715f,   0.059984564781188965f,
        -0.03951650857925415f, -0.4841443598270416f,   0.32919085025787354f,   -0.23115816712379456f,
        0.39441078901290894f,  -0.3554944396018982f,   -0.17022761702537537f,  -0.055081307888031006f,
        0.15856128931045532f,  -0.4183449149131775f,   -0.2474445104598999f,   0.03603637218475342f,
        -0.2836887538433075f,  0.4602506160736084f,    0.29092925786972046f,   -0.199321448802948f,
        0.380856454372406f,    -0.13847029209136963f,  -0.238397479057312f,    -0.1907123327255249f,
        -0.11061936616897583f, -0.08717870712280273f,  0.24449139833450317f,   -0.14727482199668884f,
        0.1437196135520935f,   0.3955056071281433f,    -0.12538021802902222f,  0.11590522527694702f,
        0.4598066806793213f,   -0.30005723237991333f,  -0.46578651666641235f,  -0.33955082297325134f,
        -0.2671887278556824f,  0.3611910939216614f,    -0.11423084139823914f,  -0.08382436633110046f,
        -0.31819307804107666f, 0.14515334367752075f,   0.3157258629798889f,    0.33179205656051636f,
        -0.2558857202529907f,  0.11888682842254639f,   0.12824326753616333f,   -0.33106181025505066f,
        0.2549159526824951f,   -0.46760573983192444f,  -0.11983257532119751f,  0.1834418773651123f});

    // W: {2, 1, 2, 2, 2}
    inputs.emplace_back(std::vector<float>{0.388077974319458f,
                                           -0.16366064548492432f,
                                           -0.42871910333633423f,
                                           0.4276432394981384f,
                                           0.21517693996429443f,
                                           0.007908165454864502f,
                                           0.33897721767425537f,
                                           0.21843165159225464f,
                                           0.34095364809036255f,
                                           -0.17043980956077576f,
                                           -0.013571739196777344f,
                                           -0.26793742179870605f,
                                           -0.34863436222076416f,
                                           -0.2672275900840759f,
                                           -0.36691007018089294f,
                                           0.37296557426452637f});

    // B: {2}
    inputs.emplace_back(std::vector<float>{0.4310183525085449f, -0.4564093053340912f});

    // {2, 2, 3, 3, 3}
    std::vector<float> expected_output{
        0.5332361459732056f,   0.6628494262695312f,   0.544619083404541f,    0.4242798388004303f,
        0.6271085739135742f,   0.6721994876861572f,   0.43064039945602417f,  0.4246789515018463f,
        0.53834068775177f,     0.6932926177978516f,   0.42797625064849854f,  0.2218741625547409f,
        0.29522019624710083f,  0.8329390287399292f,   0.37605351209640503f,  0.43735477328300476f,
        0.2920728623867035f,   0.6692450046539307f,   0.5527016520500183f,   0.22643595933914185f,
        0.5138190984725952f,   0.3041342794895172f,   0.7423423528671265f,   0.26707080006599426f,
        0.4617553651332855f,   0.32416003942489624f,  0.511577844619751f,    -0.28187549114227295f,
        -0.5031181573867798f,  -0.5793710947036743f,  -0.5992864370346069f,  -0.5055556893348694f,
        -0.7562476396560669f,  -0.44363799691200256f, -0.5730307102203369f,  -0.6302952766418457f,
        -0.4756688177585602f,  -0.728988528251648f,   -0.3900943398475647f,  -0.6694478988647461f,
        -0.38822290301322937f, -0.35774707794189453f, -0.39807581901550293f, -0.547709047794342f,
        -0.35872578620910645f, -0.5326492786407471f,  -0.40852290391921997f, -0.4537881314754486f,
        -0.4545857608318329f,  -0.379546195268631f,   -0.5250767469406128f,  -0.42439910769462585f,
        -0.5558245182037354f,  -0.38563215732574463f, 0.44995537400245667f,  0.5007325410842896f,
        0.49359965324401855f,  0.40685802698135376f,  0.407518208026886f,    0.4628955125808716f,
        0.4301188290119171f,   0.40635955333709717f,  0.4260363280773163f,   0.55128413438797f,
        0.5498291254043579f,   0.27105778455734253f,  0.40259143710136414f,  0.5747092962265015f,
        0.4187920391559601f,   0.4507707953453064f,   0.420598566532135f,    0.3950541913509369f,
        0.593889057636261f,    0.16578882932662964f,  0.5332239270210266f,   0.43014785647392273f,
        0.50260329246521f,     0.39225444197654724f,  0.4074971079826355f,   0.5073125958442688f,
        0.3823610544204712f,   -0.4240749180316925f,  -0.41936254501342773f, -0.5241475105285645f,
        -0.5220003724098206f,  -0.502869725227356f,   -0.5122783780097961f,  -0.4260129928588867f,
        -0.4105660617351532f,  -0.4483373165130615f,  -0.33759188652038574f, -0.735706090927124f,
        -0.3714444637298584f,  -0.4888814687728882f,  -0.6191370487213135f,  -0.2640320658683777f,
        -0.47542816400527954f, -0.5078460574150085f,  -0.4205915927886963f,  -0.5584549903869629f,
        -0.39770257472991943f, -0.45317384600639343f, -0.5598302483558655f,  -0.2542789578437805f,
        -0.5359901785850525f,  -0.48090484738349915f, -0.38603779673576355f, -0.4991581439971924f};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_conv_transpose_w_groups) {
    auto model = convert_model("conv_transpose_w_groups.onnx");

    Inputs inputs;
    inputs.emplace_back(
        std::vector<float>{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});
    inputs.emplace_back(std::vector<float>{0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f,
                                           11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f,
                                           22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.0f});

    std::vector<float> expected_output{28.f, 34.f, 252.f, 274.f, 732.f, 770.f, 1468.f, 1522.f};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_average_pool_2d) {
    // Pooling with strides=2 and no padding
    auto model = convert_model("average_pool_2d.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(
        ov::test::NDArray<float, 4>(
            {{{{0.f, 1.f, 2.f, 3.f}, {4.f, 5.f, 6.f, 7.f}, {8.f, 9.f, 10.f, 11.f}, {12.f, 13.f, 14.f, 15.f}}}})
            .get_vector());

    // (1, 1, 2, 2)
    auto expected_output = ov::test::NDArray<float, 4>({{{{2.5f, 4.5f}, {10.5f, 12.5f}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_average_pool_2d_pads) {
    // Pooling with strides=2 and padding=1
    auto model = convert_model("average_pool_2d_pads.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(
        ov::test::NDArray<float, 4>(
            {{{{0.f, 1.f, 2.f, 3.f}, {4.f, 5.f, 6.f, 7.f}, {8.f, 9.f, 10.f, 11.f}, {12.f, 13.f, 14.f, 15.f}}}})
            .get_vector());

    // (1, 1, 3, 3)
    auto expected_output =
        ov::test::NDArray<float, 4>({{{{0.f, 1.5f, 3.f}, {6.f, 7.5f, 9.f}, {12.f, 13.5f, 15.f}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_average_pool_empty_auto_pad) {
    const auto model = convert_model("average_pool_empty_auto_pad.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    test_case.add_expected_output<float>(
        Shape{1, 1, 3, 3},
        {1.3333333f, 2.3333333f, 1.7777777f, 3.0f, 5.0f, 3.6666666f, 2.6666666f, 4.3333333f, 3.1111111f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_pool_empty_auto_pad) {
    const auto model = convert_model("max_pool_empty_auto_pad.onnx");

    for (const auto& op : model->get_ops()) {
        if (const auto max_pool = ov::as_type_ptr<op::v8::MaxPool>(op)) {
            EXPECT_EQ(max_pool->get_auto_pad(), op::PadType::EXPLICIT);
            return;
        }
    }

    FAIL() << "MaxPool op not found in the imported model";
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_max_pool_2d_pads) {
    // Pooling with strides=2 and padding=1
    auto model = convert_model("max_pool_2d_pads.onnx");

    // input data shape (1, 1, 4, 4)
    Inputs inputs;
    inputs.push_back(
        ov::test::NDArray<float, 4>(
            {{{{0.f, 1.f, 2.f, 3.f}, {4.f, 5.f, 6.f, 7.f}, {8.f, 9.f, 10.f, 11.f}, {12.f, 13.f, 14.f, 15.f}}}})
            .get_vector());

    // (1, 1, 3, 3)
    auto expected_output =
        ov::test::NDArray<float, 4>({{{{0.f, 2.f, 3.f}, {8.f, 10.f, 11.f}, {12.f, 14.f, 15.f}}}}).get_vector();

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_global_lp_pool_p0) {
    auto model = convert_model("global_lp_pool_p0.onnx");

    std::vector<std::int64_t> input{1, 0, -4, 0, 2, 1, -6, 1, 0, 0, 0, 0, -7, 1, -1, 0, -1, 8, 0, 10, 9, 0, 0, 5};

    std::vector<std::int64_t> expected_output{6, 8};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_input(input);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_global_lp_pool_p1) {
    auto model = convert_model("global_lp_pool_p1.onnx");

    Inputs inputs{std::vector<float>(2 * 3 * 4)};
    std::iota(std::begin(inputs.front()), std::end(inputs.front()), 0.f);

    std::vector<float> expected_output{66.f, 210.f};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_global_lp_pool_p2) {
    auto model = convert_model("global_lp_pool_p2.onnx");

    Inputs inputs{std::vector<float>(2 * 3 * 4)};
    std::iota(std::begin(inputs.front()), std::end(inputs.front()), 0.f);

    std::vector<float> expected_output{22.494444f, 61.789967f};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_global_lp_pool_p3) {
    auto model = convert_model("global_lp_pool_p3.onnx");

    Inputs inputs{std::vector<float>(2 * 3 * 4)};
    std::iota(std::begin(inputs.front()), std::end(inputs.front()), 0.f);

    std::vector<float> expected_output{16.331620904278438f, 41.56697946707537f};

    auto test_case = ov::test::TestCase(model, s_device);
    test_case.add_multiple_inputs(inputs);
    test_case.add_expected_output(expected_output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_convtranspose_output_shape) {
    auto model = convert_model("convtranspose_output_shape.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input_from_file<float>(util::path_join({ov::test::utils::getExecutableDirectory(),
                                                          TEST_ONNX_MODELS_DIRNAME,
                                                          "files/convtranspose_output_shape/x.bin"})
                                             .string());
    test_case.add_input_from_file<float>(util::path_join({ov::test::utils::getExecutableDirectory(),
                                                          TEST_ONNX_MODELS_DIRNAME,
                                                          "files/convtranspose_output_shape/w.bin"})
                                             .string());
    test_case.add_expected_output_from_file<float>({1, 2, 10, 8},
                                                   util::path_join({ov::test::utils::getExecutableDirectory(),
                                                                    TEST_ONNX_MODELS_DIRNAME,
                                                                    "files/convtranspose_output_shape/y.bin"})
                                                       .string());

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_convtranspose_output_shape_auto_pads_same_upper) {
    auto model = convert_model("convtranspose_output_shape_auto_pads_same_upper.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({0.f, 0.25f, 0.5f, 0.75f, 1.f, 1.25f, 1.5f, 1.75f, 2.f, 2.25f, 2.5f, 2.75f});
    test_case.add_input<float>({0.f, 0.25f, 0.5f, 0.75f, 1.f, 1.25f, 1.5f, 1.75f, 2.f, 2.25f, 2.5f, 2.75f,

                                3.f, 3.25f, 3.5f, 3.75f, 4.f, 4.25f, 4.5f, 4.75f, 5.f, 5.25f, 5.5f, 5.75f});
    test_case.add_expected_output<float>(
        Shape{1, 2, 6, 6},
        {0.f,     0.f,      0.0625f,  0.25f,  0.4375f,  0.375f,  0.f,     0.4375f,  1.4375f,
         2.375f,  2.5625f,  1.8125f,  0.75f,  2.8125f,  6.375f,  8.625f,  7.875f,   5.0625f,
         3.f,     7.875f,   14.8125f, 18.75f, 15.1875f, 9.f,     5.25f,   12.1875f, 20.9375f,
         24.125f, 18.3125f, 10.3125f, 4.5f,   10.0625f, 16.75f,  18.625f, 13.75f,   7.5625f,

         0.f,     0.75f,    2.3125f,  4.75f,  4.1875f,  2.625f,  3.f,     7.9375f,  14.9375f,
         20.375f, 16.0625f, 9.3125f,  9.75f,  23.0625f, 40.125f, 49.125f, 37.125f,  20.8125f,
         12.f,    28.125f,  48.5625f, 59.25f, 44.4375f, 24.75f,  14.25f,  31.6875f, 52.4375f,
         60.125f, 43.8125f, 23.8125f, 10.5f,  22.8125f, 37.f,    41.125f, 29.5f,    15.8125f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_convtranspose_output_shape_auto_pads_same_lower) {
    auto model = convert_model("convtranspose_output_shape_auto_pads_same_lower.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({0.f, 0.25f, 0.5f, 0.75f, 1.f, 1.25f, 1.5f, 1.75f, 2.f, 2.25f, 2.5f, 2.75f});
    test_case.add_input<float>({0.f, 0.25f, 0.5f, 0.75f, 1.f, 1.25f, 1.5f, 1.75f, 2.f, 2.25f, 2.5f, 2.75f,

                                3.f, 3.25f, 3.5f, 3.75f, 4.f, 4.25f, 4.5f, 4.75f, 5.f, 5.25f, 5.5f, 5.75f});
    test_case.add_expected_output<float>(
        Shape{1, 2, 6, 6},
        {0.f,     0.f,      0.0625f,  0.25f,  0.4375f,  0.375f,  0.f,     0.4375f,  1.4375f,
         2.375f,  2.5625f,  1.8125f,  0.75f,  2.8125f,  6.375f,  8.625f,  7.875f,   5.0625f,
         3.f,     7.875f,   14.8125f, 18.75f, 15.1875f, 9.f,     5.25f,   12.1875f, 20.9375f,
         24.125f, 18.3125f, 10.3125f, 4.5f,   10.0625f, 16.75f,  18.625f, 13.75f,   7.5625f,

         0.f,     0.75f,    2.3125f,  4.75f,  4.1875f,  2.625f,  3.f,     7.9375f,  14.9375f,
         20.375f, 16.0625f, 9.3125f,  9.75f,  23.0625f, 40.125f, 49.125f, 37.125f,  20.8125f,
         12.f,    28.125f,  48.5625f, 59.25f, 44.4375f, 24.75f,  14.25f,  31.6875f, 52.4375f,
         60.125f, 43.8125f, 23.8125f, 10.5f,  22.8125f, 37.f,    41.125f, 29.5f,    15.8125f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_convtranspose_groups_w_pads) {
    auto model = convert_model("convtranspose_groups_w_pads.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({
        0.f,
        0.25f,
        0.5f,
        0.75f,
        1.f,
        1.25f,
        1.5f,
        1.75f,
        2.f,
        2.25f,
        2.5f,
        2.75f,
        3.f,
        3.25f,
        3.5f,
        3.75f,
        4.f,
        4.25f,
    });
    test_case.add_input<float>({
        0.f, 0.25f, 0.5f, 0.75f, 1.f, 1.25f, 1.5f, 1.75f, 2.f, 2.25f, 2.5f, 2.75f,
        3.f, 3.25f, 3.5f, 3.75f, 4.f, 4.25f, 4.5f, 4.75f, 5.f, 5.25f, 5.5f, 5.75f,
        6.f, 6.25f, 6.5f, 6.75f, 7.f, 7.25f, 7.5f, 7.75f, 8.f, 8.25f, 8.5f, 8.75f,
    });
    test_case.add_expected_output<float>(Shape{1, 4, 2, 2},
                                         {1.25f,
                                          1.625f,
                                          5.25f,
                                          5.25f,
                                          9.6875f,
                                          8.375f,
                                          25.5f,
                                          20.4375f,
                                          87.3125f,
                                          62.375f,
                                          157.125f,
                                          111.5625f,
                                          126.125f,
                                          89.375f,
                                          222.9375f,
                                          157.125f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_convtranspose_groups_pads_bias) {
    auto model = convert_model("convtranspose_groups_pads_bias.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({0.f,
                                0.25f,
                                0.5f,
                                0.75f,
                                1.f,
                                1.25f,
                                1.5f,
                                1.75f,
                                2.f,
                                2.25f,
                                2.5f,
                                2.75f,
                                3.f,
                                3.25f,
                                3.5f,
                                3.75f,
                                4.f,
                                4.25f});
    test_case.add_input<float>({0.f, 0.25f, 0.5f, 0.75f, 1.f, 1.25f, 1.5f, 1.75f, 2.f, 2.25f, 2.5f, 2.75f,
                                3.f, 3.25f, 3.5f, 3.75f, 4.f, 4.25f, 4.5f, 4.75f, 5.f, 5.25f, 5.5f, 5.75f,
                                6.f, 6.25f, 6.5f, 6.75f, 7.f, 7.25f, 7.5f, 7.75f, 8.f, 8.25f, 8.5f, 8.75f});
    test_case.add_input<float>({0.f, 0.25f, 0.5f, 0.75f});
    test_case.add_expected_output<float>(Shape{1, 4, 2, 2},
                                         {1.25f,
                                          1.625f,
                                          5.25f,
                                          5.25f,
                                          9.9375f,
                                          8.625f,
                                          25.75f,
                                          20.6875f,
                                          87.8125f,
                                          62.875f,
                                          157.625f,
                                          112.0625f,
                                          126.875f,
                                          90.125f,
                                          223.6875f,
                                          157.875f});
    test_case.run();
}
