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

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_average_pool_2d_dilations_include_pad_ceil_mode) {
    const auto model = convert_model("average_pool_2d_dilations_include_pad_ceil_mode.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    std::vector<float> input_data(1 * 1 * 32 * 32);
    std::iota(std::begin(input_data), std::end(input_data), 0.0f);
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(
        Shape{1, 1, 9, 9},
        {132.0f, 135.0f, 138.0f, 141.0f, 144.0f, 147.0f, 150.0f, 153.0f, 155.0f, 228.0f, 231.0f, 234.0f, 237.0f, 240.0f,
         243.0f, 246.0f, 249.0f, 251.0f, 324.0f, 327.0f, 330.0f, 333.0f, 336.0f, 339.0f, 342.0f, 345.0f, 347.0f, 420.0f,
         423.0f, 426.0f, 429.0f, 432.0f, 435.0f, 438.0f, 441.0f, 443.0f, 516.0f, 519.0f, 522.0f, 525.0f, 528.0f, 531.0f,
         534.0f, 537.0f, 539.0f, 612.0f, 615.0f, 618.0f, 621.0f, 624.0f, 627.0f, 630.0f, 633.0f, 635.0f, 708.0f, 711.0f,
         714.0f, 717.0f, 720.0f, 723.0f, 726.0f, 729.0f, 731.0f, 804.0f, 807.0f, 810.0f, 813.0f, 816.0f, 819.0f, 822.0f,
         825.0f, 827.0f, 868.0f, 871.0f, 874.0f, 877.0f, 880.0f, 883.0f, 886.0f, 889.0f, 891.0f});
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

// Test ConvTranspose with auto_pad='SAME_UPPER' WITHOUT explicit output_shape
// This tests ONNX spec compliance: output = input * stride
OPENVINO_TEST(${BACKEND_NAME}, onnx_model_convtranspose_auto_pad_same_upper_no_output_shape) {
    // Input: [1,1,32,32], kernel=3x3, stride=1
    // Expected output: [1,1,32,32] (per ONNX spec: output = input * stride = 32 * 1)
    // Without the fix, OpenVINO produces [1,1,34,34] and this test fails
    auto model = convert_model("convtranspose_auto_pad_same_upper_no_output_shape.onnx");

    // Verify the output shape is correct
    ASSERT_EQ(model->get_output_shape(0), (ov::Shape{1, 1, 32, 32}));

    // Verify inference produces correct results against ONNXRuntime reference
    auto test_case = ov::test::TestCase(model, s_device);

    // Input data: [1,1,32,32] with diverse values
    test_case.add_input<float>(std::vector<float>{
        -0.12729941f, 2.75357151f,  1.65996969f,  0.99329239f,  -1.21990681f, -1.22002745f, -1.70958197f, 2.33088064f,
        1.00557506f,  1.54036283f,  -1.89707756f, 2.84954929f,  2.16221309f,  -0.93830442f, -1.09087515f, -1.08297741f,
        -0.47878879f, 0.62378216f,  0.15972510f,  -0.54385430f, 1.05926442f,  -1.30253065f, -0.53927678f, -0.16819078f,
        0.28034991f,  1.92587984f,  -1.00163114f, 0.57117218f,  0.96207285f,  -1.76774788f, 1.03772426f,  -1.14737940f,
        -1.67474198f, 2.74442768f,  2.82816005f,  2.04198670f,  -0.47693115f, -1.51163948f, 1.42116511f,  0.20076247f,
        -1.38980877f, 0.47588456f,  -1.82805741f, 2.54660201f,  -0.70610011f, 1.31261146f,  -0.44144461f, 0.60034013f,
        0.73355138f,  -1.07572770f, 2.84792304f,  1.87566411f,  2.69749475f,  2.47413683f,  0.98949987f,  2.60937119f,
        -1.55753744f, -1.02008569f, -1.77386355f, -0.37334836f, -0.05661355f, -0.64325482f, 2.14368749f,  -0.21623337f,
        -0.59532744f, 0.71348041f,  -1.29537892f, 2.01098490f,  -1.62724674f, 2.93443465f,  1.86122382f,  -1.00642157f,
        -1.97238946f, 2.07730722f,  1.53428674f,  1.64503586f,  1.85635173f,  -1.62977672f, -0.20767136f, -1.42065465f,
        2.31551719f,  1.11649060f,  -0.34550989f, -1.68220830f, -0.44508839f, -0.37408340f, 1.64803088f,  1.18778741f,
        2.43606377f,  0.36107463f,  -1.40202880f, 1.56622398f,  1.80392528f,  0.80638599f,  1.85483587f,  0.46897799f,
        0.61366415f,  0.13770509f,  -1.87290442f, -1.46054292f, -1.84285402f, 1.18205202f,  -0.42822009f, 0.54285347f,
        2.53783226f,  -0.75353885f, 0.05191461f,  1.77775574f,  -0.85600919f, -1.61510050f, -0.55124271f, -1.19389355f,
        2.64848828f,  2.04060197f,  1.16701877f,  2.35730290f,  2.01836038f,  -1.06714976f, 2.46279502f,  0.69671118f,
        2.03720069f,  2.48045659f,  -0.40998262f, -1.44974041f, -0.86032420f, 0.13553895f,  2.09007382f,  2.30365300f,
        -1.96523941f, 0.55373651f,  0.08705501f,  -0.88946092f, -1.40067315f, -0.31192413f, 2.71454859f,  -0.38398534f,
        0.59395313f,  1.51509476f,  -0.18185198f, 2.85891032f,  2.81223655f,  -0.74108851f, 0.48624253f,  -0.49560845f,
        -0.57579756f, -1.81556523f, 1.04782164f,  0.51339513f,  -1.74260628f, -0.60676765f, 2.54132938f,  -0.80219054f,
        -1.27552569f, 0.44726381f,  2.92825222f,  -0.78972363f, 1.36067772f,  1.80809808f,  -0.81181228f, 1.64108169f,
        -0.16108434f, 1.16152918f,  1.16764855f,  0.67887342f,  -1.54855120f, 2.17651248f,  -0.39609969f, -1.06740749f,
        -1.79612434f, 0.95446473f,  1.38782179f,  -1.91706085f, 0.56046528f,  -0.86752111f, 1.22586393f,  -1.12816787f,
        1.45468867f,  -0.06632327f, 2.68365002f,  -1.31239533f, -0.29466826f, -1.43263245f, 2.62346816f,  2.38669682f,
        -0.71029186f, 1.29992020f,  2.08611107f,  0.77600408f,  0.64825290f,  -0.79073852f, -1.53448617f, 2.48607874f,
        2.50209022f,  1.16550732f,  -0.30485106f, -0.25395212f, 1.62977839f,  2.48555136f,  2.43543220f,  1.89937770f,
        1.21015823f,  -1.57930017f, -1.19185638f, 2.49277091f,  1.03214526f,  -1.95401478f, -1.49264228f, 1.31750882f,
        -1.97469211f, -1.19595969f, 0.74366897f,  1.45947599f,  1.25980628f,  -0.87865347f, 1.56089616f,  -0.81375456f,
        -0.37300152f, 1.73245704f,  1.24816453f,  2.24611712f,  1.28806448f,  0.84154302f,  -1.53162611f, -0.16142099f,
        -0.67398816f, -0.78005177f, 2.86505270f,  -0.03451138f, 2.46023273f,  1.15569317f,  1.97405648f,  0.51318544f,
        0.88451940f,  0.46258846f,  -1.02378511f, 1.61226058f,  -0.59613818f, -1.87842011f, 1.22736144f,  -1.11444664f,
        2.70229292f,  2.76964283f,  2.57432199f,  -0.14920650f, -1.92271698f, 2.64159274f,  0.14092074f,  2.83327413f,
        2.81809998f,  2.26504731f,  -0.52775556f, -0.07451136f, 2.25568342f,  -0.41538998f, -1.15253627f, 0.78400630f,
        2.68077397f,  1.48014903f,  0.85030586f,  -1.51411748f, 1.07503617f,  2.95026922f,  -1.29957998f, 0.59164828f,
        2.38686538f,  1.70384312f,  1.48507869f,  1.51242042f,  -0.20254424f, -0.53204077f, 2.04680586f,  2.05056691f,
        2.33536148f,  2.56620288f,  0.55671197f,  0.50758147f,  1.99147594f,  1.24981964f,  1.50983441f,  1.97896338f,
        2.45002675f,  -0.31002420f, -0.12208524f, -1.53009033f, 0.89140069f,  -1.82028866f, 0.32799008f,  0.71322316f,
        -0.56729376f, 0.95416629f,  -1.84749877f, -1.81325901f, 2.11300278f,  -0.19904679f, -1.36469746f, 0.61121631f,
        1.84996772f,  -0.92089486f, 1.11445236f,  -1.57326269f, -1.74159145f, 0.65677315f,  0.70317560f,  1.18714952f,
        1.63045669f,  2.87926030f,  0.58150172f,  -0.38521764f, 1.97593093f,  -0.64583874f, 0.19485711f,  -1.60771811f,
        -1.87324631f, 2.81324196f,  2.17990065f,  1.47987103f,  0.04476472f,  -1.13352835f, -1.21781480f, -0.74878550f,
        0.74613333f,  1.57297957f,  1.30098689f,  -0.60033053f, 2.77432632f,  1.68948460f,  0.77177024f,  1.05860376f,
        0.09800031f,  -0.76134503f, -0.22013661f, 1.78923059f,  -1.92803252f, -1.41963685f, -1.76998675f, -1.79635596f,
        2.27730298f,  1.51828933f,  0.37086916f,  -1.51082921f, 0.45807937f,  0.36735886f,  -1.13399065f, 0.16925825f,
        -0.00747633f, 1.07925045f,  1.17546821f,  -1.77347994f, -0.12693693f, 1.12929952f,  0.51568127f,  2.28244925f,
        1.29346812f,  -1.18532789f, -1.64715624f, 1.21209633f,  -1.86744344f, 0.92887789f,  2.70115113f,  0.87737089f,
        -0.05915037f, 1.21644104f,  0.29126444f,  0.72808397f,  2.70732403f,  -0.06948681f, 2.80595279f,  2.52675319f,
        -1.02104437f, -1.65319347f, -1.49610996f, -1.90889084f, -1.52778518f, 1.41503382f,  -1.64405680f, -0.40512186f,
        2.22437644f,  -1.88364029f, 2.07234240f,  -0.59072614f, -1.40917587f, 1.48368585f,  1.14471424f,  2.38736010f,
        1.67535520f,  2.01740456f,  -0.58982712f, -1.11280227f, 1.75307381f,  2.03417373f,  2.95252562f,  0.06308839f,
        -0.13990957f, 1.88206482f,  -0.29598230f, 2.65378666f,  2.29206371f,  0.14497013f,  1.75435531f,  1.77271438f,
        -1.48438060f, 2.51276445f,  0.52626187f,  2.13228726f,  -0.39975199f, 2.47761607f,  -0.05399160f, -1.94581175f,
        2.52690983f,  -1.54356658f, -0.40343180f, 2.75030994f,  2.75303578f,  0.86718947f,  1.15918601f,  0.24222761f,
        -0.53394616f, -0.35667726f, 1.36259234f,  1.76187265f,  1.95789516f,  1.94809067f,  -1.54396951f, 0.47210151f,
        -1.71220624f, 0.74764442f,  0.20765251f,  2.43852091f,  -0.24542494f, -1.41466486f, -1.28504157f, 1.80755317f,
        1.09109032f,  -1.49438667f, -1.57946599f, 1.50484562f,  -1.63618493f, 2.10930037f,  1.53121114f,  -1.59325612f,
        -1.57581139f, 2.93319798f,  -0.12864602f, -0.14678927f, 2.06399775f,  2.73624277f,  2.93000531f,  1.76689088f,
        -0.11870207f, -1.58249640f, 1.88573456f,  0.79202127f,  0.12111004f,  2.53177190f,  -1.44401264f, 0.46312553f,
        -1.94323182f, 0.34330320f,  -1.71848357f, -1.40591037f, -1.41236877f, 1.24605155f,  1.73022437f,  0.91684383f,
        2.81086278f,  -0.12564710f, -0.57143956f, 2.34299564f,  -0.88202083f, 2.81611276f,  -1.93922758f, 2.84939408f,
        -1.78420043f, 2.45571566f,  0.63850552f,  2.96482396f,  -1.63101721f, 0.76927143f,  2.84651279f,  0.61548924f,
        1.14699316f,  1.47874343f,  0.27270532f,  1.13779044f,  0.92157155f,  2.50579000f,  -1.77276814f, -0.59518403f,
        2.75205731f,  2.45131898f,  0.27828377f,  1.10066295f,  -0.61309409f, -1.05939424f, 0.31849203f,  -0.23323886f,
        0.91828054f,  -1.61132681f, 2.87197399f,  2.93105364f,  1.49080861f,  0.68048185f,  -0.45236191f, 2.06897521f,
        1.42365587f,  -1.18691528f, 2.55463600f,  2.11268616f,  2.74899960f,  1.62859750f,  1.06707597f,  0.09121518f,
        2.66364241f,  2.33031940f,  -1.77390671f, -1.86816514f, -0.11768316f, 2.05276656f,  2.93638062f,  -1.24791551f,
        0.97065359f,  -0.09554572f, 2.84957194f,  2.21059465f,  2.19164348f,  0.34346581f,  0.07409751f,  -0.63296461f,
        -1.71812248f, 2.32361197f,  2.06450510f,  2.99858832f,  2.98318410f,  0.77715850f,  1.84493709f,  2.72382855f,
        2.24823689f,  -0.76325947f, 0.25272068f,  -1.35420287f, 2.77025509f,  1.03087318f,  -0.85678595f, 1.35850346f,
        1.09064126f,  -0.20918640f, -1.43221200f, 1.35786593f,  0.60153848f,  1.86159194f,  0.60081750f,  2.26090741f,
        0.75953418f,  0.80468988f,  2.38326812f,  0.01741433f,  -1.32992387f, -1.85608661f, 1.77568626f,  1.10154772f,
        1.52039886f,  -0.93517917f, -1.31814265f, -1.92727673f, -0.24706221f, 0.94958842f,  -0.03877977f, 0.18737461f,
        2.52079344f,  -0.25872266f, 0.56994742f,  1.91826510f,  -0.01728609f, 1.11043346f,  2.31181860f,  2.74760318f,
        -1.26463258f, 2.63293815f,  0.46058145f,  -0.70877808f, 0.29567879f,  2.90016294f,  0.46309048f,  -0.35624194f,
        1.16700423f,  -0.79927188f, -1.62068331f, -1.35560143f, -1.35977077f, -1.24048650f, -1.30586410f, 1.20437372f,
        -1.09059954f, -0.27166358f, 2.48394203f,  0.36980820f,  1.33778870f,  -1.13840067f, -1.03855491f, -1.79565692f,
        -1.15532470f, -0.60704833f, -1.11494756f, -1.55648732f, -1.39682066f, 0.30389383f,  -0.96833140f, -0.17865069f,
        0.51708633f,  1.45197415f,  -1.80343926f, 1.99705195f,  1.13950193f,  -1.59120488f, 2.36789322f,  2.60436201f,
        -1.69461024f, -0.61561173f, 2.03100634f,  1.74129844f,  -1.07739496f, -0.95325339f, -0.14763948f, 0.42261493f,
        1.09127390f,  -0.15543181f, 0.31267357f,  1.73735464f,  -1.81658399f, -0.73781526f, 1.56674790f,  2.47603416f,
        0.55838722f,  0.66056740f,  -1.46413994f, 0.23706183f,  0.66308635f,  -0.78764749f, -0.65378386f, -0.11357918f,
        -1.89964402f, -0.38960418f, -0.94275999f, -0.36251324f, -1.40118933f, 2.45263648f,  0.96796227f,  1.39551163f,
        1.94585621f,  0.49221098f,  -1.56539857f, 0.68553269f,  0.93420559f,  1.72719741f,  0.15829773f,  -1.36209846f,
        -0.58112049f, -0.18458852f, 1.22958624f,  0.85389155f,  -0.21951637f, 2.93257618f,  1.02887404f,  -0.81386602f,
        -1.49108768f, -1.23570430f, -0.77021134f, -1.19659317f, -1.06716490f, -0.57452416f, -1.13313198f, 2.48382711f,
        -1.59883130f, 0.62255692f,  0.05198414f,  2.91189313f,  -1.43980551f, -0.01072200f, 2.84735227f,  2.32753563f,
        2.08536029f,  -0.71048588f, -1.14556205f, 1.34321606f,  2.64687991f,  0.78381449f,  0.85806346f,  -0.60010451f,
        1.84746468f,  -1.06478131f, -0.38160381f, 0.12718219f,  0.53805190f,  -0.78795135f, -1.42581582f, 1.05310023f,
        -0.55684721f, 0.90619111f,  -1.22818637f, 0.40570050f,  0.66294718f,  -1.74088228f, -0.31697860f, -1.32792664f,
        -1.68312514f, 2.94980121f,  -0.38823077f, 2.04937220f,  -0.72679675f, 1.40751362f,  1.80113935f,  0.97819370f,
        0.35788095f,  0.05920457f,  -0.25565866f, 2.64764571f,  2.15309715f,  2.82513452f,  -1.37851393f, 1.65433741f,
        2.69170237f,  -1.09383464f, -1.66751862f, 1.70560324f,  0.87236559f,  2.20914388f,  -1.30113816f, 1.97633660f,
        -0.99186343f, -1.18172026f, -1.17867100f, 2.07287359f,  1.32598615f,  0.61532712f,  -0.20584758f, 2.38600278f,
        -0.03777446f, 2.08299708f,  0.19567454f,  -0.11527786f, 0.31339893f,  -0.49311063f, 1.73804688f,  0.51360196f,
        -0.83893651f, 2.49787283f,  -0.08054389f, 0.71776432f,  2.53236055f,  1.12118995f,  -1.41550982f, 2.69916058f,
        1.13854027f,  -0.32547194f, -1.30363965f, 1.97012591f,  1.10036373f,  0.66730547f,  2.46946287f,  1.94298601f,
        -1.24162555f, -0.44138965f, -0.75755429f, 1.71973145f,  -1.83233786f, 0.84944844f,  1.81229341f,  2.38382816f,
        -0.28959125f, 2.10628653f,  -1.44684136f, 2.23226142f,  -1.36255670f, -0.01356355f, 1.98647678f,  -1.25041282f,
        -0.85374302f, 1.61126280f,  1.60018265f,  1.20573819f,  1.46974218f,  0.71362221f,  -0.74100471f, -0.27152002f,
        -1.09201145f, 2.54225278f,  0.91695899f,  0.00425708f,  0.31002903f,  2.73641682f,  -1.23324299f, 0.93114918f,
        0.52944338f,  1.05727112f,  -1.90944910f, 2.36061954f,  2.66059136f,  0.82566589f,  1.48325408f,  2.61249685f,
        1.53619313f,  -1.23730481f, 0.88144177f,  1.03357518f,  0.12065335f,  1.68222117f,  2.67183518f,  2.62784266f,
        0.25419685f,  -1.43380976f, 2.92420602f,  2.19449043f,  -1.37668657f, 2.60420942f,  2.34948182f,  0.59419030f,
        0.95637721f,  -0.00498648f, -1.72619176f, -0.32401380f, 2.01426721f,  -1.97683990f, -0.33250415f, -0.00915653f,
        0.68697804f,  2.59927797f,  -0.26827002f, -0.26523399f, 1.68750620f,  0.26108971f,  -0.87697589f, 0.26219758f,
        -1.29571486f, -1.11806512f, 0.49183887f,  0.09462725f,  2.57422948f,  -0.18803051f, 0.90294176f,  1.16132140f,
        -1.93452775f, 1.31768692f,  -1.10982013f, 2.80535150f,  -1.25668633f, 0.07312062f,  -1.57325161f, 2.98437119f,
        0.51097506f,  0.97692508f,  -1.66461766f, 1.74980235f,  -0.95047206f, 2.49027133f,  -0.97430182f, -1.04656136f,
        -1.81725168f, 0.36033472f,  0.82420564f,  -1.67145681f, 1.87763810f,  0.26644418f,  0.62195134f,  0.20381373f,
        0.00381530f,  0.79820168f,  -1.22379875f, -1.09035933f, 2.30892801f,  2.73057723f,  -0.13345341f, -0.64627665f,
        1.21999776f,  0.04367086f,  -1.87306821f, -1.21923697f, 1.57986116f,  1.29461968f,  -1.86452007f, -0.89013916f,
        -0.84462601f, 1.35946369f,  -1.90144730f, -1.47945714f, 1.99958038f,  -1.10727668f, 1.26373053f,  -0.80908608f,
        -1.50279307f, -0.78413904f, 1.61133468f,  2.27848244f,  2.15109921f,  -0.01408235f, 1.34042573f,  -0.97507852f,
        -0.53426135f, 2.48167920f,  -1.93499041f, -1.57245731f, -0.96056873f, -1.86733902f, -1.09282279f, 0.91520780f,
        0.10712276f,  2.46335864f,  2.08721781f,  -0.29091325f, -0.70288283f, -0.10153796f, 0.95147473f,  -0.65968180f,
        1.12074459f,  0.04705826f,  0.76023591f,  0.18063265f,  -0.52767122f, 2.74226642f,  1.81802893f,  -1.29943407f,
        2.34233999f,  0.43715599f,  2.47276115f,  1.99927628f,  0.12606752f,  -1.88765347f, -0.65661323f, 0.70817107f,
        1.16739106f,  -0.71056157f, -1.30321968f, 2.17465115f,  2.92201090f,  0.62845093f,  -1.14160359f, -0.63846338f,
        -1.90804660f, 2.57149410f,  -1.41124463f, 0.88258237f,  -0.62972391f, 0.77089000f,  1.25710189f,  2.14870906f,
        -0.96789366f, -1.94502091f, -1.31557190f, 2.50009322f,  2.36945033f,  0.98706549f,  1.00258434f,  1.32518339f,
        -1.12314355f, 2.57205963f,  0.09385262f,  -0.08430736f, 0.59458852f,  -1.76517022f, -1.16858315f, 1.69016802f,
        -1.58600664f, 1.01576054f,  -0.77325445f, -0.05352193f, -0.55653131f, -0.22163641f, 1.59522951f,  -0.51439142f,
        0.83202320f,  0.38025200f,  1.31835580f,  2.68414879f,  1.66286051f,  -0.92529809f, -1.84408438f, -0.68867975f,
        0.97538966f,  -1.74287093f, 0.48183122f,  0.98421425f,  -0.32878053f, 1.85456097f,  -1.46700871f, -1.62431109f,
        1.64094377f,  0.47745657f,  1.44201195f,  0.17413670f,  -0.76798981f, 2.09551167f,  1.99707937f,  1.47348237f,
        -0.63927430f, 0.95115334f,  -0.19513051f, -1.54208958f, 2.58656788f,  -1.31590688f, 2.75118685f,  0.23002887f,
        -1.07433534f, 0.70950472f,  2.36472917f,  1.66112447f,  2.03280568f,  1.29391682f,  1.46138287f,  2.24597836f,
        -0.75165993f, 0.44712481f,  -0.89395279f, 2.93833995f,  2.72029662f,  -1.80286598f, 1.52787590f,  2.62624168f,
        -1.09712327f, 0.83972615f,  2.57744145f,  -1.83027005f, 1.48710132f,  -0.51325494f, 2.62198091f,  2.85529113f});

    // Filters: [1,1,3,3] with diverse weights
    test_case.add_input<float>(std::vector<float>{0.88853300f,
                                                  -0.05157157f,
                                                  0.72408533f,
                                                  0.68909878f,
                                                  -0.36179906f,
                                                  0.65783095f,
                                                  -0.92598474f,
                                                  0.19253975f,
                                                  -0.53998232f});

    // Expected output: [1,1,32,32] - reference values from ONNXRuntime
    // Pattern: corners=4, edges=6, interior=9 (due to padding [1,1])
    test_case.add_expected_output<float>(
        ov::Shape{1, 1, 32, 32},
        {4.46842289f,  1.22262490f,  5.55100536f,  1.41072249f,  0.41408736f,  -0.54378760f, 0.43269676f,
         -1.49118555f, 2.87085438f,  -3.85824871f, 6.36487675f,  -2.87133741f, 3.49232101f,  0.03891933f,
         0.53778422f,  -0.35454345f, -0.66829562f, 2.68652511f,  0.71858454f,  5.39399719f,  1.77880144f,
         3.50128078f,  3.28133345f,  -0.90272337f, 2.17840219f,  -3.85390043f, 1.04400611f,  -1.51806784f,
         -2.02946448f, 3.88446283f,  -3.09743977f, 2.66113234f,  0.58747929f,  -2.70279217f, 2.47248268f,
         -1.27001810f, 4.98003912f,  3.54352379f,  -2.06440759f, -0.00762612f, -0.30980289f, -1.00793648f,
         1.78032136f,  -0.36407304f, 2.22898149f,  -0.33799464f, 0.30464977f,  2.80208588f,  -0.85281968f,
         4.37704372f,  -0.91642666f, 1.32295370f,  2.13909531f,  2.39914703f,  4.52245998f,  1.92935705f,
         0.99381840f,  -0.23142999f, -0.32283169f, -0.80397993f, 2.60288763f,  2.76461053f,  1.77400625f,
         2.02602172f,  -2.06599736f, -3.95538950f, -1.60591578f, -6.31166506f, 4.22680283f,  -3.97488689f,
         2.82776928f,  2.70439839f,  0.25900796f,  3.46565676f,  0.00923875f,  3.00993395f,  -3.54275608f,
         1.68416595f,  -5.52749825f, 3.66339636f,  0.62231672f,  0.49025378f,  2.60456181f,  -1.08307147f,
         -3.33321953f, 2.78648925f,  -3.86621356f, 7.30088854f,  1.98787987f,  3.77585459f,  2.90812898f,
         -0.29400343f, 0.83474034f,  1.32195091f,  3.18010521f,  1.24589455f,  -0.30906326f, -0.65261662f,
         -3.12892914f, -0.51569617f, -4.44491005f, -0.80097729f, 0.30531061f,  4.41142941f,  -1.80266809f,
         2.21251750f,  2.00766182f,  -1.20997047f, 2.61106396f,  1.02403927f,  -0.53221637f, -0.54510701f,
         -2.10067749f, 1.70045757f,  2.52035522f,  0.76437557f,  1.17663324f,  3.08075643f,  -2.97668600f,
         0.60305959f,  0.80618089f,  1.86772847f,  -1.66303468f, 2.42022991f,  -0.82957172f, -1.54041612f,
         3.21708703f,  -1.04228044f, 2.12360740f,  0.85723335f,  2.02134609f,  1.28497660f,  1.55131006f,
         1.09578872f,  -2.02734756f, -1.50798774f, 1.63763690f,  -1.96294093f, 0.71943015f,  3.49259615f,
         -1.46452296f, 4.65465593f,  -0.82723278f, -0.04392552f, -3.05459404f, 2.32351446f,  -5.63220453f,
         -0.99349862f, -1.55407608f, -0.55988765f, -0.51809114f, -0.85369295f, 0.88097763f,  2.04635453f,
         0.13785851f,  5.93505192f,  0.55170661f,  -2.61648273f, 2.52562594f,  -3.05214500f, 0.87410510f,
         2.84636784f,  1.98483109f,  1.77460253f,  4.94668674f,  -0.68056989f, 5.24800491f,  -0.46104658f,
         -0.56622535f, -0.53243929f, -3.56289864f, -0.03714503f, -2.36219978f, -1.06232810f, -1.00614190f,
         -0.51112700f, 0.51696676f,  1.11392629f,  -0.81975353f, 4.76641846f,  -1.58373427f, 2.94765759f,
         -0.12692720f, 0.86798787f,  3.46557307f,  -0.68706006f, 4.87450695f,  0.47746658f,  1.05288696f,
         -0.45139998f, -0.35919702f, -2.25525904f, -1.86702085f, 2.34149599f,  -1.07225323f, 6.21087360f,
         -0.84906971f, 6.94897127f,  2.99065161f,  5.60922432f,  -0.19292337f, 0.12165236f,  4.45734978f,
         -3.96630692f, 1.39540529f,  -0.49167573f, -0.50199747f, -1.69825506f, 3.22229910f,  0.46407574f,
         3.45782661f,  -0.43632483f, 4.21560526f,  -1.72699690f, 1.78810894f,  3.22471094f,  2.08300018f,
         0.14486417f,  2.81813216f,  0.94466984f,  1.49007964f,  1.03669763f,  -0.71876121f, -0.51685190f,
         0.28573492f,  3.58906579f,  -2.34383154f, 3.84810257f,  -0.49746168f, -0.62450027f, 0.49661076f,
         0.80176342f,  2.82535124f,  2.81507540f,  2.60173821f,  -0.68244326f, 1.27675605f,  3.28125763f,
         -1.56876123f, 9.41231441f,  3.85805845f,  4.75027752f,  2.50572419f,  1.28180957f,  4.06122875f,
         -1.69917417f, 7.75314426f,  3.52355289f,  2.19955730f,  1.73140562f,  -2.72315478f, 0.58247280f,
         -5.66675425f, 2.75545073f,  -0.61709297f, -0.04514498f, 1.51967835f,  -2.72686076f, -0.19748497f,
         -1.35065222f, -1.52856445f, -3.86198258f, 2.56883621f,  -0.58629549f, -0.30647904f, 4.91377640f,
         -2.43626595f, 1.04606318f,  0.98978931f,  -0.37671798f, 4.09922600f,  0.73089349f,  4.16362572f,
         -0.76946896f, 2.68730974f,  4.11309624f,  -3.21574855f, 4.95183706f,  -4.28943682f, -1.61499083f,
         -1.45081258f, 1.47826231f,  0.76966119f,  0.81133056f,  -1.66215205f, 0.24901283f,  -2.97149849f,
         -0.11216557f, 1.36748624f,  -2.32634926f, 1.35156596f,  2.58868194f,  -2.87838793f, 4.31999969f,
         0.51695824f,  -0.64729220f, -2.25448775f, -0.10322928f, -3.07785654f, -2.18575621f, -0.18167657f,
         -5.68265390f, -2.75448751f, -0.88184738f, -0.92785287f, 1.04928935f,  -0.59247768f, 0.59002686f,
         -3.25248790f, -1.25988209f, -3.34502339f, -4.12041807f, 2.33082652f,  -1.03418469f, 2.78752375f,
         0.70514745f,  2.58542252f,  -1.22712314f, 2.38373423f,  -0.31430370f, -1.29869127f, 2.55319238f,
         1.26212776f,  -1.23777926f, 3.13514304f,  2.82996035f,  2.15325546f,  1.19183791f,  2.73435473f,
         -1.86676455f, 4.49313879f,  1.16612589f,  1.12818086f,  2.89846063f,  -0.79982543f, 0.38815671f,
         -3.53931141f, -1.64358592f, -4.07860374f, -3.45395875f, 0.23393105f,  -4.62477589f, 3.39298177f,
         1.31435204f,  -3.32152987f, 3.49837494f,  -5.40281296f, 0.67272353f,  0.87986028f,  0.66876608f,
         6.33664799f,  0.73262143f,  -0.89154863f, -0.57507277f, 1.11088562f,  -5.00805235f, 2.31823945f,
         2.19727445f,  -0.12887526f, 3.12535548f,  3.31421924f,  -0.73533010f, 3.64449930f,  5.72789097f,
         1.36092544f,  9.35740376f,  4.35907602f,  -1.41083086f, 4.48775721f,  -3.12211418f, 2.47584105f,
         -2.34577918f, 4.60199022f,  -2.24166799f, 0.74093813f,  3.55622005f,  -6.37225914f, 4.24702120f,
         0.23667920f,  1.75946426f,  3.65114713f,  2.11217690f,  0.36632437f,  0.87714475f,  1.84130561f,
         1.40680528f,  1.21115589f,  6.71010637f,  1.06697333f,  1.08320272f,  1.42915130f,  -2.07551098f,
         0.87193966f,  -2.07431459f, 4.59627199f,  -2.29288721f, 1.73021173f,  -2.65584517f, -0.43173438f,
         -0.64760333f, 3.32931638f,  -0.44082278f, 5.64066553f,  -1.52077508f, 5.72914505f,  1.49023890f,
         -0.36428687f, 0.95052063f,  0.66719997f,  -2.94661903f, 4.98902750f,  2.45712233f,  2.30307198f,
         6.23949289f,  0.92667961f,  2.54689956f,  -2.99809670f, 2.41789818f,  -0.22537935f, 1.52120996f,
         3.66470075f,  -5.09569073f, 3.89627886f,  -6.61506414f, 0.38206470f,  -3.49345016f, -2.34555125f,
         -4.79316711f, -0.24584234f, -2.91898680f, 1.02555788f,  3.72416019f,  -3.39801836f, 2.50208998f,
         -0.58428514f, -3.53273177f, 3.75704503f,  -2.54838347f, 4.86008453f,  -5.32398033f, 8.84705353f,
         -4.32583380f, 4.46357965f,  -1.55088067f, 2.19859552f,  1.30239582f,  2.44519949f,  2.73826361f,
         0.43468201f,  1.74839187f,  -0.28636885f, -0.21449631f, 2.84633756f,  -2.12093592f, 2.09768009f,
         1.24589097f,  1.57257175f,  0.40306979f,  -0.00440824f, -1.41044915f, 0.27802634f,  0.75614303f,
         -1.32255769f, 3.83360291f,  -1.53057098f, 5.38627195f,  2.13750315f,  4.07933283f,  3.37026882f,
         -2.42120934f, 7.50322437f,  -2.37978077f, 2.42308044f,  3.27454519f,  2.84078884f,  0.53679651f,
         3.84856105f,  0.16409910f,  -1.44391596f, 1.17583644f,  3.97972965f,  -1.34696579f, 2.26125479f,
         -1.86779857f, -0.12964523f, 2.65523076f,  0.03177297f,  6.66691351f,  -1.67869496f, 7.06167078f,
         4.80398035f,  5.22412825f,  1.34205806f,  1.23491740f,  -2.47908211f, -3.87420487f, 0.26133448f,
         2.62374115f,  1.94933975f,  7.42345333f,  0.52292359f,  6.99625301f,  0.39230359f,  6.37587357f,
         -2.94989204f, 5.86318779f,  -5.96664476f, 7.26113701f,  -1.31731248f, 1.53893590f,  2.30857229f,
         -1.44005263f, -0.74845022f, -1.37250018f, 0.21673965f,  -1.92625737f, 0.07720304f,  3.70314574f,
         1.61063421f,  2.33963299f,  -0.34364104f, 3.99103904f,  -1.38456416f, 3.75748348f,  -0.40145433f,
         1.91119027f,  0.15553260f,  0.60645616f,  3.48951149f,  -4.03585672f, -0.76474226f, -1.11259747f,
         -1.03603375f, 2.43516612f,  0.20641556f,  3.30384445f,  0.48420370f,  0.91145408f,  -0.86253095f,
         -0.65463471f, -0.92578071f, 0.68438685f,  4.29200459f,  -0.08150956f, 0.22094542f,  -0.18306947f,
         2.69922733f,  0.32101983f,  3.09214401f,  -1.64804518f, 4.93723917f,  -1.23343027f, 1.68790925f,
         -1.88417625f, -3.46729875f, -4.80928516f, -3.88981247f, -2.24656224f, -0.45018762f, 1.26963830f,
         -1.92359257f, 1.32501256f,  -5.09644985f, -0.67172617f, -3.10124111f, -3.59783268f, -4.18507290f,
         -2.59578967f, -3.07823253f, -1.24628472f, 0.89520305f,  -5.26159382f, 1.55842972f,  -1.21157169f,
         -0.08089194f, 0.55897939f,  3.93905234f,  -2.07300115f, 2.57561445f,  0.84461868f,  -0.94251084f,
         1.95271671f,  -0.44854814f, 0.71979952f,  -1.60090256f, -1.91269195f, 0.13136628f,  -1.14125764f,
         0.66804528f,  -3.49120283f, 0.50337470f,  -3.35998917f, 1.67515612f,  2.38808346f,  2.62827778f,
         1.51426780f,  -0.21528447f, 1.04319215f,  -0.71924853f, -1.60394681f, 1.79318678f,  -4.17813206f,
         -1.83091033f, -1.53848517f, -2.93892026f, -3.67382550f, -2.90911794f, -1.75921893f, -2.11628890f,
         -2.85766768f, 1.75146818f,  -3.69157457f, -0.45187509f, 2.64592862f,  1.71017528f,  0.40626037f,
         4.40675783f,  1.21687329f,  2.48120618f,  1.49812376f,  4.48013496f,  1.96704674f,  -0.94705671f,
         2.11740589f,  -1.95300794f, -0.09265548f, 1.82599163f,  -3.53230524f, 5.63902760f,  0.77569848f,
         4.07875872f,  1.26657581f,  1.58090401f,  -1.04151273f, 0.76378548f,  -0.44039226f, -1.65847731f,
         0.23016292f,  1.30441570f,  -0.84908098f, -0.19051600f, -1.23164546f, -0.11010855f, -2.54324627f,
         4.74622917f,  -2.76842260f, 3.04188061f,  6.05935669f,  0.33714068f,  -0.53215230f, 1.21172202f,
         1.05526388f,  2.57071042f,  3.27224278f,  0.22537431f,  1.64087236f,  -2.16304183f, 0.67665178f,
         -2.08081388f, 2.39165902f,  1.42149377f,  -1.94498539f, -0.15793994f, -2.52406406f, -1.09218001f,
         -1.42434406f, -1.13693166f, -0.90806276f, -1.35520852f, -0.89114177f, -0.27586940f, -1.47392857f,
         3.71026659f,  -1.13324571f, 6.73871803f,  -1.30732048f, 2.85595202f,  2.35288811f,  -0.06812286f,
         1.98389220f,  0.19481635f,  1.36196148f,  2.07928705f,  2.28805685f,  2.99274206f,  1.91889942f,
         3.73740888f,  3.55322433f,  -0.91521329f, 1.00105727f,  -0.11956118f, -0.74281299f, -0.71994114f,
         -1.25141609f, 3.51414847f,  -2.95959139f, 3.24749756f,  -1.96460819f, 3.99397016f,  0.89362502f,
         1.77934968f,  3.10791302f,  -1.57731533f, 1.08992982f,  3.61010027f,  -1.46781754f, 1.86637092f,
         0.79423213f,  0.20357013f,  0.18928665f,  -0.64325243f, -1.45625329f, 2.99198794f,  -0.97209716f,
         3.32580304f,  0.93858469f,  3.35336661f,  -2.11049938f, 7.07799101f,  -2.09264421f, 2.63290000f,
         -0.14904469f, 3.02488089f,  -1.44840527f, 5.05596161f,  2.71256614f,  4.54392290f,  -0.18147278f,
         0.40836972f,  -0.97409832f, 1.22351289f,  -2.74947691f, 4.92297602f,  0.34785542f,  6.87182331f,
         0.10613492f,  0.27995783f,  -0.41120863f, 1.23793423f,  -0.70981729f, -1.40435553f, 1.35703075f,
         -3.14161849f, 0.00806129f,  2.41883063f,  -0.73568642f, 1.78956199f,  2.48887062f,  -1.87124658f,
         1.31160164f,  0.10052288f,  -4.04556513f, 3.89579535f,  -0.25866473f, 2.10268784f,  0.54813278f,
         0.97756505f,  2.33083582f,  0.48331344f,  1.59691548f,  2.84565711f,  -1.14739466f, 2.62202072f,
         -1.71579432f, 3.25150108f,  2.53106308f,  1.80714715f,  1.83954227f,  -1.55849314f, 0.25002933f,
         2.44392824f,  -2.42426157f, 4.76816225f,  0.97849786f,  1.95124662f,  3.13304424f,  -1.53952873f,
         3.82290244f,  0.04378355f,  0.24694537f,  3.78889370f,  1.31366324f,  -0.53445184f, 1.60840607f,
         1.40792680f,  -1.31759822f, -0.75849378f, 1.43619704f,  -1.78857040f, -3.12328267f, -0.21848798f,
         -0.22284007f, 2.53751612f,  0.06193513f,  3.17334557f,  3.47897840f,  -0.95583671f, 2.37764120f,
         -0.04311061f, -1.13762045f, -4.34118128f, 3.52375007f,  -4.65443897f, 5.40035057f,  0.10900342f,
         2.88155174f,  4.34761238f,  -0.66580772f, 1.77895868f,  -0.46430019f, 1.33659208f,  -3.87676907f,
         4.86844206f,  -2.51048279f, 3.78174472f,  2.50960112f,  0.62631512f,  -1.48797607f, 1.54579568f,
         -2.55178070f, -1.25191855f, 1.63021159f,  -2.84022045f, -1.65325987f, -0.18720627f, 0.41033754f,
         -2.76701403f, 2.45559692f,  -3.20244360f, -0.34140706f, -1.56772137f, -0.52985138f, 1.84886646f,
         -3.11865759f, -1.29521751f, 2.89618325f,  -1.57364964f, 1.58078098f,  -1.06757283f, -1.31071854f,
         1.83744681f,  -6.41835594f, 1.55159092f,  -2.38235521f, -1.32500398f, -3.44755125f, 1.41241229f,
         -5.85707188f, 2.92639542f,  -2.45455074f, 2.18894148f,  -2.90235519f, 3.44608235f,  -0.73017585f,
         0.69557697f,  -2.35140157f, -3.25743055f, 0.00408101f,  -1.04894626f, 3.93254709f,  0.15747873f,
         4.26778603f,  -1.40763998f, 1.88030314f,  3.56710029f,  -3.33477879f, 1.32685399f,  -3.62404513f,
         -1.80059254f, -3.39359188f, 0.11665065f,  1.77078927f,  -0.21022639f, 3.62710619f,  -2.23914242f,
         3.50252628f,  -2.57065105f, 1.77723086f,  -2.90179086f, 1.18325305f,  -2.25456548f, 0.58041370f,
         -1.75020504f, 2.85054398f,  -3.29892063f, 5.57361221f,  -1.56906581f, 5.56049156f,  -1.32058680f,
         4.56358576f,  3.94858742f,  1.01180983f,  1.49214303f,  0.41054320f,  -2.08273339f, 0.42599005f,
         0.47346768f,  -1.57389641f, 3.07698727f,  -1.51253927f, -1.48645735f, -0.23779047f, -1.15742815f,
         -4.64732075f, 4.72518635f,  -1.06655097f, 4.12859011f,  -1.31923044f, 1.11321735f,  1.97395074f,
         1.30312407f,  3.19384813f,  -1.87027371f, 1.95319867f,  1.03104186f,  -0.01473618f, 6.97375488f,
         0.00601494f,  3.68166161f,  3.49324846f,  2.77545285f,  1.39260459f,  1.17709875f,  -0.83246660f,
         -2.49445462f, -2.60882902f, 0.93630409f,  -2.53302240f, -2.32854223f, 0.79721391f,  1.97388887f,
         0.94940281f,  2.99273109f,  3.10509706f,  -0.36287475f, 0.75319576f,  -0.78976685f, -2.92005682f,
         4.74376535f,  0.07984865f,  2.49105906f,  -0.73470640f, 1.19504333f,  -2.72909355f, -1.14577556f,
         -0.91538036f, 0.46066147f,  -0.01829237f, 1.10657585f,  -0.61527383f, 1.24755621f,  -3.51518178f,
         2.96183729f,  -2.63029003f, 0.41475648f,  -0.21229708f, 1.23330688f,  1.51130962f,  3.38699579f,
         0.48074529f,  3.03461289f,  -2.18941617f, -1.64126337f, -0.21963894f, -3.59941530f, 4.79890966f,
         -1.93569088f, 5.09483051f,  -1.92877293f, 5.43708944f,  1.01301849f,  5.98991632f,  1.46246362f,
         2.18598914f,  0.27719933f,  0.61382431f,  0.50982606f,  1.60895967f,  0.81997985f,  0.35023186f,
         0.62198722f,  -0.49569237f, 0.01365757f,  1.50323224f,  -1.94238460f, 4.36762905f,  -2.53150916f,
         3.02090168f,  1.77294421f,  3.96695471f,  2.86457849f,  3.48836088f,  -0.35922062f, 0.86888671f,
         -1.01417041f, 3.13454247f,  -2.73006344f, 2.85414290f,  -0.79936498f, -0.74460888f, 1.11478472f,
         -0.93073875f, -1.68111801f, 0.62053579f,  0.97649419f,  2.50766397f,  2.65244436f,  -0.40923917f,
         4.23108530f,  -2.58068967f, 2.76489401f,  0.45716035f,  -2.60567927f, 5.38766766f,  0.29117453f,
         -1.74118853f, 3.45414424f,  -1.37885427f, -1.78277636f, 3.34848475f,  -4.27803516f, 1.93966556f,
         -1.43013191f, -0.10290492f});

    // Use higher tolerance for diverse test data due to accumulated rounding errors
    test_case.run_with_tolerance_as_fp(1e-4f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_convtranspose_auto_pad_same_upper_stride2) {
    // Input: [1,1,16,16], kernel=3x3, stride=2
    // Expected output: [1,1,32,32] (per ONNX spec: output = input * stride = 16 * 2)
    auto model = convert_model("convtranspose_auto_pad_same_upper_stride2_no_output_shape.onnx");

    // Verify the output shape is correct
    ASSERT_EQ(model->get_output_shape(0), (ov::Shape{1, 1, 32, 32}));

    // Verify inference runs correctly
    auto test_case = ov::test::TestCase(model, s_device);

    // Input data: [1,1,16,16] with diverse values
    test_case.add_input<float>(std::vector<float>{
        -0.12729941f, 2.75357151f,  1.65996969f,  0.99329239f,  -1.21990681f, -1.22002745f, -1.70958197f, 2.33088064f,
        1.00557506f,  1.54036283f,  -1.89707756f, 2.84954929f,  2.16221309f,  -0.93830442f, -1.09087515f, -1.08297741f,
        -0.47878879f, 0.62378216f,  0.15972510f,  -0.54385430f, 1.05926442f,  -1.30253065f, -0.53927678f, -0.16819078f,
        0.28034991f,  1.92587984f,  -1.00163114f, 0.57117218f,  0.96207285f,  -1.76774788f, 1.03772426f,  -1.14737940f,
        -1.67474198f, 2.74442768f,  2.82816005f,  2.04198670f,  -0.47693115f, -1.51163948f, 1.42116511f,  0.20076247f,
        -1.38980877f, 0.47588456f,  -1.82805741f, 2.54660201f,  -0.70610011f, 1.31261146f,  -0.44144461f, 0.60034013f,
        0.73355138f,  -1.07572770f, 2.84792304f,  1.87566411f,  2.69749475f,  2.47413683f,  0.98949987f,  2.60937119f,
        -1.55753744f, -1.02008569f, -1.77386355f, -0.37334836f, -0.05661355f, -0.64325482f, 2.14368749f,  -0.21623337f,
        -0.59532744f, 0.71348041f,  -1.29537892f, 2.01098490f,  -1.62724674f, 2.93443465f,  1.86122382f,  -1.00642157f,
        -1.97238946f, 2.07730722f,  1.53428674f,  1.64503586f,  1.85635173f,  -1.62977672f, -0.20767136f, -1.42065465f,
        2.31551719f,  1.11649060f,  -0.34550989f, -1.68220830f, -0.44508839f, -0.37408340f, 1.64803088f,  1.18778741f,
        2.43606377f,  0.36107463f,  -1.40202880f, 1.56622398f,  1.80392528f,  0.80638599f,  1.85483587f,  0.46897799f,
        0.61366415f,  0.13770509f,  -1.87290442f, -1.46054292f, -1.84285402f, 1.18205202f,  -0.42822009f, 0.54285347f,
        2.53783226f,  -0.75353885f, 0.05191461f,  1.77775574f,  -0.85600919f, -1.61510050f, -0.55124271f, -1.19389355f,
        2.64848828f,  2.04060197f,  1.16701877f,  2.35730290f,  2.01836038f,  -1.06714976f, 2.46279502f,  0.69671118f,
        2.03720069f,  2.48045659f,  -0.40998262f, -1.44974041f, -0.86032420f, 0.13553895f,  2.09007382f,  2.30365300f,
        -1.96523941f, 0.55373651f,  0.08705501f,  -0.88946092f, -1.40067315f, -0.31192413f, 2.71454859f,  -0.38398534f,
        0.59395313f,  1.51509476f,  -0.18185198f, 2.85891032f,  2.81223655f,  -0.74108851f, 0.48624253f,  -0.49560845f,
        -0.57579756f, -1.81556523f, 1.04782164f,  0.51339513f,  -1.74260628f, -0.60676765f, 2.54132938f,  -0.80219054f,
        -1.27552569f, 0.44726381f,  2.92825222f,  -0.78972363f, 1.36067772f,  1.80809808f,  -0.81181228f, 1.64108169f,
        -0.16108434f, 1.16152918f,  1.16764855f,  0.67887342f,  -1.54855120f, 2.17651248f,  -0.39609969f, -1.06740749f,
        -1.79612434f, 0.95446473f,  1.38782179f,  -1.91706085f, 0.56046528f,  -0.86752111f, 1.22586393f,  -1.12816787f,
        1.45468867f,  -0.06632327f, 2.68365002f,  -1.31239533f, -0.29466826f, -1.43263245f, 2.62346816f,  2.38669682f,
        -0.71029186f, 1.29992020f,  2.08611107f,  0.77600408f,  0.64825290f,  -0.79073852f, -1.53448617f, 2.48607874f,
        2.50209022f,  1.16550732f,  -0.30485106f, -0.25395212f, 1.62977839f,  2.48555136f,  2.43543220f,  1.89937770f,
        1.21015823f,  -1.57930017f, -1.19185638f, 2.49277091f,  1.03214526f,  -1.95401478f, -1.49264228f, 1.31750882f,
        -1.97469211f, -1.19595969f, 0.74366897f,  1.45947599f,  1.25980628f,  -0.87865347f, 1.56089616f,  -0.81375456f,
        -0.37300152f, 1.73245704f,  1.24816453f,  2.24611712f,  1.28806448f,  0.84154302f,  -1.53162611f, -0.16142099f,
        -0.67398816f, -0.78005177f, 2.86505270f,  -0.03451138f, 2.46023273f,  1.15569317f,  1.97405648f,  0.51318544f,
        0.88451940f,  0.46258846f,  -1.02378511f, 1.61226058f,  -0.59613818f, -1.87842011f, 1.22736144f,  -1.11444664f,
        2.70229292f,  2.76964283f,  2.57432199f,  -0.14920650f, -1.92271698f, 2.64159274f,  0.14092074f,  2.83327413f,
        2.81809998f,  2.26504731f,  -0.52775556f, -0.07451136f, 2.25568342f,  -0.41538998f, -1.15253627f, 0.78400630f});

    // Filters: [1,1,3,3] with diverse weights
    test_case.add_input<float>(std::vector<float>{0.87230957f,
                                                  0.39205959f,
                                                  0.14012234f,
                                                  -0.80564702f,
                                                  0.23001446f,
                                                  0.98010772f,
                                                  -0.71983194f,
                                                  0.03665930f,
                                                  0.75474614f});

    // Expected output: [1,1,32,32] - reference values from ONNXRuntime
    // Pattern with stride=2: corners/edges=1, even rows interior=2, odd rows interior=4
    test_case.add_expected_output<float>(
        ov::Shape{1, 1, 32, 32},
        {-0.11104450f, -0.04990895f, 2.38412929f,  1.07956409f,  1.83384430f,  0.65080702f,  1.09905732f,
         0.38942981f,  -0.92495394f, -0.47827616f, -1.23517787f, -0.47832346f, -1.66223788f, -0.67025799f,
         1.79369879f,  0.91384411f,  1.20378125f,  0.39424536f,  1.48457670f,  0.60391402f,  -1.43899965f,
         -0.74376744f, 2.21986628f,  1.11719310f,  2.28540468f,  0.84771639f,  -0.51551759f, -0.36787125f,
         -1.08305824f, -0.42768806f, -1.09754753f, -0.42459169f, 0.10255839f,  -0.02928071f, -2.34317374f,
         0.63336128f,  1.36144710f,  0.38181704f,  0.82670599f,  0.22847161f,  1.95634782f,  -0.28059620f,
         -0.21272862f, -0.28062394f, 0.18156123f,  -0.39322856f, -3.55344152f, 0.53613627f,  1.47437572f,
         0.23129681f,  -0.25541687f, 0.35430571f,  3.03809643f,  -0.43635526f, -4.15507126f, 0.65543753f,
         1.05088472f,  0.49734026f,  2.87514377f,  -0.21582359f, -0.04077911f, -0.25091705f, -0.19667763f,
         -0.24910046f, -0.32601786f, -0.19238044f, -1.60114539f, 0.34550381f,  1.11008382f,  0.12347509f,
         0.08582377f,  -0.17680988f, 2.47561169f,  0.37057382f,  -1.03028870f, -0.55539501f, -0.34312922f,
         -0.27410072f, -3.19042206f, 0.01950765f,  1.25636280f,  0.14677756f,  1.36939812f,  0.81152827f,
         1.92428613f,  -0.46224463f, -3.12512064f, 0.32839602f,  1.51351547f,  0.45645514f,  0.90012801f,
         -0.72746015f, 0.73458099f,  0.36685902f,  -0.89923370f, -0.48954231f, 0.38573477f,  -0.11012834f,
         -0.97181284f, 0.14347892f,  0.48269168f,  0.03673908f,  0.59470236f,  -0.12509435f, -1.38642907f,
         0.24364613f,  2.08757305f,  -0.29960087f, -0.84215367f, -0.12404145f, -0.39304692f, -0.03868631f,
         -0.39070815f, 0.06448453f,  -1.27680624f, 0.44298020f,  2.69453073f,  -0.23038964f, -1.44186962f,
         0.13137786f,  -0.21528083f, 0.22129066f,  2.36711574f,  -0.40660757f, -2.56862283f, 0.23869158f,
         1.94146442f,  -0.26391384f, -1.11624599f, -0.67415076f, 1.34893942f,  1.09884667f,  3.20740843f,
         1.11466277f,  2.68956852f,  0.78064317f,  -1.30286789f, -0.14815354f, 0.35163260f,  -0.64040262f,
         0.43299001f,  0.53741193f,  0.08831605f,  0.07254510f,  -1.51295841f, -0.53461039f, -0.95434141f,
         0.25717652f,  0.64660662f,  -0.75342655f, 0.79814845f,  1.01935852f,  -0.52054280f, -0.24156441f,
         3.04466534f,  0.44981751f,  -2.28233814f, -0.13503034f, 2.07096481f,  0.19330697f,  1.34925091f,
         -0.38521487f, -3.85246754f, 0.63125807f,  0.41133618f,  0.65051770f,  1.12678111f,  0.46968648f,
         2.38560510f,  -0.10970106f, 0.75040388f,  -0.34769893f, -2.62652683f, 0.32688853f,  1.23115122f,
         0.04617827f,  1.31646419f,  -0.31967610f, -1.74555731f, 0.10946033f,  1.93918717f,  -0.42047963f,
         -3.84335542f, 0.58575529f,  3.06481171f,  -0.16241324f, -1.74955571f, 0.30191961f,  1.64214909f,
         -0.10153864f, -0.91632545f, 0.13808692f,  1.84541667f,  0.22620100f,  -4.07511234f, -0.32114053f,
         2.36908340f,  1.22023392f,  2.69987297f,  0.81022996f,  4.50036478f,  1.04009473f,  3.26435685f,
         0.91459346f,  -0.95407212f, 0.44004184f,  3.34293413f,  1.03038883f,  0.15892985f,  -0.66159689f,
         -2.49958611f, -0.38248879f, -0.01522881f, -0.76247549f, -3.78707790f, -0.05301815f, 2.32861257f,
         -0.04808103f, -2.04683614f, -0.20407480f, 3.08827925f,  0.82427019f,  -0.65356654f, -0.06276832f,
         -0.59098351f, 0.16872743f,  1.58561623f,  -0.24743292f, -3.34874964f, 0.65506345f,  1.28014827f,
         0.43142986f,  -0.33487558f, 0.62046278f,  0.65055454f,  0.56908727f,  1.62773299f,  0.22759928f,
         -1.13241577f, 0.60019308f,  3.81229019f,  -0.35825613f, -0.70472544f, -0.23463446f, 0.42931402f,
         -0.40801427f, -1.43779039f, -0.08587552f, -0.32031107f, -0.01302194f, 0.46274894f,  -0.14795791f,
         -2.35751438f, 0.49307913f,  2.27525234f,  -0.04973680f, -1.04734349f, -0.20651235f, 1.86694527f,
         0.24029140f,  -3.89192414f, -0.40346286f, 2.37198591f,  0.85718644f,  -1.66377175f, -0.53908944f,
         2.58668280f,  1.24117339f,  3.18981457f,  0.76598501f,  -1.74859977f, -0.29891950f, 1.22902167f,
         -0.83039248f, 1.09442413f,  0.77703255f,  2.13642788f,  0.53650326f,  0.57989979f,  0.63126540f,
         1.60878873f,  0.72572505f,  -0.74124700f, -0.66255087f, -2.43811059f, -0.00283346f, 0.50524151f,
         -0.56490821f, 0.47962376f,  -0.13693392f, -1.15829837f, 0.16411081f,  1.74290586f,  -0.29795587f,
         -2.88975477f, 0.46255559f,  3.28196836f,  -0.37429029f, -3.95899582f, 0.67496240f,  1.37657273f,
         0.42810839f,  2.63502026f,  -0.23149151f, 0.60264814f,  -0.45367810f, -3.60673046f, 0.47781071f,
         0.79989123f,  0.35290813f,  0.17844808f,  0.37838203f,  0.11674809f,  0.42698774f,  3.13244939f,
         -0.37487221f, -1.43004704f, -0.04776742f, 0.94100583f,  -0.32677111f, 2.44838357f,  0.88599640f,
         0.33547401f,  0.46388656f,  1.32600546f,  -0.18294816f, -3.94107342f, -0.58580464f, 2.06515741f,
         -0.23415489f, -3.72914124f, -0.03908865f, 2.26016045f,  0.71435750f,  3.39625025f,  0.42878875f,
         2.95163321f,  0.88277578f,  -2.32764959f, 0.21771541f,  -0.70899773f, -0.49343297f, 1.14362419f,
         0.67435902f,  1.69836736f,  0.77529871f,  3.53042793f,  0.25640491f,  0.65040445f,  0.71959311f,
         1.53489137f,  0.13178711f,  -1.86548948f, 0.53260243f,  1.36995912f,  0.25680897f,  1.37264013f,
         -0.07947227f, 1.01662922f,  -0.38693222f, -1.29016113f, -0.10237677f, -0.13485539f, -0.08604459f,
         -1.69437313f, 0.37907094f,  0.65831047f,  0.27320829f,  -0.79844785f, 0.56032991f,  2.09670639f,
         0.08305238f,  1.48343229f,  -0.32248691f, -2.63596296f, 0.36025417f,  0.08174121f,  0.41492888f,
         1.11837864f,  0.18548043f,  -0.70399779f, 0.42663908f,  1.44010818f,  0.10787172f,  -1.13147807f,
         0.32547817f,  1.15005159f,  0.09491837f,  -0.52308083f, -0.74695629f, -0.58634639f, -0.63428843f,
         -2.76144528f, -0.73882526f, 0.70623869f,  0.44972122f,  -1.67655194f, -0.10747212f, 0.80237073f,
         0.25637439f,  1.43276262f,  1.08428586f,  1.27698445f,  -0.28219539f, 1.22144282f,  -0.03104378f,
         -0.62756610f, 0.75440288f,  -0.61402321f, -0.26947597f, -0.74777043f, -0.60365415f, -1.43371928f,
         -0.14812300f, -0.05634129f, -0.45088503f, -0.49439669f, 0.14115162f,  0.49051529f,  0.03167416f,
         1.64386559f,  -0.43079510f, -0.65896606f, -0.33594599f, 0.05320048f,  -0.42388308f, -2.75851226f,
         0.27188906f,  1.50353265f,  -0.09849681f, -0.85705006f, 0.12486415f,  -1.51254201f, 0.58373809f,
         3.09443521f,  -0.17332482f, -0.78037411f, 0.01194111f,  -1.38136172f, 0.40890953f,  2.43203354f,
         -0.19689448f, 0.46221966f,  -0.37149647f, -1.13886547f, -0.12679379f, 0.42157954f,  -0.27461278f,
         1.86856675f,  1.06086171f,  2.51518536f,  0.80508572f,  2.75604439f,  0.38888153f,  1.85760128f,
         0.87066072f,  2.31515169f,  0.72375977f,  -2.88983345f, -0.37505311f, 3.19918394f,  0.94986415f,
         0.23887971f,  0.29305291f,  0.45759830f,  0.89173925f,  4.90702391f,  0.94486260f,  -0.61616474f,
         -0.15883447f, -2.56257319f, -0.50321335f, 1.00432706f,  -0.36867908f, 0.51421249f,  -0.00606911f,
         1.01999462f,  0.79922533f,  2.74571896f,  0.85940194f,  -2.13374662f, 0.60919058f,  0.95179880f,
         0.46936795f,  1.05980456f,  0.26843119f,  -0.75534999f, 0.54221374f,  0.68432474f,  0.46425205f,
         2.83795667f,  -0.24545987f, -3.03006506f, 0.56647849f,  1.85250127f,  0.16025364f,  -0.95841265f,
         0.46858561f,  -0.00169635f, 0.57054090f,  2.76141596f,  -0.09430193f, 0.76615191f,  -0.33346125f,
         -0.72778416f, -0.19788700f, -0.95240694f, 0.03117592f,  -1.55101895f, 0.48074719f,  0.19256628f,
         0.52987349f,  -3.62076378f, -0.67339921f, 0.73770154f,  0.29190475f,  0.85360885f,  0.07691285f,
         -1.57974601f, -0.26230460f, -1.02016878f, -0.47515568f, 1.82315814f,  -0.16141382f, -0.25400668f,
         1.15454912f,  1.40268493f,  -0.12500419f, -0.47629589f, 0.30754739f,  1.15691531f,  0.68493927f,
         2.22090101f,  -0.08632649f, 3.20251012f,  1.06771672f,  2.37884092f,  1.07102549f,  -0.99929309f,
         -0.28558210f, -1.08189344f, 0.26725671f,  -0.44495833f, -0.10985772f, 1.58328927f,  -0.45203349f,
         -2.37226248f, 0.12736741f,  0.47258586f,  0.02002391f,  0.80191481f,  -0.20458888f, 0.25668061f,
         -0.32217509f, -1.12150979f, -0.07174706f, -2.49268723f, 0.62438542f,  2.96990681f,  -0.08832218f,
         -0.85486358f, 0.13661781f,  -0.63849354f, 0.34849370f,  1.63146448f,  -0.04182858f, -2.48150706f,
         0.65759069f,  0.53637004f,  0.64685506f,  3.35335040f,  -0.17046107f, -1.11808634f, 0.11184281f,
         0.87585557f,  -0.11399711f, 0.91236836f,  -0.29779127f, -3.54627109f, -0.69151014f, 1.01488912f,
         0.41399992f,  1.30062950f,  0.16867447f,  -1.11122191f, -0.73455322f, -1.60608697f, -0.24932401f,
         -0.05763814f, 1.09586596f,  1.98153841f,  -0.32858312f, -1.94241607f, -0.47830817f, -0.43090692f,
         0.23089638f,  3.89142895f,  1.14138281f,  -2.47375703f, -0.20481308f, 1.20968807f,  0.63656139f,
         4.42386627f,  0.68171442f,  -1.36414325f, -0.30045348f, 2.04152274f,  0.62523311f,  0.46388960f,
         -0.13244176f, 0.89836103f,  -0.41760626f, -2.62362385f, 0.24101412f,  0.61336291f,  0.11808830f,
         1.90710807f,  -0.40082464f, -1.21910131f, -0.13956533f, -2.64211226f, 0.58454251f,  3.13705897f,
         -0.18451542f, 0.24139029f,  -0.29338935f, -1.61048937f, 0.10287714f,  -1.92077112f, 0.67354035f,
         3.50624108f,  -0.18164785f, -1.87024021f, 0.31297556f,  -0.12307799f, 0.41588870f,  2.42616510f,
         -0.18672857f, -2.11779594f, 0.37747252f,  0.27396205f,  -0.08426300f, 1.86296225f,  0.38883132f,
         -0.94323915f, 0.49620023f,  1.17708254f,  0.28497955f,  0.38617599f,  -0.67100704f, 0.80315137f,
         0.83107895f,  -2.32782912f, -0.06213132f, 1.50888872f,  -0.44789508f, -1.40363002f, -0.75094765f,
         -0.70374131f, 0.39060342f,  -0.42592683f, 0.65145653f,  1.30074954f,  -0.78055280f, -1.35522389f,
         0.26961729f,  -0.95277381f, -0.27383637f, 2.89679718f,  0.45085123f,  -2.60635591f, -0.38214812f,
         0.12977712f,  -0.03705173f, -1.09366250f, 0.26716849f,  0.19771111f,  0.26857606f,  0.59748900f,
         0.15615070f,  1.91295481f,  -0.35618916f, -3.27124786f, 0.50062937f,  2.45233321f,  -0.09110866f,
         0.47173327f,  -0.24551916f, 0.40086794f,  -0.41313457f, -2.52935696f, 0.21954069f,  -0.18261617f,
         0.31921908f,  2.90468931f,  -0.44095170f, -2.33046341f, 0.12891512f,  1.24823213f,  -0.19954240f,
         -1.83787775f, 0.28196642f,  2.11038375f,  -0.25949493f, 1.38489258f,  0.56441945f,  -0.81170362f,
         0.01657818f,  2.36782932f,  1.09495592f,  -0.37617213f, -0.48965013f, 1.18613589f,  -0.17229633f,
         -4.02647495f, -0.48188785f, 4.01557207f,  1.01403511f,  2.91894412f,  0.89659697f,  0.20212120f,
         -0.34432143f, -1.00826693f, 0.54463619f,  1.72326279f,  0.86875641f,  3.39664125f,  0.23396173f,
         -1.17612243f, 0.27470002f,  0.44854435f,  -0.34181935f, -2.98652148f, -0.55667067f, 3.69092178f,
         0.93333316f,  -1.17196560f, 0.33459944f,  1.47918475f,  -0.01525531f, -2.22707844f, 0.61727828f,
         3.68759346f,  -0.30186990f, -1.04889023f, -0.06777796f, 0.86538935f,  -0.32952619f, -3.51772356f,
         0.60343564f,  0.64844632f,  0.54897475f,  2.91146469f,  -0.16337740f, -1.74343944f, 0.29900044f,
         -0.40660739f, 0.47983572f,  1.41942823f,  0.17849216f,  0.23830462f,  0.14910755f,  1.27241373f,
         -0.18188129f, 0.46124530f,  -0.35295400f, -3.50686383f, 0.57183403f,  1.13546574f,  1.03429639f,
         2.51294422f,  0.45451698f,  -2.08444500f, -0.02113903f, 2.70593739f,  -0.14767587f, 0.60767323f,
         0.62816793f,  3.20539331f,  0.92196494f,  -0.49699795f, 1.05100906f,  2.26013565f,  0.83216393f,
         3.63441896f,  0.44841534f,  -2.67988253f, -0.57152563f, -1.78150260f, -0.39080334f, 3.02335358f,
         1.00576258f,  1.36869597f,  0.42842695f,  -0.50141394f, -0.79507816f, -1.06808186f, -0.64145792f,
         -2.00758338f, 0.60767984f,  -2.01580143f, 0.57551694f,  1.51333046f,  0.26808354f,  1.38792515f,
         -0.07012015f, -0.09419112f, -0.05841266f, -1.56192648f, 0.37487260f,  -0.40511858f, 0.57171273f,
         0.47400939f,  0.56018460f,  0.85675800f,  0.43688434f,  0.88663441f,  0.27835390f,  2.45844388f,
         -0.36326188f, -0.58766878f, -0.27414420f, -3.17644119f, 0.57337338f,  1.61163938f,  0.23740834f,
         2.58585978f,  -0.44945166f, -0.71260214f, -0.34332931f, -2.52439737f, 0.30304608f,  -3.52362728f,
         -0.68247211f, -0.27047217f, -0.42616078f, 1.58023262f,  0.28038692f,  1.33003724f,  0.56289184f,
         -0.06138960f, 0.55366570f,  -1.14904094f, -0.25336593f, 1.36132395f,  0.70124555f,  -0.02022910f,
         -0.24941041f, 0.12303948f,  -0.10187526f, 3.50916600f,  0.62133038f,  0.99750745f,  0.44566226f,
         -0.55972004f, 0.97199500f,  2.57676005f,  0.54283577f,  3.10014248f,  0.25830218f,  -1.61846662f,
         -0.65520793f, -2.43037510f, -0.01498769f, 1.59090483f,  -0.45420775f, -0.97188962f, -0.27508801f,
         -1.77130401f, 0.17105462f,  -0.44694680f, 0.33570057f,  0.41548443f,  0.28977367f,  1.94263041f,
         -0.20210300f, -2.11870623f, 0.35902870f,  2.18544531f,  -0.18717532f, -0.49705958f, -0.08579575f,
         -1.76133049f, 0.39849016f,  0.69241440f,  0.28709587f,  -0.58624184f, 0.51663941f,  1.16371143f,
         0.29627344f,  0.58445537f,  0.19356707f,  2.05875278f,  -0.35229614f, -1.37111020f, -0.03712916f,
         0.83352011f,  -0.33663434f, -1.40438867f, -0.34966984f, 0.95194757f,  1.15053368f,  -0.11794287f,
         0.03997286f,  2.33593369f,  1.01074147f,  2.93617249f,  0.42088977f,  0.09718347f,  0.83116913f,
         2.48811293f,  0.17136760f,  0.49780375f,  0.33311033f,  -1.00113809f, 0.24487291f,  -0.41914201f,
         -0.35562792f, 0.58815563f,  0.71444333f,  0.47395504f,  -0.18650214f, -1.35570407f, -0.70560229f,
         2.54509521f,  0.42505047f,  -1.83995461f, -0.44284707f, 0.54299653f,  -0.15502702f, -0.03213459f,
         -0.17942318f, -3.07275581f, 0.65900356f,  2.83586407f,  -0.00793812f, -2.01590395f, 0.56588912f,
         1.48021221f,  0.26582614f,  -0.45768893f, 0.45406154f,  1.52134168f,  0.11804007f,  -0.20963341f,
         0.20345224f,  0.49424127f,  0.10640203f,  1.27819598f,  -0.23548537f, -2.30233264f, 0.37084323f,
         2.06046581f,  -0.13712040f, 0.92906392f,  -0.43206379f, -2.82987404f, 0.28231087f,  2.10079718f,
         -0.25633883f, 2.84239411f,  1.03475201f,  2.84745359f,  1.05726886f,  -0.01740289f, 1.11431837f,
         2.41779542f,  -0.05976301f, -3.49511290f, -0.66362923f, 3.05981731f,  1.07802868f,  -0.05566132f,
         0.12761687f,  3.61174250f,  1.12962532f,  2.60588002f,  1.13728905f,  2.70530295f,  0.90499169f,
         0.94310760f,  -0.24444288f, -2.07220197f, 0.02989146f,  3.60318017f,  0.86250836f,  0.85593671f,
         -0.23171920f, -3.36479807f, -0.40686870f, 2.25096083f,  0.26652235f,  -2.17709422f, 0.62156641f,
         0.41718364f,  0.63705790f,  0.64055347f,  0.59213126f,  2.64332056f,  -0.03431965f, 1.40279269f,
         -0.44225270f, -4.01266098f, 0.60760450f,  2.47551322f,  0.03241381f,  -2.14450121f, 0.65169400f,
         0.50652003f,  0.64820373f,  0.93721294f,  0.52099365f,  2.64517498f,  -0.12139141f, -0.45722741f,
         -0.01713869f, -1.89031374f, 0.51883978f,  2.54547048f,  -0.09554570f, 0.52141047f,  -0.26510000f,
         -1.76124203f, 0.18033278f});

    // Use higher tolerance for diverse test data due to accumulated rounding errors
    test_case.run_with_tolerance_as_fp(1e-4f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_convtranspose_auto_pad_same_lower_no_output_shape) {
    // Test SAME_LOWER variant
    // Input: [1,1,32,32], kernel=3x3, stride=1
    // Expected output: [1,1,32,32]
    auto model = convert_model("convtranspose_auto_pad_same_lower_no_output_shape.onnx");

    // Verify the output shape is correct
    ASSERT_EQ(model->get_output_shape(0), (ov::Shape{1, 1, 32, 32}));

    // Verify inference runs correctly
    auto test_case = ov::test::TestCase(model, s_device);

    // Input data: [1,1,32,32] with diverse values
    test_case.add_input<float>(std::vector<float>{
        -0.12729941f, 2.75357151f,  1.65996969f,  0.99329239f,  -1.21990681f, -1.22002745f, -1.70958197f, 2.33088064f,
        1.00557506f,  1.54036283f,  -1.89707756f, 2.84954929f,  2.16221309f,  -0.93830442f, -1.09087515f, -1.08297741f,
        -0.47878879f, 0.62378216f,  0.15972510f,  -0.54385430f, 1.05926442f,  -1.30253065f, -0.53927678f, -0.16819078f,
        0.28034991f,  1.92587984f,  -1.00163114f, 0.57117218f,  0.96207285f,  -1.76774788f, 1.03772426f,  -1.14737940f,
        -1.67474198f, 2.74442768f,  2.82816005f,  2.04198670f,  -0.47693115f, -1.51163948f, 1.42116511f,  0.20076247f,
        -1.38980877f, 0.47588456f,  -1.82805741f, 2.54660201f,  -0.70610011f, 1.31261146f,  -0.44144461f, 0.60034013f,
        0.73355138f,  -1.07572770f, 2.84792304f,  1.87566411f,  2.69749475f,  2.47413683f,  0.98949987f,  2.60937119f,
        -1.55753744f, -1.02008569f, -1.77386355f, -0.37334836f, -0.05661355f, -0.64325482f, 2.14368749f,  -0.21623337f,
        -0.59532744f, 0.71348041f,  -1.29537892f, 2.01098490f,  -1.62724674f, 2.93443465f,  1.86122382f,  -1.00642157f,
        -1.97238946f, 2.07730722f,  1.53428674f,  1.64503586f,  1.85635173f,  -1.62977672f, -0.20767136f, -1.42065465f,
        2.31551719f,  1.11649060f,  -0.34550989f, -1.68220830f, -0.44508839f, -0.37408340f, 1.64803088f,  1.18778741f,
        2.43606377f,  0.36107463f,  -1.40202880f, 1.56622398f,  1.80392528f,  0.80638599f,  1.85483587f,  0.46897799f,
        0.61366415f,  0.13770509f,  -1.87290442f, -1.46054292f, -1.84285402f, 1.18205202f,  -0.42822009f, 0.54285347f,
        2.53783226f,  -0.75353885f, 0.05191461f,  1.77775574f,  -0.85600919f, -1.61510050f, -0.55124271f, -1.19389355f,
        2.64848828f,  2.04060197f,  1.16701877f,  2.35730290f,  2.01836038f,  -1.06714976f, 2.46279502f,  0.69671118f,
        2.03720069f,  2.48045659f,  -0.40998262f, -1.44974041f, -0.86032420f, 0.13553895f,  2.09007382f,  2.30365300f,
        -1.96523941f, 0.55373651f,  0.08705501f,  -0.88946092f, -1.40067315f, -0.31192413f, 2.71454859f,  -0.38398534f,
        0.59395313f,  1.51509476f,  -0.18185198f, 2.85891032f,  2.81223655f,  -0.74108851f, 0.48624253f,  -0.49560845f,
        -0.57579756f, -1.81556523f, 1.04782164f,  0.51339513f,  -1.74260628f, -0.60676765f, 2.54132938f,  -0.80219054f,
        -1.27552569f, 0.44726381f,  2.92825222f,  -0.78972363f, 1.36067772f,  1.80809808f,  -0.81181228f, 1.64108169f,
        -0.16108434f, 1.16152918f,  1.16764855f,  0.67887342f,  -1.54855120f, 2.17651248f,  -0.39609969f, -1.06740749f,
        -1.79612434f, 0.95446473f,  1.38782179f,  -1.91706085f, 0.56046528f,  -0.86752111f, 1.22586393f,  -1.12816787f,
        1.45468867f,  -0.06632327f, 2.68365002f,  -1.31239533f, -0.29466826f, -1.43263245f, 2.62346816f,  2.38669682f,
        -0.71029186f, 1.29992020f,  2.08611107f,  0.77600408f,  0.64825290f,  -0.79073852f, -1.53448617f, 2.48607874f,
        2.50209022f,  1.16550732f,  -0.30485106f, -0.25395212f, 1.62977839f,  2.48555136f,  2.43543220f,  1.89937770f,
        1.21015823f,  -1.57930017f, -1.19185638f, 2.49277091f,  1.03214526f,  -1.95401478f, -1.49264228f, 1.31750882f,
        -1.97469211f, -1.19595969f, 0.74366897f,  1.45947599f,  1.25980628f,  -0.87865347f, 1.56089616f,  -0.81375456f,
        -0.37300152f, 1.73245704f,  1.24816453f,  2.24611712f,  1.28806448f,  0.84154302f,  -1.53162611f, -0.16142099f,
        -0.67398816f, -0.78005177f, 2.86505270f,  -0.03451138f, 2.46023273f,  1.15569317f,  1.97405648f,  0.51318544f,
        0.88451940f,  0.46258846f,  -1.02378511f, 1.61226058f,  -0.59613818f, -1.87842011f, 1.22736144f,  -1.11444664f,
        2.70229292f,  2.76964283f,  2.57432199f,  -0.14920650f, -1.92271698f, 2.64159274f,  0.14092074f,  2.83327413f,
        2.81809998f,  2.26504731f,  -0.52775556f, -0.07451136f, 2.25568342f,  -0.41538998f, -1.15253627f, 0.78400630f,
        2.68077397f,  1.48014903f,  0.85030586f,  -1.51411748f, 1.07503617f,  2.95026922f,  -1.29957998f, 0.59164828f,
        2.38686538f,  1.70384312f,  1.48507869f,  1.51242042f,  -0.20254424f, -0.53204077f, 2.04680586f,  2.05056691f,
        2.33536148f,  2.56620288f,  0.55671197f,  0.50758147f,  1.99147594f,  1.24981964f,  1.50983441f,  1.97896338f,
        2.45002675f,  -0.31002420f, -0.12208524f, -1.53009033f, 0.89140069f,  -1.82028866f, 0.32799008f,  0.71322316f,
        -0.56729376f, 0.95416629f,  -1.84749877f, -1.81325901f, 2.11300278f,  -0.19904679f, -1.36469746f, 0.61121631f,
        1.84996772f,  -0.92089486f, 1.11445236f,  -1.57326269f, -1.74159145f, 0.65677315f,  0.70317560f,  1.18714952f,
        1.63045669f,  2.87926030f,  0.58150172f,  -0.38521764f, 1.97593093f,  -0.64583874f, 0.19485711f,  -1.60771811f,
        -1.87324631f, 2.81324196f,  2.17990065f,  1.47987103f,  0.04476472f,  -1.13352835f, -1.21781480f, -0.74878550f,
        0.74613333f,  1.57297957f,  1.30098689f,  -0.60033053f, 2.77432632f,  1.68948460f,  0.77177024f,  1.05860376f,
        0.09800031f,  -0.76134503f, -0.22013661f, 1.78923059f,  -1.92803252f, -1.41963685f, -1.76998675f, -1.79635596f,
        2.27730298f,  1.51828933f,  0.37086916f,  -1.51082921f, 0.45807937f,  0.36735886f,  -1.13399065f, 0.16925825f,
        -0.00747633f, 1.07925045f,  1.17546821f,  -1.77347994f, -0.12693693f, 1.12929952f,  0.51568127f,  2.28244925f,
        1.29346812f,  -1.18532789f, -1.64715624f, 1.21209633f,  -1.86744344f, 0.92887789f,  2.70115113f,  0.87737089f,
        -0.05915037f, 1.21644104f,  0.29126444f,  0.72808397f,  2.70732403f,  -0.06948681f, 2.80595279f,  2.52675319f,
        -1.02104437f, -1.65319347f, -1.49610996f, -1.90889084f, -1.52778518f, 1.41503382f,  -1.64405680f, -0.40512186f,
        2.22437644f,  -1.88364029f, 2.07234240f,  -0.59072614f, -1.40917587f, 1.48368585f,  1.14471424f,  2.38736010f,
        1.67535520f,  2.01740456f,  -0.58982712f, -1.11280227f, 1.75307381f,  2.03417373f,  2.95252562f,  0.06308839f,
        -0.13990957f, 1.88206482f,  -0.29598230f, 2.65378666f,  2.29206371f,  0.14497013f,  1.75435531f,  1.77271438f,
        -1.48438060f, 2.51276445f,  0.52626187f,  2.13228726f,  -0.39975199f, 2.47761607f,  -0.05399160f, -1.94581175f,
        2.52690983f,  -1.54356658f, -0.40343180f, 2.75030994f,  2.75303578f,  0.86718947f,  1.15918601f,  0.24222761f,
        -0.53394616f, -0.35667726f, 1.36259234f,  1.76187265f,  1.95789516f,  1.94809067f,  -1.54396951f, 0.47210151f,
        -1.71220624f, 0.74764442f,  0.20765251f,  2.43852091f,  -0.24542494f, -1.41466486f, -1.28504157f, 1.80755317f,
        1.09109032f,  -1.49438667f, -1.57946599f, 1.50484562f,  -1.63618493f, 2.10930037f,  1.53121114f,  -1.59325612f,
        -1.57581139f, 2.93319798f,  -0.12864602f, -0.14678927f, 2.06399775f,  2.73624277f,  2.93000531f,  1.76689088f,
        -0.11870207f, -1.58249640f, 1.88573456f,  0.79202127f,  0.12111004f,  2.53177190f,  -1.44401264f, 0.46312553f,
        -1.94323182f, 0.34330320f,  -1.71848357f, -1.40591037f, -1.41236877f, 1.24605155f,  1.73022437f,  0.91684383f,
        2.81086278f,  -0.12564710f, -0.57143956f, 2.34299564f,  -0.88202083f, 2.81611276f,  -1.93922758f, 2.84939408f,
        -1.78420043f, 2.45571566f,  0.63850552f,  2.96482396f,  -1.63101721f, 0.76927143f,  2.84651279f,  0.61548924f,
        1.14699316f,  1.47874343f,  0.27270532f,  1.13779044f,  0.92157155f,  2.50579000f,  -1.77276814f, -0.59518403f,
        2.75205731f,  2.45131898f,  0.27828377f,  1.10066295f,  -0.61309409f, -1.05939424f, 0.31849203f,  -0.23323886f,
        0.91828054f,  -1.61132681f, 2.87197399f,  2.93105364f,  1.49080861f,  0.68048185f,  -0.45236191f, 2.06897521f,
        1.42365587f,  -1.18691528f, 2.55463600f,  2.11268616f,  2.74899960f,  1.62859750f,  1.06707597f,  0.09121518f,
        2.66364241f,  2.33031940f,  -1.77390671f, -1.86816514f, -0.11768316f, 2.05276656f,  2.93638062f,  -1.24791551f,
        0.97065359f,  -0.09554572f, 2.84957194f,  2.21059465f,  2.19164348f,  0.34346581f,  0.07409751f,  -0.63296461f,
        -1.71812248f, 2.32361197f,  2.06450510f,  2.99858832f,  2.98318410f,  0.77715850f,  1.84493709f,  2.72382855f,
        2.24823689f,  -0.76325947f, 0.25272068f,  -1.35420287f, 2.77025509f,  1.03087318f,  -0.85678595f, 1.35850346f,
        1.09064126f,  -0.20918640f, -1.43221200f, 1.35786593f,  0.60153848f,  1.86159194f,  0.60081750f,  2.26090741f,
        0.75953418f,  0.80468988f,  2.38326812f,  0.01741433f,  -1.32992387f, -1.85608661f, 1.77568626f,  1.10154772f,
        1.52039886f,  -0.93517917f, -1.31814265f, -1.92727673f, -0.24706221f, 0.94958842f,  -0.03877977f, 0.18737461f,
        2.52079344f,  -0.25872266f, 0.56994742f,  1.91826510f,  -0.01728609f, 1.11043346f,  2.31181860f,  2.74760318f,
        -1.26463258f, 2.63293815f,  0.46058145f,  -0.70877808f, 0.29567879f,  2.90016294f,  0.46309048f,  -0.35624194f,
        1.16700423f,  -0.79927188f, -1.62068331f, -1.35560143f, -1.35977077f, -1.24048650f, -1.30586410f, 1.20437372f,
        -1.09059954f, -0.27166358f, 2.48394203f,  0.36980820f,  1.33778870f,  -1.13840067f, -1.03855491f, -1.79565692f,
        -1.15532470f, -0.60704833f, -1.11494756f, -1.55648732f, -1.39682066f, 0.30389383f,  -0.96833140f, -0.17865069f,
        0.51708633f,  1.45197415f,  -1.80343926f, 1.99705195f,  1.13950193f,  -1.59120488f, 2.36789322f,  2.60436201f,
        -1.69461024f, -0.61561173f, 2.03100634f,  1.74129844f,  -1.07739496f, -0.95325339f, -0.14763948f, 0.42261493f,
        1.09127390f,  -0.15543181f, 0.31267357f,  1.73735464f,  -1.81658399f, -0.73781526f, 1.56674790f,  2.47603416f,
        0.55838722f,  0.66056740f,  -1.46413994f, 0.23706183f,  0.66308635f,  -0.78764749f, -0.65378386f, -0.11357918f,
        -1.89964402f, -0.38960418f, -0.94275999f, -0.36251324f, -1.40118933f, 2.45263648f,  0.96796227f,  1.39551163f,
        1.94585621f,  0.49221098f,  -1.56539857f, 0.68553269f,  0.93420559f,  1.72719741f,  0.15829773f,  -1.36209846f,
        -0.58112049f, -0.18458852f, 1.22958624f,  0.85389155f,  -0.21951637f, 2.93257618f,  1.02887404f,  -0.81386602f,
        -1.49108768f, -1.23570430f, -0.77021134f, -1.19659317f, -1.06716490f, -0.57452416f, -1.13313198f, 2.48382711f,
        -1.59883130f, 0.62255692f,  0.05198414f,  2.91189313f,  -1.43980551f, -0.01072200f, 2.84735227f,  2.32753563f,
        2.08536029f,  -0.71048588f, -1.14556205f, 1.34321606f,  2.64687991f,  0.78381449f,  0.85806346f,  -0.60010451f,
        1.84746468f,  -1.06478131f, -0.38160381f, 0.12718219f,  0.53805190f,  -0.78795135f, -1.42581582f, 1.05310023f,
        -0.55684721f, 0.90619111f,  -1.22818637f, 0.40570050f,  0.66294718f,  -1.74088228f, -0.31697860f, -1.32792664f,
        -1.68312514f, 2.94980121f,  -0.38823077f, 2.04937220f,  -0.72679675f, 1.40751362f,  1.80113935f,  0.97819370f,
        0.35788095f,  0.05920457f,  -0.25565866f, 2.64764571f,  2.15309715f,  2.82513452f,  -1.37851393f, 1.65433741f,
        2.69170237f,  -1.09383464f, -1.66751862f, 1.70560324f,  0.87236559f,  2.20914388f,  -1.30113816f, 1.97633660f,
        -0.99186343f, -1.18172026f, -1.17867100f, 2.07287359f,  1.32598615f,  0.61532712f,  -0.20584758f, 2.38600278f,
        -0.03777446f, 2.08299708f,  0.19567454f,  -0.11527786f, 0.31339893f,  -0.49311063f, 1.73804688f,  0.51360196f,
        -0.83893651f, 2.49787283f,  -0.08054389f, 0.71776432f,  2.53236055f,  1.12118995f,  -1.41550982f, 2.69916058f,
        1.13854027f,  -0.32547194f, -1.30363965f, 1.97012591f,  1.10036373f,  0.66730547f,  2.46946287f,  1.94298601f,
        -1.24162555f, -0.44138965f, -0.75755429f, 1.71973145f,  -1.83233786f, 0.84944844f,  1.81229341f,  2.38382816f,
        -0.28959125f, 2.10628653f,  -1.44684136f, 2.23226142f,  -1.36255670f, -0.01356355f, 1.98647678f,  -1.25041282f,
        -0.85374302f, 1.61126280f,  1.60018265f,  1.20573819f,  1.46974218f,  0.71362221f,  -0.74100471f, -0.27152002f,
        -1.09201145f, 2.54225278f,  0.91695899f,  0.00425708f,  0.31002903f,  2.73641682f,  -1.23324299f, 0.93114918f,
        0.52944338f,  1.05727112f,  -1.90944910f, 2.36061954f,  2.66059136f,  0.82566589f,  1.48325408f,  2.61249685f,
        1.53619313f,  -1.23730481f, 0.88144177f,  1.03357518f,  0.12065335f,  1.68222117f,  2.67183518f,  2.62784266f,
        0.25419685f,  -1.43380976f, 2.92420602f,  2.19449043f,  -1.37668657f, 2.60420942f,  2.34948182f,  0.59419030f,
        0.95637721f,  -0.00498648f, -1.72619176f, -0.32401380f, 2.01426721f,  -1.97683990f, -0.33250415f, -0.00915653f,
        0.68697804f,  2.59927797f,  -0.26827002f, -0.26523399f, 1.68750620f,  0.26108971f,  -0.87697589f, 0.26219758f,
        -1.29571486f, -1.11806512f, 0.49183887f,  0.09462725f,  2.57422948f,  -0.18803051f, 0.90294176f,  1.16132140f,
        -1.93452775f, 1.31768692f,  -1.10982013f, 2.80535150f,  -1.25668633f, 0.07312062f,  -1.57325161f, 2.98437119f,
        0.51097506f,  0.97692508f,  -1.66461766f, 1.74980235f,  -0.95047206f, 2.49027133f,  -0.97430182f, -1.04656136f,
        -1.81725168f, 0.36033472f,  0.82420564f,  -1.67145681f, 1.87763810f,  0.26644418f,  0.62195134f,  0.20381373f,
        0.00381530f,  0.79820168f,  -1.22379875f, -1.09035933f, 2.30892801f,  2.73057723f,  -0.13345341f, -0.64627665f,
        1.21999776f,  0.04367086f,  -1.87306821f, -1.21923697f, 1.57986116f,  1.29461968f,  -1.86452007f, -0.89013916f,
        -0.84462601f, 1.35946369f,  -1.90144730f, -1.47945714f, 1.99958038f,  -1.10727668f, 1.26373053f,  -0.80908608f,
        -1.50279307f, -0.78413904f, 1.61133468f,  2.27848244f,  2.15109921f,  -0.01408235f, 1.34042573f,  -0.97507852f,
        -0.53426135f, 2.48167920f,  -1.93499041f, -1.57245731f, -0.96056873f, -1.86733902f, -1.09282279f, 0.91520780f,
        0.10712276f,  2.46335864f,  2.08721781f,  -0.29091325f, -0.70288283f, -0.10153796f, 0.95147473f,  -0.65968180f,
        1.12074459f,  0.04705826f,  0.76023591f,  0.18063265f,  -0.52767122f, 2.74226642f,  1.81802893f,  -1.29943407f,
        2.34233999f,  0.43715599f,  2.47276115f,  1.99927628f,  0.12606752f,  -1.88765347f, -0.65661323f, 0.70817107f,
        1.16739106f,  -0.71056157f, -1.30321968f, 2.17465115f,  2.92201090f,  0.62845093f,  -1.14160359f, -0.63846338f,
        -1.90804660f, 2.57149410f,  -1.41124463f, 0.88258237f,  -0.62972391f, 0.77089000f,  1.25710189f,  2.14870906f,
        -0.96789366f, -1.94502091f, -1.31557190f, 2.50009322f,  2.36945033f,  0.98706549f,  1.00258434f,  1.32518339f,
        -1.12314355f, 2.57205963f,  0.09385262f,  -0.08430736f, 0.59458852f,  -1.76517022f, -1.16858315f, 1.69016802f,
        -1.58600664f, 1.01576054f,  -0.77325445f, -0.05352193f, -0.55653131f, -0.22163641f, 1.59522951f,  -0.51439142f,
        0.83202320f,  0.38025200f,  1.31835580f,  2.68414879f,  1.66286051f,  -0.92529809f, -1.84408438f, -0.68867975f,
        0.97538966f,  -1.74287093f, 0.48183122f,  0.98421425f,  -0.32878053f, 1.85456097f,  -1.46700871f, -1.62431109f,
        1.64094377f,  0.47745657f,  1.44201195f,  0.17413670f,  -0.76798981f, 2.09551167f,  1.99707937f,  1.47348237f,
        -0.63927430f, 0.95115334f,  -0.19513051f, -1.54208958f, 2.58656788f,  -1.31590688f, 2.75118685f,  0.23002887f,
        -1.07433534f, 0.70950472f,  2.36472917f,  1.66112447f,  2.03280568f,  1.29391682f,  1.46138287f,  2.24597836f,
        -0.75165993f, 0.44712481f,  -0.89395279f, 2.93833995f,  2.72029662f,  -1.80286598f, 1.52787590f,  2.62624168f,
        -1.09712327f, 0.83972615f,  2.57744145f,  -1.83027005f, 1.48710132f,  -0.51325494f, 2.62198091f,  2.85529113f});

    // Filters: [1,1,3,3] with diverse weights
    test_case.add_input<float>(std::vector<float>{0.88853300f,
                                                  -0.05157157f,
                                                  0.72408533f,
                                                  0.68909878f,
                                                  -0.36179906f,
                                                  0.65783095f,
                                                  -0.92598474f,
                                                  0.19253975f,
                                                  -0.53998232f});

    // Expected output: [1,1,32,32] - reference values from ONNXRuntime
    // SAME_LOWER produces DIFFERENT output than SAME_UPPER due to different padding
    // Pattern: corners=4, edges=6, interior=9 (due to padding [1,1])
    test_case.add_expected_output<float>(
        ov::Shape{1, 1, 32, 32},
        {4.46842289f,  1.22262490f,  5.55100536f,  1.41072249f,  0.41408736f,  -0.54378760f, 0.43269676f,
         -1.49118555f, 2.87085438f,  -3.85824871f, 6.36487675f,  -2.87133741f, 3.49232101f,  0.03891933f,
         0.53778422f,  -0.35454345f, -0.66829562f, 2.68652511f,  0.71858454f,  5.39399719f,  1.77880144f,
         3.50128078f,  3.28133345f,  -0.90272337f, 2.17840219f,  -3.85390043f, 1.04400611f,  -1.51806784f,
         -2.02946448f, 3.88446283f,  -3.09743977f, 2.66113234f,  0.58747929f,  -2.70279217f, 2.47248268f,
         -1.27001810f, 4.98003912f,  3.54352379f,  -2.06440759f, -0.00762612f, -0.30980289f, -1.00793648f,
         1.78032136f,  -0.36407304f, 2.22898149f,  -0.33799464f, 0.30464977f,  2.80208588f,  -0.85281968f,
         4.37704372f,  -0.91642666f, 1.32295370f,  2.13909531f,  2.39914703f,  4.52245998f,  1.92935705f,
         0.99381840f,  -0.23142999f, -0.32283169f, -0.80397993f, 2.60288763f,  2.76461053f,  1.77400625f,
         2.02602172f,  -2.06599736f, -3.95538950f, -1.60591578f, -6.31166506f, 4.22680283f,  -3.97488689f,
         2.82776928f,  2.70439839f,  0.25900796f,  3.46565676f,  0.00923875f,  3.00993395f,  -3.54275608f,
         1.68416595f,  -5.52749825f, 3.66339636f,  0.62231672f,  0.49025378f,  2.60456181f,  -1.08307147f,
         -3.33321953f, 2.78648925f,  -3.86621356f, 7.30088854f,  1.98787987f,  3.77585459f,  2.90812898f,
         -0.29400343f, 0.83474034f,  1.32195091f,  3.18010521f,  1.24589455f,  -0.30906326f, -0.65261662f,
         -3.12892914f, -0.51569617f, -4.44491005f, -0.80097729f, 0.30531061f,  4.41142941f,  -1.80266809f,
         2.21251750f,  2.00766182f,  -1.20997047f, 2.61106396f,  1.02403927f,  -0.53221637f, -0.54510701f,
         -2.10067749f, 1.70045757f,  2.52035522f,  0.76437557f,  1.17663324f,  3.08075643f,  -2.97668600f,
         0.60305959f,  0.80618089f,  1.86772847f,  -1.66303468f, 2.42022991f,  -0.82957172f, -1.54041612f,
         3.21708703f,  -1.04228044f, 2.12360740f,  0.85723335f,  2.02134609f,  1.28497660f,  1.55131006f,
         1.09578872f,  -2.02734756f, -1.50798774f, 1.63763690f,  -1.96294093f, 0.71943015f,  3.49259615f,
         -1.46452296f, 4.65465593f,  -0.82723278f, -0.04392552f, -3.05459404f, 2.32351446f,  -5.63220453f,
         -0.99349862f, -1.55407608f, -0.55988765f, -0.51809114f, -0.85369295f, 0.88097763f,  2.04635453f,
         0.13785851f,  5.93505192f,  0.55170661f,  -2.61648273f, 2.52562594f,  -3.05214500f, 0.87410510f,
         2.84636784f,  1.98483109f,  1.77460253f,  4.94668674f,  -0.68056989f, 5.24800491f,  -0.46104658f,
         -0.56622535f, -0.53243929f, -3.56289864f, -0.03714503f, -2.36219978f, -1.06232810f, -1.00614190f,
         -0.51112700f, 0.51696676f,  1.11392629f,  -0.81975353f, 4.76641846f,  -1.58373427f, 2.94765759f,
         -0.12692720f, 0.86798787f,  3.46557307f,  -0.68706006f, 4.87450695f,  0.47746658f,  1.05288696f,
         -0.45139998f, -0.35919702f, -2.25525904f, -1.86702085f, 2.34149599f,  -1.07225323f, 6.21087360f,
         -0.84906971f, 6.94897127f,  2.99065161f,  5.60922432f,  -0.19292337f, 0.12165236f,  4.45734978f,
         -3.96630692f, 1.39540529f,  -0.49167573f, -0.50199747f, -1.69825506f, 3.22229910f,  0.46407574f,
         3.45782661f,  -0.43632483f, 4.21560526f,  -1.72699690f, 1.78810894f,  3.22471094f,  2.08300018f,
         0.14486417f,  2.81813216f,  0.94466984f,  1.49007964f,  1.03669763f,  -0.71876121f, -0.51685190f,
         0.28573492f,  3.58906579f,  -2.34383154f, 3.84810257f,  -0.49746168f, -0.62450027f, 0.49661076f,
         0.80176342f,  2.82535124f,  2.81507540f,  2.60173821f,  -0.68244326f, 1.27675605f,  3.28125763f,
         -1.56876123f, 9.41231441f,  3.85805845f,  4.75027752f,  2.50572419f,  1.28180957f,  4.06122875f,
         -1.69917417f, 7.75314426f,  3.52355289f,  2.19955730f,  1.73140562f,  -2.72315478f, 0.58247280f,
         -5.66675425f, 2.75545073f,  -0.61709297f, -0.04514498f, 1.51967835f,  -2.72686076f, -0.19748497f,
         -1.35065222f, -1.52856445f, -3.86198258f, 2.56883621f,  -0.58629549f, -0.30647904f, 4.91377640f,
         -2.43626595f, 1.04606318f,  0.98978931f,  -0.37671798f, 4.09922600f,  0.73089349f,  4.16362572f,
         -0.76946896f, 2.68730974f,  4.11309624f,  -3.21574855f, 4.95183706f,  -4.28943682f, -1.61499083f,
         -1.45081258f, 1.47826231f,  0.76966119f,  0.81133056f,  -1.66215205f, 0.24901283f,  -2.97149849f,
         -0.11216557f, 1.36748624f,  -2.32634926f, 1.35156596f,  2.58868194f,  -2.87838793f, 4.31999969f,
         0.51695824f,  -0.64729220f, -2.25448775f, -0.10322928f, -3.07785654f, -2.18575621f, -0.18167657f,
         -5.68265390f, -2.75448751f, -0.88184738f, -0.92785287f, 1.04928935f,  -0.59247768f, 0.59002686f,
         -3.25248790f, -1.25988209f, -3.34502339f, -4.12041807f, 2.33082652f,  -1.03418469f, 2.78752375f,
         0.70514745f,  2.58542252f,  -1.22712314f, 2.38373423f,  -0.31430370f, -1.29869127f, 2.55319238f,
         1.26212776f,  -1.23777926f, 3.13514304f,  2.82996035f,  2.15325546f,  1.19183791f,  2.73435473f,
         -1.86676455f, 4.49313879f,  1.16612589f,  1.12818086f,  2.89846063f,  -0.79982543f, 0.38815671f,
         -3.53931141f, -1.64358592f, -4.07860374f, -3.45395875f, 0.23393105f,  -4.62477589f, 3.39298177f,
         1.31435204f,  -3.32152987f, 3.49837494f,  -5.40281296f, 0.67272353f,  0.87986028f,  0.66876608f,
         6.33664799f,  0.73262143f,  -0.89154863f, -0.57507277f, 1.11088562f,  -5.00805235f, 2.31823945f,
         2.19727445f,  -0.12887526f, 3.12535548f,  3.31421924f,  -0.73533010f, 3.64449930f,  5.72789097f,
         1.36092544f,  9.35740376f,  4.35907602f,  -1.41083086f, 4.48775721f,  -3.12211418f, 2.47584105f,
         -2.34577918f, 4.60199022f,  -2.24166799f, 0.74093813f,  3.55622005f,  -6.37225914f, 4.24702120f,
         0.23667920f,  1.75946426f,  3.65114713f,  2.11217690f,  0.36632437f,  0.87714475f,  1.84130561f,
         1.40680528f,  1.21115589f,  6.71010637f,  1.06697333f,  1.08320272f,  1.42915130f,  -2.07551098f,
         0.87193966f,  -2.07431459f, 4.59627199f,  -2.29288721f, 1.73021173f,  -2.65584517f, -0.43173438f,
         -0.64760333f, 3.32931638f,  -0.44082278f, 5.64066553f,  -1.52077508f, 5.72914505f,  1.49023890f,
         -0.36428687f, 0.95052063f,  0.66719997f,  -2.94661903f, 4.98902750f,  2.45712233f,  2.30307198f,
         6.23949289f,  0.92667961f,  2.54689956f,  -2.99809670f, 2.41789818f,  -0.22537935f, 1.52120996f,
         3.66470075f,  -5.09569073f, 3.89627886f,  -6.61506414f, 0.38206470f,  -3.49345016f, -2.34555125f,
         -4.79316711f, -0.24584234f, -2.91898680f, 1.02555788f,  3.72416019f,  -3.39801836f, 2.50208998f,
         -0.58428514f, -3.53273177f, 3.75704503f,  -2.54838347f, 4.86008453f,  -5.32398033f, 8.84705353f,
         -4.32583380f, 4.46357965f,  -1.55088067f, 2.19859552f,  1.30239582f,  2.44519949f,  2.73826361f,
         0.43468201f,  1.74839187f,  -0.28636885f, -0.21449631f, 2.84633756f,  -2.12093592f, 2.09768009f,
         1.24589097f,  1.57257175f,  0.40306979f,  -0.00440824f, -1.41044915f, 0.27802634f,  0.75614303f,
         -1.32255769f, 3.83360291f,  -1.53057098f, 5.38627195f,  2.13750315f,  4.07933283f,  3.37026882f,
         -2.42120934f, 7.50322437f,  -2.37978077f, 2.42308044f,  3.27454519f,  2.84078884f,  0.53679651f,
         3.84856105f,  0.16409910f,  -1.44391596f, 1.17583644f,  3.97972965f,  -1.34696579f, 2.26125479f,
         -1.86779857f, -0.12964523f, 2.65523076f,  0.03177297f,  6.66691351f,  -1.67869496f, 7.06167078f,
         4.80398035f,  5.22412825f,  1.34205806f,  1.23491740f,  -2.47908211f, -3.87420487f, 0.26133448f,
         2.62374115f,  1.94933975f,  7.42345333f,  0.52292359f,  6.99625301f,  0.39230359f,  6.37587357f,
         -2.94989204f, 5.86318779f,  -5.96664476f, 7.26113701f,  -1.31731248f, 1.53893590f,  2.30857229f,
         -1.44005263f, -0.74845022f, -1.37250018f, 0.21673965f,  -1.92625737f, 0.07720304f,  3.70314574f,
         1.61063421f,  2.33963299f,  -0.34364104f, 3.99103904f,  -1.38456416f, 3.75748348f,  -0.40145433f,
         1.91119027f,  0.15553260f,  0.60645616f,  3.48951149f,  -4.03585672f, -0.76474226f, -1.11259747f,
         -1.03603375f, 2.43516612f,  0.20641556f,  3.30384445f,  0.48420370f,  0.91145408f,  -0.86253095f,
         -0.65463471f, -0.92578071f, 0.68438685f,  4.29200459f,  -0.08150956f, 0.22094542f,  -0.18306947f,
         2.69922733f,  0.32101983f,  3.09214401f,  -1.64804518f, 4.93723917f,  -1.23343027f, 1.68790925f,
         -1.88417625f, -3.46729875f, -4.80928516f, -3.88981247f, -2.24656224f, -0.45018762f, 1.26963830f,
         -1.92359257f, 1.32501256f,  -5.09644985f, -0.67172617f, -3.10124111f, -3.59783268f, -4.18507290f,
         -2.59578967f, -3.07823253f, -1.24628472f, 0.89520305f,  -5.26159382f, 1.55842972f,  -1.21157169f,
         -0.08089194f, 0.55897939f,  3.93905234f,  -2.07300115f, 2.57561445f,  0.84461868f,  -0.94251084f,
         1.95271671f,  -0.44854814f, 0.71979952f,  -1.60090256f, -1.91269195f, 0.13136628f,  -1.14125764f,
         0.66804528f,  -3.49120283f, 0.50337470f,  -3.35998917f, 1.67515612f,  2.38808346f,  2.62827778f,
         1.51426780f,  -0.21528447f, 1.04319215f,  -0.71924853f, -1.60394681f, 1.79318678f,  -4.17813206f,
         -1.83091033f, -1.53848517f, -2.93892026f, -3.67382550f, -2.90911794f, -1.75921893f, -2.11628890f,
         -2.85766768f, 1.75146818f,  -3.69157457f, -0.45187509f, 2.64592862f,  1.71017528f,  0.40626037f,
         4.40675783f,  1.21687329f,  2.48120618f,  1.49812376f,  4.48013496f,  1.96704674f,  -0.94705671f,
         2.11740589f,  -1.95300794f, -0.09265548f, 1.82599163f,  -3.53230524f, 5.63902760f,  0.77569848f,
         4.07875872f,  1.26657581f,  1.58090401f,  -1.04151273f, 0.76378548f,  -0.44039226f, -1.65847731f,
         0.23016292f,  1.30441570f,  -0.84908098f, -0.19051600f, -1.23164546f, -0.11010855f, -2.54324627f,
         4.74622917f,  -2.76842260f, 3.04188061f,  6.05935669f,  0.33714068f,  -0.53215230f, 1.21172202f,
         1.05526388f,  2.57071042f,  3.27224278f,  0.22537431f,  1.64087236f,  -2.16304183f, 0.67665178f,
         -2.08081388f, 2.39165902f,  1.42149377f,  -1.94498539f, -0.15793994f, -2.52406406f, -1.09218001f,
         -1.42434406f, -1.13693166f, -0.90806276f, -1.35520852f, -0.89114177f, -0.27586940f, -1.47392857f,
         3.71026659f,  -1.13324571f, 6.73871803f,  -1.30732048f, 2.85595202f,  2.35288811f,  -0.06812286f,
         1.98389220f,  0.19481635f,  1.36196148f,  2.07928705f,  2.28805685f,  2.99274206f,  1.91889942f,
         3.73740888f,  3.55322433f,  -0.91521329f, 1.00105727f,  -0.11956118f, -0.74281299f, -0.71994114f,
         -1.25141609f, 3.51414847f,  -2.95959139f, 3.24749756f,  -1.96460819f, 3.99397016f,  0.89362502f,
         1.77934968f,  3.10791302f,  -1.57731533f, 1.08992982f,  3.61010027f,  -1.46781754f, 1.86637092f,
         0.79423213f,  0.20357013f,  0.18928665f,  -0.64325243f, -1.45625329f, 2.99198794f,  -0.97209716f,
         3.32580304f,  0.93858469f,  3.35336661f,  -2.11049938f, 7.07799101f,  -2.09264421f, 2.63290000f,
         -0.14904469f, 3.02488089f,  -1.44840527f, 5.05596161f,  2.71256614f,  4.54392290f,  -0.18147278f,
         0.40836972f,  -0.97409832f, 1.22351289f,  -2.74947691f, 4.92297602f,  0.34785542f,  6.87182331f,
         0.10613492f,  0.27995783f,  -0.41120863f, 1.23793423f,  -0.70981729f, -1.40435553f, 1.35703075f,
         -3.14161849f, 0.00806129f,  2.41883063f,  -0.73568642f, 1.78956199f,  2.48887062f,  -1.87124658f,
         1.31160164f,  0.10052288f,  -4.04556513f, 3.89579535f,  -0.25866473f, 2.10268784f,  0.54813278f,
         0.97756505f,  2.33083582f,  0.48331344f,  1.59691548f,  2.84565711f,  -1.14739466f, 2.62202072f,
         -1.71579432f, 3.25150108f,  2.53106308f,  1.80714715f,  1.83954227f,  -1.55849314f, 0.25002933f,
         2.44392824f,  -2.42426157f, 4.76816225f,  0.97849786f,  1.95124662f,  3.13304424f,  -1.53952873f,
         3.82290244f,  0.04378355f,  0.24694537f,  3.78889370f,  1.31366324f,  -0.53445184f, 1.60840607f,
         1.40792680f,  -1.31759822f, -0.75849378f, 1.43619704f,  -1.78857040f, -3.12328267f, -0.21848798f,
         -0.22284007f, 2.53751612f,  0.06193513f,  3.17334557f,  3.47897840f,  -0.95583671f, 2.37764120f,
         -0.04311061f, -1.13762045f, -4.34118128f, 3.52375007f,  -4.65443897f, 5.40035057f,  0.10900342f,
         2.88155174f,  4.34761238f,  -0.66580772f, 1.77895868f,  -0.46430019f, 1.33659208f,  -3.87676907f,
         4.86844206f,  -2.51048279f, 3.78174472f,  2.50960112f,  0.62631512f,  -1.48797607f, 1.54579568f,
         -2.55178070f, -1.25191855f, 1.63021159f,  -2.84022045f, -1.65325987f, -0.18720627f, 0.41033754f,
         -2.76701403f, 2.45559692f,  -3.20244360f, -0.34140706f, -1.56772137f, -0.52985138f, 1.84886646f,
         -3.11865759f, -1.29521751f, 2.89618325f,  -1.57364964f, 1.58078098f,  -1.06757283f, -1.31071854f,
         1.83744681f,  -6.41835594f, 1.55159092f,  -2.38235521f, -1.32500398f, -3.44755125f, 1.41241229f,
         -5.85707188f, 2.92639542f,  -2.45455074f, 2.18894148f,  -2.90235519f, 3.44608235f,  -0.73017585f,
         0.69557697f,  -2.35140157f, -3.25743055f, 0.00408101f,  -1.04894626f, 3.93254709f,  0.15747873f,
         4.26778603f,  -1.40763998f, 1.88030314f,  3.56710029f,  -3.33477879f, 1.32685399f,  -3.62404513f,
         -1.80059254f, -3.39359188f, 0.11665065f,  1.77078927f,  -0.21022639f, 3.62710619f,  -2.23914242f,
         3.50252628f,  -2.57065105f, 1.77723086f,  -2.90179086f, 1.18325305f,  -2.25456548f, 0.58041370f,
         -1.75020504f, 2.85054398f,  -3.29892063f, 5.57361221f,  -1.56906581f, 5.56049156f,  -1.32058680f,
         4.56358576f,  3.94858742f,  1.01180983f,  1.49214303f,  0.41054320f,  -2.08273339f, 0.42599005f,
         0.47346768f,  -1.57389641f, 3.07698727f,  -1.51253927f, -1.48645735f, -0.23779047f, -1.15742815f,
         -4.64732075f, 4.72518635f,  -1.06655097f, 4.12859011f,  -1.31923044f, 1.11321735f,  1.97395074f,
         1.30312407f,  3.19384813f,  -1.87027371f, 1.95319867f,  1.03104186f,  -0.01473618f, 6.97375488f,
         0.00601494f,  3.68166161f,  3.49324846f,  2.77545285f,  1.39260459f,  1.17709875f,  -0.83246660f,
         -2.49445462f, -2.60882902f, 0.93630409f,  -2.53302240f, -2.32854223f, 0.79721391f,  1.97388887f,
         0.94940281f,  2.99273109f,  3.10509706f,  -0.36287475f, 0.75319576f,  -0.78976685f, -2.92005682f,
         4.74376535f,  0.07984865f,  2.49105906f,  -0.73470640f, 1.19504333f,  -2.72909355f, -1.14577556f,
         -0.91538036f, 0.46066147f,  -0.01829237f, 1.10657585f,  -0.61527383f, 1.24755621f,  -3.51518178f,
         2.96183729f,  -2.63029003f, 0.41475648f,  -0.21229708f, 1.23330688f,  1.51130962f,  3.38699579f,
         0.48074529f,  3.03461289f,  -2.18941617f, -1.64126337f, -0.21963894f, -3.59941530f, 4.79890966f,
         -1.93569088f, 5.09483051f,  -1.92877293f, 5.43708944f,  1.01301849f,  5.98991632f,  1.46246362f,
         2.18598914f,  0.27719933f,  0.61382431f,  0.50982606f,  1.60895967f,  0.81997985f,  0.35023186f,
         0.62198722f,  -0.49569237f, 0.01365757f,  1.50323224f,  -1.94238460f, 4.36762905f,  -2.53150916f,
         3.02090168f,  1.77294421f,  3.96695471f,  2.86457849f,  3.48836088f,  -0.35922062f, 0.86888671f,
         -1.01417041f, 3.13454247f,  -2.73006344f, 2.85414290f,  -0.79936498f, -0.74460888f, 1.11478472f,
         -0.93073875f, -1.68111801f, 0.62053579f,  0.97649419f,  2.50766397f,  2.65244436f,  -0.40923917f,
         4.23108530f,  -2.58068967f, 2.76489401f,  0.45716035f,  -2.60567927f, 5.38766766f,  0.29117453f,
         -1.74118853f, 3.45414424f,  -1.37885427f, -1.78277636f, 3.34848475f,  -4.27803516f, 1.93966556f,
         -1.43013191f, -0.10290492f});

    // Use higher tolerance for diverse test data due to accumulated rounding errors
    test_case.run_with_tolerance_as_fp(1e-4f);
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_conv_transpose_autopad) {
    const auto model = convert_model("conv_transpose_1x3x8x8.onnx");
    auto test_case = test::TestCase(model, s_device);

    test_case.add_input<float>(
        Shape{1, 3, 8, 8},
        {0.625151098f, 0.596945405f, 0.139096975f,  0.276130885f,    0.00713499123f, 0.0607993789f, 0.0772799626f,
         0.098484844f, 1.34567487f,  1.26065361f,   2.18789816f,     0.0653471872f,  1.77078462f,   1.35865974f,
         1.12412751f,  1.16306293f,  1.52882469f,   1.59743035f,     0.0923668891f,  1.24418771f,   0.447759897f,
         0.206789434f, 0.263984501f, 0.38207826f,   1.03937542f,     2.09021282f,    0.467289686f,  1.86369658f,
         1.23900259f,  0.384096682f, 0.296756178f,  2.11388206f,     1.25714934f,    0.192125842f,  0.486225724f,
         1.38173127f,  0.213047206f, 0.124264345f,  0.187946454f,    1.44937944f,    0.385222673f,  0.558024764f,
         1.00838554f,  0.124311149f, 1.90255845f,   1.11999547f,     0.974563539f,   0.582694709f,  0.531468391f,
         3.04861355f,  0.285169095f, 1.22178805f,   0.128970414f,    0.566250026f,   0.207558841f,  0.828584075f,
         0.89728272f,  0.458865613f, 0.844876111f,  1.37122619f,     1.16179717f,    0.920694292f,  1.14122105f,
         1.79164004f,  1.12682891f,  0.842717826f,  1.87564576f,     0.692874193f,   0.854609072f,  0.118006214f,
         1.14124298f,  1.92377281f,  0.0637631491f, 0.223139569f,    0.888881445f,   0.53058058f,   0.852253318f,
         1.32553041f,  0.644229054f, 0.412511975f,  0.000415288785f, 1.32483172f,    0.0566556714f, 0.104107559f,
         0.204130545f, 0.557883739f, 0.928778589f,  2.15578008f,     1.29505754f,    1.46797705f,   1.22587049f,
         0.174201131f, 0.917162478f, 0.479762316f,  0.0272906683f,   0.58244282f,    0.104637481f,  0.748023987f,
         0.389915824f, 0.398318321f, 0.0204038732f, 0.816221416f,    1.15234113f,    0.56337899f,   1.17305529f,
         0.882097483f, 0.985680401f, 1.11193633f,   0.781882703f,    0.922793448f,   1.66662455f,   0.944629848f,
         0.172961116f, 0.547528088f, 1.73304665f,   1.14912224f,     1.02735782f,    0.89158535f,   1.65481675f,
         0.572695494f, 1.31630123f,  0.0261730403f, 0.594345629f,    0.933356583f,   0.238472551f,  0.299513876f,
         0.158894345f, 0.197219521f, 1.52590966f,   1.35690475f,     0.141676888f,   2.15699768f,   1.94352877f,
         1.40682673f,  1.04016232f,  0.0508034416f, 0.224050492f,    2.41353512f,    0.352399886f,  1.54169703f,
         0.575035036f, 0.148580253f, 0.541164219f,  0.982789278f,    0.893933475f,   0.430977315f,  0.426253587f,
         1.35645676f,  1.1283102f,   1.80685508f,   1.7404393f,      0.435301125f,   0.270964503f,  1.02027822f,
         2.35678649f,  1.50359774f,  0.175944671f,  1.42923343f,     0.51326853f,    1.76939189f,   0.409514606f,
         0.098843053f, 1.47446668f,  1.30807877f,   0.258639991f,    0.16241923f,    1.01321244f,   0.661336064f,
         1.56858552f,  0.324210107f, 0.276222378f,  0.283114731f,    0.528313696f,   0.657391787f,  0.718641877f,
         0.193789124f, 0.694960773f, 1.91718662f,   0.33368668f,     1.39595342f,    1.22667146f,   2.16012883f,
         0.930284202f, 0.264723539f, 0.833213866f,  0.346019775f,    0.542626441f,   0.751732051f,  0.824235022f,
         0.240106314f, 0.243192792f, 1.26579654f});

    test_case.add_expected_output<float>(
        Shape{1, 1, 16, 16},
        {2.6570032f,   1.4026494f,   4.097457f,    1.4312422f,   4.358069f,    1.9452412f,   3.8355894f,
         0.14656305f,  3.5327482f,   -0.09384227f, 3.1709785f,   0.2848494f,   3.059483f,    0.88048124f,
         4.6876864f,   1.9655691f,   -1.4274524f,  -1.9585024f,  -1.2689599f,  -1.4720103f,  -0.6884266f,
         -0.11167741f, -0.89471614f, -2.882082f,   -1.8259774f,  -2.6774187f,  -0.4556129f,  -1.2079989f,
         -1.0779876f,  -1.2382134f,  -0.18473399f, 0.00806427f,  2.5595012f,   0.91405666f,  0.6244595f,
         -0.68860304f, 4.978584f,    3.3394523f,   -0.48534f,    -2.4429312f,  5.646074f,    1.5901803f,
         4.397019f,    2.4571092f,   5.218195f,    1.0488592f,   3.2577524f,   0.9674945f,   0.43771517f,
         1.2669424f,   -2.5042286f,  -2.7755532f,  -1.9331845f,  0.69382787f,  -1.3249675f,  -1.7260445f,
         -1.0543696f,  0.21531725f,  -1.0718178f,  0.55455464f,  -0.31781197f, 0.25291705f,  -0.90415967f,
         -0.37044144f, 0.8667301f,   2.4778519f,   0.78733087f,  1.1406101f,   0.06660879f,  -0.6937479f,
         -0.79506147f, 0.44638246f,  0.70710737f,  -0.68043125f, -1.4824747f,  -1.430282f,   -0.9336823f,
         -0.91168797f, 2.0403967f,   0.47893757f,  -0.29100597f, 0.15847456f,  -1.8225821f,  0.11050332f,
         0.5951805f,   0.59405357f,  -0.8604816f,  -0.8099879f,  -1.055466f,   -0.67057717f, -1.4942209f,
         -2.1827173f,  -1.7207037f,  -2.299592f,   -1.2838408f,  -0.76404536f, 1.3764476f,   1.8355037f,
         2.890913f,    2.3728309f,   0.17107165f,  -0.0519259f,  3.8047943f,   1.4985598f,   2.3641415f,
         2.1788077f,   2.6996064f,   -1.3380355f,  3.250296f,    -1.0669993f,  4.1230874f,   1.3102139f,
         -0.46833718f, 0.27577424f,  -1.8712362f,  -0.9149951f,  -2.8570867f,  -3.5502553f,  -1.6344188f,
         -0.9639236f,  -1.3047136f,  0.7507315f,   -0.86908996f, -1.410333f,   0.27828562f,  0.5098876f,
         -2.219245f,   -1.6510864f,  1.9861047f,   1.5123394f,   -2.5923595f,  -1.1425136f,  -0.95536673f,
         -2.388658f,   0.80486345f,  0.46455407f,  -0.35799634f, 0.14485347f,  -0.60110354f, -0.24908698f,
         2.7458048f,   0.47857296f,  2.2998524f,   0.5031518f,   0.2554748f,   0.8855421f,   -0.02953649f,
         0.7384027f,   -0.51014626f, -1.4048489f,  -1.4614314f,  -0.8871366f,  0.18441951f,  0.9485445f,
         0.3192644f,   0.5616672f,   -0.4762156f,  -1.1695982f,  -0.38912475f, 0.17468858f,  0.95902777f,
         0.0249002f,   3.191238f,    1.4911141f,   2.3547568f,   1.2146041f,   1.6097362f,   -0.46672237f,
         3.1703544f,   4.050436f,    3.7028737f,   2.091999f,    3.748374f,    0.8613107f,   1.7032791f,
         0.81649125f,  -1.3934654f,  -2.1264868f,  -0.11187541f, 0.33751798f,  -0.17415702f, 0.46710902f,
         0.01545f,     0.14659834f,  -0.41714227f, 0.3795176f,   -1.2214614f,  -0.14365876f, -1.4849302f,
         -0.7912582f,  0.1787306f,   0.53471124f,  2.2546692f,   -0.8649975f,  3.3309882f,   3.8740916f,
         1.1473688f,   0.5549442f,   3.116909f,    1.325516f,    0.17125642f,  -1.0625385f,  0.02054965f,
         -1.0995616f,  2.2210755f,   -1.0817534f,  1.3485643f,   1.5398469f,   0.2551086f,   0.14614928f,
         -2.8864846f,  -1.6558629f,  -2.1308398f,  -0.3219006f,  -1.1134721f,  -1.5906883f,  -1.253935f,
         -1.4817764f,  -1.8708516f,  -2.9484243f,  -1.5679508f,  -1.3551403f,  0.5339719f,   0.7206881f,
         2.5979834f,   1.5220752f,   -0.69305575f, -1.5217942f,  0.13409078f,  0.66243625f,  0.45986307f,
         0.5982121f,   1.7192272f,   0.666829f,    1.8630754f,   -0.16500485f, 2.9745436f,   0.8539709f,
         0.14285994f,  2.3918753f,   -0.9979347f,  -0.7793964f,  0.68560743f,  0.8504962f,   -0.36467838f,
         0.21346569f,  -1.098107f,   -0.2560147f,  -0.6450561f,  0.02996862f,  -0.19393313f, 0.97208583f,
         0.11212575f,  1.1170075f,   -1.437528f,   -0.57256365f});

    test_case.run_with_tolerance_as_fp(0.00001f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_convtranspose_output_shape_with_batch_channels) {
    // Test ConvTranspose with output_shape containing batch and channel dimensions
    // Per ONNX spec, output_shape should only contain spatial dimensions [H, W]
    // However, some models incorrectly include all dimensions [N, C, H, W]
    // This test verifies that OpenVINO gracefully handles this by extracting spatial dims
    auto model = convert_model("convtranspose_output_shape_with_batch_channels.onnx");

    // Verify model loaded correctly and output shape is correct
    // Expected output: [1, 2, 10, 8] from input [1, 1, 3, 3]
    ASSERT_EQ(model->get_output_shape(0), (ov::Shape{1, 2, 10, 8}));

    auto test_case = ov::test::TestCase(model, s_device);

    // Input data: [1,1,3,3]
    test_case.add_input<float>({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});

    // Filters: [1,2,3,3]
    test_case.add_input<float>(
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f});

    // Expected output shape: [1, 2, 10, 8]
    // This test verifies that the conversion succeeded despite invalid output_shape format
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_convtranspose_output_shape_with_batch_channels_stride2) {
    // Test ConvTranspose with output_shape=[1,2,6,6] (includes batch/channels)
    // Input: [1,1,4,4], output_shape attribute: [1,2,6,6]
    // Should extract spatial dims: [6,6] and produce output [1,2,6,6]
    auto model = convert_model("convtranspose_output_shape_with_batch_channels_stride2.onnx");

    // Verify output shape matches expected: [1, 2, 6, 6]
    ASSERT_EQ(model->get_output_shape(0), (ov::Shape{1, 2, 6, 6}));

    auto test_case = ov::test::TestCase(model, s_device);

    // Input data: [1,1,4,4] - values 0 to 15
    std::vector<float> input_data(16);
    std::iota(input_data.begin(), input_data.end(), 0.0f);
    test_case.add_input<float>(input_data);

    // Filters: [1,2,3,3] - all 0.1f
    std::vector<float> filter_data(18, 0.1f);
    test_case.add_input<float>(filter_data);

    // Bias: [2] - all 0.0f
    std::vector<float> bias_data(2, 0.0f);
    test_case.add_input<float>(bias_data);

    // Expected output from ONNX Runtime: [1, 2, 6, 6]
    test_case.add_expected_output<float>(
        ov::Shape{1, 2, 6, 6},
        {1.00000000f, 0.60000002f, 1.40000010f, 0.80000001f, 1.79999995f, 1.00000000f, 0.89999998f, 0.50000000f,
         1.10000002f, 0.60000002f, 1.29999995f, 0.69999999f, 2.60000014f, 1.40000010f, 3.00000000f, 1.60000002f,
         3.40000010f, 1.79999995f, 1.70000005f, 0.90000004f, 1.90000010f, 1.00000000f, 2.09999990f, 1.10000002f,
         4.20000029f, 2.20000005f, 4.59999990f, 2.40000010f, 5.00000000f, 2.59999990f, 2.50000000f, 1.30000007f,
         2.70000005f, 1.39999998f, 2.90000010f, 1.50000000f, 1.00000000f, 0.60000002f, 1.40000010f, 0.80000001f,
         1.79999995f, 1.00000000f, 0.89999998f, 0.50000000f, 1.10000002f, 0.60000002f, 1.29999995f, 0.69999999f,
         2.60000014f, 1.40000010f, 3.00000000f, 1.60000002f, 3.40000010f, 1.79999995f, 1.70000005f, 0.90000004f,
         1.90000010f, 1.00000000f, 2.09999990f, 1.10000002f, 4.20000029f, 2.20000005f, 4.59999990f, 2.40000010f,
         5.00000000f, 2.59999990f, 2.50000000f, 1.30000007f, 2.70000005f, 1.39999998f, 2.90000010f, 1.50000000f});

    test_case.run();
}
