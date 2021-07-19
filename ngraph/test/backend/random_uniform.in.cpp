// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_float32_case1)
{
    auto out_shape = make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f32, element::Type_t::f32, 150, 10), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f32, Shape{3, 2, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{3, 2, 4}, vector<float>{0.7011235952377319336,0.3053963184356689453,0.9393105506896972656,0.9456034898757934570,
                                                 0.1169477701187133789,0.5077005624771118164,0.5197197198867797852,0.2272746562957763672,
                                                 0.9913740158081054688,0.3551903963088989258,0.8269231319427490234,0.5986485481262207031,
                                                 0.3136410713195800781,0.5748131275177001953,0.4139908552169799805,0.9630825519561767578,
                                                 0.3714079856872558594,0.8525316715240478516,0.0935858488082885742,0.0820095539093017578,
                                                 0.2365508079528808594,0.8105630874633789062,0.7422660589218139648,0.7610669136047363281});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_float16_case1)
{
    auto out_shape = make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f16, Shape{}, 0);
    auto max_val = make_shared<opset8::Constant>(element::f16, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f16, element::Type_t::f16, 150, 10), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f16, Shape{3, 2, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{3, 2, 4}, vector<float16>{0.6044921875,0.8066406250,0.8320312500,0.3837890625,
                                                                0.0361328125,0.0830078125,0.5439453125,0.8339843750,
                                                                0.3359375000,0.7197265625,0.1542968750,0.1289062500,
                                                                0.3476562500,0.8691406250,0.4130859375,0.5722656250,
                                                                0.5742187500,0.9394531250,0.6552734375,0.8222656250,
                                                                0.8242187500,0.1328125000,0.6435546875,0.6601562500});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_double)
{
    auto out_shape = make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f64, Shape{}, 0);
    auto max_val = make_shared<opset8::Constant>(element::f64, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f64, element::Type_t::f64, 150, 10), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f64, Shape{3, 2, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{3, 2, 4}, vector<double>{0.6089892188187620014616641128668561577796936035156250,0.5144852007657354509007063825265504419803619384765625,0.9355828939149166689759340442833490669727325439453125,0.1577579346062143450524217769270762801170349121093750,
                                                                  0.9309922128081142833622152465977706015110015869140625,0.6153855781977037864294288738165050745010375976562500,0.5091292384542769333677370013901963829994201660156250,0.3119275084940391629118039418244734406471252441406250,
                                                                  0.9712641199166545113996562577085569500923156738281250,0.7486870327629326915541696507716551423072814941406250,0.8924066383588371564883345854468643665313720703125000,0.9381286963095272213308817299548536539077758789062500,
                                                                  0.8092744319940818886749411831260658800601959228515625,0.0753459178107278582103845110395923256874084472656250,0.6800388020136303168783342698588967323303222656250000,0.6126767468248481840475960780167952179908752441406250,
                                                                  0.8341116087085584940297167122480459511280059814453125,0.0966951703057401523722091951640322804450988769531250,0.7800837835828973165774868903099559247493743896484375,0.4718041387914264639391603850526735186576843261718750,
                                                                  0.0911535576997615226702009749715216457843780517578125,0.7567904145507127200431796154589392244815826416015625,0.3918933695224615654240096773719415068626403808593750,0.7331953749194290992363676195964217185974121093750000});

    test_case.run();
}


NGRAPH_TEST(${BACKEND_NAME}, random_uniform_float32_case2)
{
    auto out_shape = make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, -20);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 150);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f32, element::Type_t::f32, 150, 10), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f32, Shape{3, 2, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{3, 2, 4}, vector<float>{99.1910095214843750000,31.9173736572265625000,139.6828002929687500000,140.7525939941406250000,
                                                                -0.1188793182373046875,66.3090972900390625000,68.3523559570312500000,18.6366920471191406250,
                                                                148.5335845947265625000,40.3823661804199218750,120.5769348144531250000,81.7702560424804687500,
                                                                33.3189811706542968750,77.7182312011718750000,50.3784484863281250000,143.7240295410156250000,
                                                                43.1393585205078125000,124.9303894042968750000,-4.0904054641723632812,-6.0583763122558593750,
                                                                20.2136383056640625000,117.7957305908203125000,106.1852264404296875000,109.3813781738281250000,});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_float16_case2)
{
    auto out_shape = make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f16, Shape{}, -1.5);
    auto max_val = make_shared<opset8::Constant>(element::f16, Shape{}, -1.0);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f16, element::Type_t::f16, 150, 10), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f16, Shape{3, 2, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{3, 2, 4}, vector<float16>{-1.1972656250,-1.0966796875,-1.0839843750,-1.3085937500,
                                                                  -1.4824218750,-1.4589843750,-1.2285156250,-1.0830078125,
                                                                  -1.3320312500,-1.1406250000,-1.4228515625,-1.4355468750,
                                                                  -1.3261718750,-1.0654296875,-1.2929687500,-1.2138671875,
                                                                  -1.2128906250,-1.0302734375,-1.1718750000,-1.0888671875,
                                                                  -1.0878906250,-1.4335937500,-1.1777343750,-1.1699218750});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_double_min_max)
{
    auto out_shape = make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f64, Shape{}, -10);
    auto max_val = make_shared<opset8::Constant>(element::f64, Shape{}, 15);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f64, element::Type_t::f64, 150, 10), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f64, Shape{3, 2, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{3, 2, 4}, vector<double>{5.2247304704690495924523929716087877750396728515625000,2.8621300191433860504730546381324529647827148437500000,13.3895723478729173905321658821776509284973144531250000,-6.0560516348446409296002457267604768276214599609375000,
                                                                 13.2748053202028586383676156401634216308593750000000000,5.3846394549425937725573021452873945236206054687500000,2.7282309613569228901042151846922934055328369140625000,-2.2018122876490213712941113044507801532745361328125000,
                                                                 14.2816029979163623409021965926513075828552246093750000,8.7171758190733186211218708194792270660400390625000000,12.3101659589709271358515252359211444854736328125000000,13.4532174077381796450936235487461090087890625000000000,
                                                                 10.2318607998520469948289246531203389167785644531250000,-8.1163520547318039888295970740728080272674560546875000,7.0009700503407579219583567464724183082580566406250000,5.3169186706212041571006921003572642803192138671875000,
                                                                 10.8527902177139630168767325812950730323791503906250000,-7.5826207423564966347839799709618091583251953125000000,9.5020945895724331364817771827802062034606933593750000,1.7951034697856620425682194763794541358947753906250000,
                                                                 -7.7211610575059621552895805507432669401168823242187500,8.9197603637678177790348854614421725273132324218750000,-0.2026657619384604203105482156388461589813232421875000,8.3298843729857274809091904899105429649353027343750000});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_int32)
{
    auto out_shape = make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{2, 3, 4});
    auto min_val = make_shared<opset8::Constant>(element::i32, Shape{}, -100);
    auto max_val = make_shared<opset8::Constant>(element::i32, Shape{}, 50);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::i32, element::Type_t::i32, 100, 350), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::i32, Shape{2, 3, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{2, 3, 4}, vector<int32_t>{22,-56,-33,-89,-98,-33,-3,-48,-82,5,-66,21,
                                                                  29,-42,-73,-37,3,36,-35,20,-11,-8,-78,47,});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_int64)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{3}, vector<int32_t>{5, 4, 3});
    auto min_val = make_shared<opset8::Constant>(element::i64, Shape{}, -2600);
    auto max_val = make_shared<opset8::Constant>(element::i64, Shape{}, 3700);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::i64, element::Type_t::i64, 755, 951), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::i64, Shape{5, 4, 3});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{5, 4, 3}, vector<int64_t>{2116,-1581,2559,-339,-1660,519,90,2027,-210,3330,
                                                                  1831,-1737,2683,2661,3473,1220,3534,-2384,2199,1935,
                                                                  499,2861,2743,3223,-531,-836,-65,3435,632,1765,
                                                                  2613,1891,1698,3069,169,-792,-32,2976,-1552,-2588,
                                                                  3327,-1756,2637,-1084,3567,-778,-1465,2967,1242,2672,
                                                                  -1585,-2271,3536,-1502,400,2241,3126,908,1073,-2110});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_f32_to_f16)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{3}, vector<int32_t>{4, 3, 5});
    auto min_val = make_shared<opset8::Constant>(element::f16, Shape{}, -10);
    auto max_val = make_shared<opset8::Constant>(element::f16, Shape{}, 20);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f32, element::Type_t::f16, 1976, 3461), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f16, Shape{4, 3, 5});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{4, 3, 5}, vector<float16>{-7.0667686462,0.7613515854,15.2241306305,-7.0545663834,-8.1617279053,
                                                                  -5.6791114807,13.1542701721,4.1214733124,-5.1889219284,1.7976608276,
                                                                  -7.0543441772,14.0094909668,12.3061199188,5.8478431702,11.7749271393,
                                                                  -9.2398548126,-9.1900157928,7.6471080780,1.7071838379,-0.5436820984,
                                                                  4.7661933899,-0.4901742935,1.5420894623,6.2070121765,-6.2156248093,
                                                                  10.6620826721,8.8282279968,15.9779663086,9.0587406158,17.5764923096,
                                                                  -5.7205781937,-4.4252395630,10.3589363098,1.2601642609,5.0108146667,
                                                                  3.7169313431,14.7530841827,8.7054824829,16.7723121643,0.4061698914,
                                                                  -3.6634311676,-4.8043107986,2.8337402344,0.1496973038,7.6015071869,
                                                                  -1.1772308350,7.7872123718,14.3392868042,-7.6612958908,0.5786533356,
                                                                  16.3981590271,-0.7194814682,-8.2340545654,-9.6088333130,8.2351722717,
                                                                  1.7502002716,16.9785060883,-5.7213220596,-1.6345300674,-7.0849504471});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_f16_to_bf16)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{3}, vector<int32_t>{4, 3, 5});
    auto min_val = make_shared<opset8::Constant>(element::bf16, Shape{}, -5);
    auto max_val = make_shared<opset8::Constant>(element::bf16, Shape{}, 5);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f16, element::Type_t::bf16, 1976, 3461), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::bf16, Shape{4, 3, 5});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{4, 3, 5}, bfloat16::from_float_vector(vector<float>{4.679687500,0.664062500,3.687500000,-2.001953125,4.703125000,
                                                                   3.906250000,1.593750000,-3.964843750,2.453125000,0.476562500,
                                                                   -1.396484375,-3.085937500,-4.421875000,0.175781250,-4.929687500,
                                                                   0.703125000,-3.203125000,3.367187500,3.421875000,-2.949218750,
                                                                   -3.447265625,3.164062500,2.597656250,0.945312500,-1.132812500,
                                                                   -3.740234375,-1.386718750,2.167968750,-1.933593750,-2.792968750,
                                                                   0.671875000,-2.187500000,-1.533203125,2.753906250,4.531250000,
                                                                   1.367187500,-2.578125000,3.437500000,1.257812500,0.781250000,
                                                                   -1.943359375,2.695312500,-0.332031250,0.437500000,-1.152343750,
                                                                   -2.958984375,-4.054687500,-2.519531250,1.218750000,1.777343750,
                                                                   -0.429687500,-2.998046875,-2.792968750,3.140625000,-0.820312500,
                                                                   0.878906250,4.304687500,-1.357421875,-1.689453125,-4.968750000}));

    test_case.run();
}


NGRAPH_TEST(${BACKEND_NAME}, random_uniform_f32_to_f64)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{3}, vector<int32_t>{4, 3, 5});
    auto min_val = make_shared<opset8::Constant>(element::f64, Shape{}, -1);
    auto max_val = make_shared<opset8::Constant>(element::f64, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f32, element::Type_t::f64, 1976, 3461), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f64, Shape{4, 3, 5});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{4, 3, 5}, vector<double>{-0.8044512271881103516,-0.2825765609741210938,0.6816086769104003906,-0.8036377429962158203,
                                                                 -0.8774485588073730469,-0.7119407653808593750,0.5436179637908935547,-0.0585684776306152344,
                                                                 -0.6792614459991455078,-0.2134892940521240234,-0.8036229610443115234,0.6006326675415039062,
                                                                 0.4870746135711669922,0.0565228462219238281,0.4516618251800537109,-0.9493236541748046875,
                                                                 -0.9460010528564453125,0.1764738559722900391,-0.2195210456848144531,-0.3695788383483886719,
                                                                 -0.0155870914459228516,-0.3660116195678710938,-0.2305274009704589844,0.0804674625396728516,
                                                                 -0.7477083206176757812,0.3774721622467041016,0.2552151679992675781,0.7318644523620605469,
                                                                 0.2705826759338378906,0.8384327888488769531,-0.7147052288055419922,-0.6283493041992187500,
                                                                 0.3572623729705810547,-0.2493224143981933594,0.0007209777832031250,-0.0855379104614257812,
                                                                 0.6502056121826171875,0.2470321655273437500,0.7848207950592041016,-0.3062553405761718750,
                                                                 -0.5775620937347412109,-0.6536207199096679688,-0.1444172859191894531,-0.3233535289764404297,
                                                                 0.1734337806701660156,-0.4118154048919677734,0.1858141422271728516,0.6226191520690917969,
                                                                 -0.8440864086151123047,-0.2947564125061035156,0.7598772048950195312,-0.3812987804412841797,
                                                                 -0.8822703361511230469,-0.9739222526550292969,0.2156782150268554688,-0.2166533470153808594,
                                                                 0.7985670566558837891,-0.7147548198699951172,-0.4423019886016845703,-0.8056633472442626953});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_f64_to_i32)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{3}, vector<int32_t>{4, 3, 5});
    auto min_val = make_shared<opset8::Constant>(element::i32, Shape{}, -100);
    auto max_val = make_shared<opset8::Constant>(element::i32, Shape{}, 100);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f64, element::Type_t::i32, 1976, 3461), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::i32, Shape{4, 3, 5});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{4, 3, 5}, vector<int32_t>{56,45,-2,-65,-43,57,90,61,-57,-76,88,-84,-98,-96,-84,-72,-14,-99,20,-72,
                                                                 38,-16,39,49,25,-92,-6,73,-61,-54,5,-65,-80,-11,71,-35,49,36,55,74,
                                                                 64,-88,3,-46,-61,27,15,-23,-4,-10,35,17,31,-82,-78,46,19,-41,2,75});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_f16_to_i64)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{3}, vector<int32_t>{4, 3, 5});
    auto min_val = make_shared<opset8::Constant>(element::i64, Shape{}, -200);
    auto max_val = make_shared<opset8::Constant>(element::i64, Shape{}, 50);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f16, element::Type_t::i64, 1976, 3461), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::i64, Shape{4, 3, 5});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{4, 3, 5}, vector<int64_t>{42,-58,17,-125,43,23,-35,-174,-14,-63,-110,-152,-186,-71,-198,-57,-155,9,11,-149,
                                                                  -161,4,-10,-51,-103,-169,-110,-21,-123,-145,-58,-130,-113,-6,38,-41,-140,11,-44,-56,
                                                                  -124,-8,-83,-64,-104,-149,-176,-138,-45,-31,-86,-150,-145,4,-96,-53,33,-109,-117,-199});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_i32_to_i64)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{3}, vector<int32_t>{4, 3, 5});
    auto min_val = make_shared<opset8::Constant>(element::i64, Shape{}, -1500);
    auto max_val = make_shared<opset8::Constant>(element::i64, Shape{}, 50);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::i32, element::Type_t::i64, 1976, 3770), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::i64, Shape{4, 3, 5});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{4, 3, 5}, vector<int64_t>{-1269,-342,-164,-1154,-799,-820,-762,-288,-1065,-1327,-945,-969,-609,-1029,-1224,-1088,0,-727,-808,-101,
                                                                  -579,-284,-1000,-922,-684,-859,-1068,-668,-1079,-861,-607,-877,-197,-889,-441,-1115,-1425,-308,-1209,-1318,
                                                                  -142,32,-511,-1060,-1304,-1403,-93,-676,-675,-1389,-229,-376,-1111,-1218,-38,-447,-1031,-1147,-181,-1490});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_i64_to_i32)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{3}, vector<int32_t>{4, 3, 5});
    auto min_val = make_shared<opset8::Constant>(element::i32, Shape{}, -1500);
    auto max_val = make_shared<opset8::Constant>(element::i32, Shape{}, 50);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::i64, element::Type_t::i32, 1976, 3770), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::i32, Shape{4, 3, 5});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{4, 3, 5}, vector<int32_t>{-1101,-1198,-869,-1060,-807,-619,-1143,-72,-442,-854,7,-362,-848,-96,-135,-99,-791,-1381,-1493,-1287,
                                                                  -1020,-921,-792,-889,-169,-1375,-789,-600,-1293,-1071,-766,-621,-1064,-798,-174,-397,-277,-1153,-474,-830,
                                                                  -736,-172,-881,-1464,49,-495,-1355,-1280,-530,-714,-986,-1277,-1077,-37,-1110,-1018,-1165,-401,-188,-79});

    test_case.run();
}


NGRAPH_TEST(${BACKEND_NAME}, random_uniform_i32_to_f64)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{3}, vector<int32_t>{4, 3, 5});
    auto min_val = make_shared<opset8::Constant>(element::f64, Shape{}, -1500);
    auto max_val = make_shared<opset8::Constant>(element::f64, Shape{}, -1000);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::i32, element::Type_t::f64, 1976, 3770), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f64, Shape{4, 3, 5});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{4, 3, 5}, vector<double>{-1169,-1342,-1264,-1204,-1349,-1370,-1112,-1338,-1065,-1077,-1395,-1469,-1459,-1229,-1024,-1238,-1300,-1377,-1308,-1401,
                                                                 -1179,-1384,-1250,-1022,-1434,-1209,-1118,-1118,-1179,-1311,-1107,-1277,-1097,-1339,-1491,-1465,-1325,-1008,-1359,-1268,
                                                                 -1192,-1118,-1411,-1210,-1004,-1353,-1493,-1376,-1175,-1339,-1429,-1026,-1461,-1268,-1188,-1297,-1231,-1397,-1431,-1140});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_i32_to_bf16)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{3}, vector<int32_t>{4, 3, 5});
    auto min_val = make_shared<opset8::Constant>(element::bf16, Shape{}, 100);
    auto max_val = make_shared<opset8::Constant>(element::bf16, Shape{}, 500);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::i32, element::Type_t::bf16, 1976, 3770), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::bf16, Shape{4, 3, 5});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{4, 3, 5}, bfloat16::from_float_vector(vector<float>{231,358,436,496,251,130,488,362,135,423,305,131,341,471,376,462,200,223,492,499,
                                                                                            121,316,350,378,366,191,382,282,421,389,393,323,403,361,109,435,175,392,141,132,
                                                                                            408,282,389,390,496,347,307,424,125,261,471,474,139,132,112,303,369,403,169,360}));

    test_case.run();
}