// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/shape.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_float32)
{
    auto out_shape = make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f32, 150, 10), ParameterVector{});

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

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_float16)
{
    auto out_shape = make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f16, Shape{}, 0);
    auto max_val = make_shared<opset8::Constant>(element::f16, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f16, 150, 10), ParameterVector{});

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
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f64, 150, 10), ParameterVector{});

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


NGRAPH_TEST(${BACKEND_NAME}, random_uniform_float32_min_max)
{
    auto out_shape = make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, -20);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 150);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f32, 150, 10), ParameterVector{});

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

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_float16_min_max)
{
    auto out_shape = make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f16, Shape{}, -1.5);
    auto max_val = make_shared<opset8::Constant>(element::f16, Shape{}, -1.0);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f16, 150, 10), ParameterVector{});

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
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::f64, 150, 10), ParameterVector{});

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
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::i32, 100, 350), ParameterVector{});

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
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::i64, 755, 951), ParameterVector{});

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


NGRAPH_TEST(${BACKEND_NAME}, random_uniform_bfloat16)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{3}, vector<int32_t>{4, 2, 5});
    auto min_val = make_shared<opset8::Constant>(element::bf16, Shape{}, -5);
    auto max_val = make_shared<opset8::Constant>(element::bf16, Shape{}, 10);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::bf16, 234, 61), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::bf16, Shape{4, 2, 5});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{4, 2, 5}, vector<bfloat16>{-2.421875,-1.953125,4.9375,-0.78125,8.125,-2.0625,8.5,5.4375,6.125,6.5,
                                                                  9.0625,1.4375,0.875,0.96875,5.4375,0.15625,8,8.125,8.8125,4.5,
                                                                  -1.84375,9.1875,2.625,6.5,5.75,6.75,-3.125,1.09375,-2.421875,0.96875,
                                                                  5.5625,-3.9375,-1.484375,4.25,7.625,9.5,0.875,0.75,9.75,-0.1875});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_bfloat16_simple)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{1}, vector<int32_t>{8});
    auto min_val = make_shared<opset8::Constant>(element::bf16, Shape{}, -5);
    auto max_val = make_shared<opset8::Constant>(element::bf16, Shape{}, 10);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, min_val, max_val, element::Type_t::bf16, 234, 61), ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::bf16, Shape{8});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{8}, vector<bfloat16>{-2.421875,-1.953125,4.9375,-0.78125,8.125,-2.0625,8.5,5.4375});

    test_case.run();
}