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
    auto out_shape =
        make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, 0);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(
                                       out_shape, min_val, max_val, element::Type_t::f32, 150, 10),
                                   ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f32, Shape{3, 2, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(
        Shape{3, 2, 4},
        vector<float>{0.7011235952377319336, 0.3053963184356689453, 0.9393105506896972656,
                      0.9456034898757934570, 0.1169477701187133789, 0.5077005624771118164,
                      0.5197197198867797852, 0.2272746562957763672, 0.9913740158081054688,
                      0.3551903963088989258, 0.8269231319427490234, 0.5986485481262207031,
                      0.3136410713195800781, 0.5748131275177001953, 0.4139908552169799805,
                      0.9630825519561767578, 0.3714079856872558594, 0.8525316715240478516,
                      0.0935858488082885742, 0.0820095539093017578, 0.2365508079528808594,
                      0.8105630874633789062, 0.7422660589218139648, 0.7610669136047363281});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_float16_case1)
{
    auto out_shape =
        make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f16, Shape{}, 0);
    auto max_val = make_shared<opset8::Constant>(element::f16, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(
                                       out_shape, min_val, max_val, element::Type_t::f16, 150, 10),
                                   ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f16, Shape{3, 2, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(
        Shape{3, 2, 4},
        vector<float16>{0.6044921875, 0.8066406250, 0.8320312500, 0.3837890625, 0.0361328125,
                        0.0830078125, 0.5439453125, 0.8339843750, 0.3359375000, 0.7197265625,
                        0.1542968750, 0.1289062500, 0.3476562500, 0.8691406250, 0.4130859375,
                        0.5722656250, 0.5742187500, 0.9394531250, 0.6552734375, 0.8222656250,
                        0.8242187500, 0.1328125000, 0.6435546875, 0.6601562500});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_double_case1)
{
    auto out_shape =
        make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f64, Shape{}, 0);
    auto max_val = make_shared<opset8::Constant>(element::f64, Shape{}, 1);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(
                                       out_shape, min_val, max_val, element::Type_t::f64, 150, 10),
                                   ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f64, Shape{3, 2, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(
        Shape{3, 2, 4},
        vector<double>{0.6089892188187620014616641128668561577796936035156250,
                       0.5144852007657354509007063825265504419803619384765625,
                       0.9355828939149166689759340442833490669727325439453125,
                       0.1577579346062143450524217769270762801170349121093750,
                       0.9309922128081142833622152465977706015110015869140625,
                       0.6153855781977037864294288738165050745010375976562500,
                       0.5091292384542769333677370013901963829994201660156250,
                       0.3119275084940391629118039418244734406471252441406250,
                       0.9712641199166545113996562577085569500923156738281250,
                       0.7486870327629326915541696507716551423072814941406250,
                       0.8924066383588371564883345854468643665313720703125000,
                       0.9381286963095272213308817299548536539077758789062500,
                       0.8092744319940818886749411831260658800601959228515625,
                       0.0753459178107278582103845110395923256874084472656250,
                       0.6800388020136303168783342698588967323303222656250000,
                       0.6126767468248481840475960780167952179908752441406250,
                       0.8341116087085584940297167122480459511280059814453125,
                       0.0966951703057401523722091951640322804450988769531250,
                       0.7800837835828973165774868903099559247493743896484375,
                       0.4718041387914264639391603850526735186576843261718750,
                       0.0911535576997615226702009749715216457843780517578125,
                       0.7567904145507127200431796154589392244815826416015625,
                       0.3918933695224615654240096773719415068626403808593750,
                       0.7331953749194290992363676195964217185974121093750000});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_float32_case2)
{
    auto out_shape =
        make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f32, Shape{}, -650);
    auto max_val = make_shared<opset8::Constant>(element::f32, Shape{}, 450);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(
                                       out_shape, min_val, max_val, element::Type_t::f32, 150, 10),
                                   ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f32, Shape{3, 2, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    auto expected_result =
        vector<float>{121.2359619140625000000,  -314.0640563964843750000, 383.2415771484375000000,
                      390.1638183593750000000,  -521.3574218750000000000, -91.5293579101562500000,
                      -78.3082885742187500000,  -399.9978637695312500000, 440.5114746093750000000,
                      -259.2905578613281250000, 259.6154174804687500000,  8.5134277343750000000,
                      -304.9948120117187500000, -17.7055664062500000000,  -194.6100463867187500000,
                      409.3907470703125000000,  -241.4512023925781250000, 287.7848510742187500000,
                      -547.0555419921875000000, -559.7894897460937500000, -389.7940979003906250000,
                      241.6193847656250000000,  166.4926757812500000000,  187.1735839843750000000};

    auto handle = backend->compile(f);

    handle->call({result}, {});

    auto result_random_uniform = read_vector<float>(result);
    size_t num_of_elems = result_random_uniform.size();
    for (std::size_t j = 0; j < num_of_elems; ++j)
    {
        EXPECT_NEAR(result_random_uniform[j], expected_result[j], 0.0001);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_float16_case2)
{
    auto out_shape =
        make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f16, Shape{}, -1.5);
    auto max_val = make_shared<opset8::Constant>(element::f16, Shape{}, -1.0);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(
                                       out_shape, min_val, max_val, element::Type_t::f16, 150, 10),
                                   ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f16, Shape{3, 2, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(
        Shape{3, 2, 4},
        vector<float16>{-1.1972656250, -1.0966796875, -1.0839843750, -1.3085937500, -1.4824218750,
                        -1.4589843750, -1.2285156250, -1.0830078125, -1.3320312500, -1.1406250000,
                        -1.4228515625, -1.4355468750, -1.3261718750, -1.0654296875, -1.2929687500,
                        -1.2138671875, -1.2128906250, -1.0302734375, -1.1718750000, -1.0888671875,
                        -1.0878906250, -1.4335937500, -1.1777343750, -1.1699218750});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_double_case2)
{
    auto out_shape =
        make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{3, 2, 4});
    auto min_val = make_shared<opset8::Constant>(element::f64, Shape{}, -750);
    auto max_val = make_shared<opset8::Constant>(element::f64, Shape{}, 420);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(
                                       out_shape, min_val, max_val, element::Type_t::f64, 150, 10),
                                   ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::f64, Shape{3, 2, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    auto expected_result = vector<double>{-37.4826139820484058873262256383895874023437500000000000,
                                          -148.0523151040895299956900998950004577636718750000000000,
                                          344.6319858804524756124010309576988220214843750000000000,
                                          -565.4232165107291621097829192876815795898437500000000000,
                                          339.2608889854936933261342346668243408203125000000000000,
                                          -29.9988735086865290213609114289283752441406250000000000,
                                          -154.3187910084959639789303764700889587402343750000000000,
                                          -385.0448150619741909395088441669940948486328125000000000,
                                          386.3790203024857419222826138138771057128906250000000000,
                                          125.9638283326312375720590353012084960937500000000000000,
                                          294.1157668798393842735094949603080749511718750000000000,
                                          347.6105746821467619156464934349060058593750000000000000,
                                          196.8510854330758093055919744074344635009765625000000000,
                                          -661.8452761614483961238875053822994232177734375000000000,
                                          45.6453983559474636422237381339073181152343750000000000,
                                          -33.1682062149276362106320448219776153564453125000000000,
                                          225.9105821890134393470361828804016113281250000000000000,
                                          -636.8666507422840368235483765602111816406250000000000000,
                                          162.6980267919898324180394411087036132812500000000000000,
                                          -197.9891576140310007758671417832374572753906250000000000,
                                          -643.3503374912789922746014781296253204345703125000000000,
                                          135.4447850243338962172856554388999938964843750000000000,
                                          -291.4847576587199569075892213732004165649414062500000000,
                                          107.8385886557319963685586117208003997802734375000000000};

    auto handle = backend->compile(f);

    handle->call({result}, {});

    auto result_random_uniform = read_vector<double>(result);
    size_t num_of_elems = result_random_uniform.size();
    for (std::size_t j = 0; j < num_of_elems; ++j)
    {
        EXPECT_NEAR(result_random_uniform[j], expected_result[j], 1.0E-10);
    }
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_int32)
{
    auto out_shape =
        make_shared<opset8::Constant>(element::i64, Shape{3}, vector<int64_t>{2, 3, 4});
    auto min_val = make_shared<opset8::Constant>(element::i32, Shape{}, -100);
    auto max_val = make_shared<opset8::Constant>(element::i32, Shape{}, 50);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(
                                       out_shape, min_val, max_val, element::Type_t::i32, 100, 350),
                                   ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::i32, Shape{2, 3, 4});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(Shape{2, 3, 4},
                                  vector<int32_t>{
                                      22, -56, -33, -89, -98, -33, -3,  -48, -82, 5,  -66, 21,
                                      29, -42, -73, -37, 3,   36,  -35, 20,  -11, -8, -78, 47,
                                  });

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_int64)
{
    auto out_shape =
        make_shared<opset8::Constant>(element::i32, Shape{3}, vector<int32_t>{5, 4, 3});
    auto min_val = make_shared<opset8::Constant>(element::i64, Shape{}, -2600);
    auto max_val = make_shared<opset8::Constant>(element::i64, Shape{}, 3700);
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(
                                       out_shape, min_val, max_val, element::Type_t::i64, 755, 951),
                                   ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::i64, Shape{5, 4, 3});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(
        Shape{5, 4, 3},
        vector<int64_t>{2116,  -1581, 2559, -339,  -1660, 519,  90,    2027,  -210,  3330,
                        1831,  -1737, 2683, 2661,  3473,  1220, 3534,  -2384, 2199,  1935,
                        499,   2861,  2743, 3223,  -531,  -836, -65,   3435,  632,   1765,
                        2613,  1891,  1698, 3069,  169,   -792, -32,   2976,  -1552, -2588,
                        3327,  -1756, 2637, -1084, 3567,  -778, -1465, 2967,  1242,  2672,
                        -1585, -2271, 3536, -1502, 400,   2241, 3126,  908,   1073,  -2110});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_bfloat16_case1)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{2}, vector<int64_t>{7, 3});
    auto min_val = make_shared<opset8::Constant>(element::bf16, Shape{}, 0);
    auto max_val = make_shared<opset8::Constant>(element::bf16, Shape{}, 1);
    auto f =
        make_shared<Function>(make_shared<opset8::RandomUniform>(
                                  out_shape, min_val, max_val, element::Type_t::bf16, 4978, 5164),
                              ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::bf16, Shape{7, 3});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(
        Shape{7, 3},
        vector<bfloat16>{0.8984375, 0.84375,   0.1640625, 0.1875,    0.46875,   0.6875,   0.5234375,
                         0.3046875, 0.9140625, 0.453125,  0.953125,  0.328125,  0.359375, 0.1875,
                         0.9453125, 0.390625,  0.21875,   0.9921875, 0.8203125, 0.453125, 0.875});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, random_uniform_bfloat16_case2)
{
    auto out_shape = make_shared<opset8::Constant>(element::i32, Shape{2}, vector<int64_t>{7, 3});
    auto min_val = make_shared<opset8::Constant>(element::bf16, Shape{}, -150);
    auto max_val = make_shared<opset8::Constant>(element::bf16, Shape{}, 200);
    auto f =
        make_shared<Function>(make_shared<opset8::RandomUniform>(
                                  out_shape, min_val, max_val, element::Type_t::bf16, 4978, 5164),
                              ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto result = backend->create_tensor(element::bf16, Shape{7, 3});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_expected_output(
        Shape{7, 3}, vector<bfloat16>{164, 146, -92.5, -84.5, 14,  90,    33,  -43.5, 170, 8,  182,
                                      -35, -24, -84.5, 180,   -14, -73.5, 198, 138,   8,   156});

    test_case.run();
}