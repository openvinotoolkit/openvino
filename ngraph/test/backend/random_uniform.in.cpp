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
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, element::Type_t::f32, 150, 10), ParameterVector{});

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
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, element::Type_t::f16, 150, 10), ParameterVector{});

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
    auto f = make_shared<Function>(make_shared<opset8::RandomUniform>(out_shape, element::Type_t::f64, 150, 10), ParameterVector{});

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