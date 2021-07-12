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