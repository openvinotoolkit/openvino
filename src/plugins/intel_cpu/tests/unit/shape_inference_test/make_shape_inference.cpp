// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>
#include "ngraph_functions/builders.hpp"
#include <thread>
#include <atomic>
#include <ngraph_ops/type_relaxed.hpp>

using namespace ov;
using namespace ov::intel_cpu;

TEST(StaticShapeInferenceTest, MakeShapeInference) {
    auto inp1_f32 = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto inp2_f32 = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto inp1 = std::make_shared<op::v0::Parameter>(element::i8, PartialShape{-1, -1, -1, -1});
    auto inp2 = std::make_shared<op::v0::Parameter>(element::i8, PartialShape{-1, -1, -1, -1});

    auto matMulRelaxed = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset3::MatMul>>(
            *as_type_ptr<ngraph::opset3::MatMul>(ngraph::builder::makeMatMul(inp1_f32, inp2_f32, false, false)),
            element::f32);

    auto matMul = matMulRelaxed->clone_with_new_inputs({inp1, inp2});

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(matMul->output(0)));

    auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{inp1, inp2}, "testFunction");
    std::atomic_flag wrongPrcFlag;
    wrongPrcFlag.clear();

    auto testFunc = [&]() {
        if (matMul->get_input_element_type(0) != element::i8) {
            wrongPrcFlag.test_and_set();
        }
        if (matMul->get_input_element_type(1) != element::i8) {
            wrongPrcFlag.test_and_set();
        }
        auto shapeInfer = make_shape_inference(matMul);
    };

    const size_t numThreads = 24;
    std::vector<std::thread> threads(numThreads);

    for (auto&& thread : threads)
        thread = std::thread(testFunc);

    for (auto&& th : threads) {
        th.join();
    }

    ASSERT_FALSE(wrongPrcFlag.test_and_set());
}