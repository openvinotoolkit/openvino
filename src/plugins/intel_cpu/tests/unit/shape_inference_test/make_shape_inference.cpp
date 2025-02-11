// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "shape_inference/shape_inference.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::MatMul;
using ov::op::v0::Parameter;
using ov::op::v0::Result;

TEST(StaticShapeInferenceTest, MakeShapeInference) {
    auto inp1_f32 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(4));
    auto inp2_f32 = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(4));

    auto inp1 = std::make_shared<Parameter>(element::i8, PartialShape::dynamic(4));
    auto inp2 = std::make_shared<Parameter>(element::i8, PartialShape::dynamic(4));

    auto matMulRelaxed = std::make_shared<ov::op::TypeRelaxed<MatMul>>(
        *as_type_ptr<MatMul>(std::make_shared<MatMul>(inp1_f32, inp2_f32, false, false)),
        element::f32);

    auto matMul = matMulRelaxed->clone_with_new_inputs({inp1, inp2});

    ov::ResultVector results;
    results.push_back(std::make_shared<Result>(matMul->output(0)));

    auto model = std::make_shared<ov::Model>(results, ov::ParameterVector{inp1, inp2}, "testFunction");
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
