// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "ie_common.h"
#include "openvino/op/matmul.hpp"
#include "ov_models/builders.hpp"
#include "ov_ops/type_relaxed.hpp"

using namespace ov;

class TypeRelaxedThreading : public testing::Test {
public:
    static void runParallel(std::function<void(void)> func,
                            const unsigned int iterations = 100,
                            const unsigned int threadsNum = 24) {
        std::vector<std::thread> threads(threadsNum);
        for (auto& thread : threads) {
            thread = std::thread([&]() {
                for (unsigned int i = 0; i < iterations; ++i) {
                    func();
                }
            });
        }
        for (auto& thread : threads) {
            if (thread.joinable())
                thread.join();
        }
    }
};

TEST_F(TypeRelaxedThreading, TypeRelaxedCloning) {
    auto inp1_f32 = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto inp2_f32 = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto inp1 = std::make_shared<op::v0::Parameter>(element::i8, PartialShape{-1, -1, -1, -1});
    auto inp2 = std::make_shared<op::v0::Parameter>(element::i8, PartialShape{-1, -1, -1, -1});

    auto matMulRelaxed = std::make_shared<ov::op::TypeRelaxed<ov::op::v0::MatMul>>(
        *as_type_ptr<op::v0::MatMul>(ngraph::builder::makeMatMul(inp1_f32, inp2_f32, false, false)),
        element::f32);
    auto matMul = matMulRelaxed->clone_with_new_inputs({inp1, inp2});

    runParallel([&]() {
        auto inp3 = std::make_shared<op::v0::Parameter>(element::i8, PartialShape{-1, -1, -1, -1});
        auto inp4 = std::make_shared<op::v0::Parameter>(element::i8, PartialShape{-1, -1, -1, -1});
        auto copied_matMul = matMulRelaxed->clone_with_new_inputs({inp3, inp4});
    });
}
