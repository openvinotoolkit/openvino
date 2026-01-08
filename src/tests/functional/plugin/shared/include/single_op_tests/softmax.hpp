// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/softmax.hpp"
#include <limits>
#include <cmath>


namespace ov {
namespace test {
namespace subgraph {

TEST_P(SoftMaxLayerTest, CompareWithRefs) {
    run();
}

TEST_P(SoftMaxLayerTest, CompareQueryModel) {
    query_model();
}

TEST_P(SoftMax8LayerTest, CompareWithRefs) {
    run();
}

TEST_P(SoftMax8LayerTest, CompareQueryModel) {
    query_model();
}



// =======================
// GPU Numerical Edge Cases
// =======================

static void prepare_input(const std::vector<float>& values,
                          std::map<std::string, ov::Tensor>& inputsData,
                          std::vector<ov::Shape>& inputDynamicShapes) {
    inputDynamicShapes.clear();
    inputDynamicShapes.push_back({values.size()});

    auto& tensor = inputsData.begin()->second;
    auto* data = tensor.data<float>();
    for (size_t i = 0; i < values.size(); ++i)
        data[i] = values[i];
}

static void check_output(const std::vector<float>& expected,
                         const std::vector<float>& actual) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::isnan(expected[i])) {
            EXPECT_TRUE(std::isnan(actual[i]));
        } else {
            EXPECT_NEAR(expected[i], actual[i], 1e-6f);
        }
    }
}

TEST_P(SoftMaxLayerTest, MixedInfinityCases) {
    if (targetDevice != ov::test::utils::DEVICE_GPU)
        GTEST_SKIP();

    std::vector<std::pair<std::vector<float>, std::vector<float>>> cases = {
        {{INFINITY, 1.f, 2.f}, {NAN, 0.f, 0.f}},
        {{INFINITY, -INFINITY, 1.f}, {NAN, 0.f, 0.f}}
    };

    for (auto& c : cases) {
        prepare_input(c.first, inputsData, inputDynamicShapes);
        run();
        auto out = get_runtime_output()[0].as<std::vector<float>>();
        check_output(c.second, out);
    }
}

TEST_P(SoftMaxLayerTest, MultipleInfinityCases) {
    if (targetDevice != ov::test::utils::DEVICE_GPU)
        GTEST_SKIP();

    std::vector<std::pair<std::vector<float>, std::vector<float>>> cases = {
        {{INFINITY, INFINITY, 1.f}, {NAN, NAN, 0.f}},
        {{INFINITY, INFINITY, INFINITY}, {NAN, NAN, NAN}},
        {{INFINITY, -INFINITY, -INFINITY}, {NAN, 0.f, 0.f}}
    };

    for (auto& c : cases) {
        prepare_input(c.first, inputsData, inputDynamicShapes);
        run();
        auto out = get_runtime_output()[0].as<std::vector<float>>();
        check_output(c.second, out);
    }
}

TEST_P(SoftMaxLayerTest, NegativeInfinityOnlyCase) {
    if (targetDevice != ov::test::utils::DEVICE_GPU)
        GTEST_SKIP();

    prepare_input({-INFINITY, 1.f, 2.f}, inputsData, inputDynamicShapes);
    run();

    auto out = get_runtime_output()[0].as<std::vector<float>>();
    check_output({0.f, 0.2689414f, 0.7310586f}, out);
}

TEST_P(SoftMaxLayerTest, NaNPropagationCases) {
    if (targetDevice != ov::test::utils::DEVICE_GPU)
        GTEST_SKIP();

    std::vector<std::vector<float>> cases = {
        {NAN, 1.f, 2.f},
        {1.f, NAN, 2.f},
        {NAN, NAN, NAN}
    };

    for (auto& c : cases) {
        prepare_input(c, inputsData, inputDynamicShapes);
        run();

        auto out = get_runtime_output()[0].as<std::vector<float>>();
        for (float v : out)
            EXPECT_TRUE(std::isnan(v));
    }
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov
