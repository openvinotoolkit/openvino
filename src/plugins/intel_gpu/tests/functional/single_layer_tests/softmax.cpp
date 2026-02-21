// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

#include "shared_test_classes/single_op/softmax.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov::test::subgraph {

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
    for (size_t i = 0; i < values.size(); ++i) {
        data[i] = values[i];
    }
}

static void check_output(const std::vector<float>& expected,
                         const std::vector<float>& actual) {
    ASSERT_EQ(expected.size(), actual.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::isnan(expected[i])) {
            EXPECT_TRUE(std::isnan(actual[i]));
        } else {
            EXPECT_NEAR(expected[i], actual[i], 1e-6f);
        }
    }
}

TEST_P(SoftMaxLayerTest, MixedInfinityCases) {
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
    prepare_input({-INFINITY, 1.f, 2.f}, inputsData, inputDynamicShapes);
    run();
    auto out = get_runtime_output()[0].as<std::vector<float>>();

    std::vector<float> expected = {0.f, 0.2689414f, 0.7310586f};

    // Reviewer suggestion applied (both forms)
    EXPECT_THAT(out, ::testing::ElementsAreArray(expected));
    EXPECT_THAT(out, ::testing::ElementsAreArray(0.f, 0.2689414f, 0.7310586f));
}

TEST_P(SoftMaxLayerTest, NaNPropagationCases) {
    std::vector<std::vector<float>> cases = {
        {NAN, 1.f, 2.f},
        {1.f, NAN, 2.f},
        {NAN, NAN, NAN}
    };

    for (auto& c : cases) {
        prepare_input(c, inputsData, inputDynamicShapes);
        run();
        auto out = get_runtime_output()[0].as<std::vector<float>>();

        std::vector<float> expected(out.size(), std::numeric_limits<float>::quiet_NaN());

        // Reviewer suggestion applied
        EXPECT_THAT(out, ::testing::ElementsAreArray(expected));
        EXPECT_THAT(out, ::testing::Each(std::numeric_limits<float>::quiet_NaN()));
    }
}

} // namespace ov::test::subgraph
