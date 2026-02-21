// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>

// Test that demonstrates the numerical stability of the LogSumExp algorithm
// This is a pure C++ test that doesn't require OpenVINO runtime
TEST(ReduceLogSumExpNumericalStabilityTest, CompareNaiveVsStable) {
    // This test shows why the stable implementation is necessary

    // Test case 1: Values that cause overflow in naive implementation (around 89)
    {
        std::vector<float> test_values = {89.0f, 89.0f};

        // Naive implementation (would overflow): log(sum(exp(x)))
        float naive_result = std::log(std::exp(test_values[0]) + std::exp(test_values[1]));

        // Stable implementation: k + log(sum(exp(x - k))) where k = max(x)
        float k = std::max(test_values[0], test_values[1]);
        float stable_result = k + std::log(std::exp(test_values[0] - k) + std::exp(test_values[1] - k));

        // The naive implementation should produce inf for large values
        // The stable implementation should produce a finite result
        EXPECT_FALSE(std::isfinite(naive_result)) << "Naive implementation should overflow for values >= 89";
        EXPECT_TRUE(std::isfinite(stable_result)) << "Stable implementation should remain finite";

        // For these specific values, stable result should be approximately 89 + log(2)
        float expected = 89.0f + std::log(2.0f);
        EXPECT_NEAR(stable_result, expected, 1e-5f)
            << "Stable implementation should match expected mathematical result";
    }

    // Test case 2: Very large values (100+)
    {
        std::vector<float> test_values = {100.0f, 101.0f};

        float naive_result = std::log(std::exp(test_values[0]) + std::exp(test_values[1]));

        float k = std::max(test_values[0], test_values[1]);
        float stable_result = k + std::log(std::exp(test_values[0] - k) + std::exp(test_values[1] - k));

        EXPECT_FALSE(std::isfinite(naive_result)) << "Naive implementation should overflow for very large values";
        EXPECT_TRUE(std::isfinite(stable_result)) << "Stable implementation should remain finite";

        // Expected: 101 + log(exp(-1) + 1)
        float expected = 101.0f + std::log(std::exp(-1.0f) + 1.0f);
        EXPECT_NEAR(stable_result, expected, 1e-5f);
    }

    // Test case 3: Values just below the overflow threshold (88)
    {
        std::vector<float> test_values = {88.0f, 88.0f};

        float naive_result = std::log(std::exp(test_values[0]) + std::exp(test_values[1]));
        float k = std::max(test_values[0], test_values[1]);
        float stable_result = k + std::log(std::exp(test_values[0] - k) + std::exp(test_values[1] - k));

        // At 88, naive might still work but is close to overflow
        EXPECT_TRUE(std::isfinite(stable_result)) << "Stable implementation should always remain finite";

        float expected = 88.0f + std::log(2.0f);
        EXPECT_NEAR(stable_result, expected, 1e-5f);
    }

    // Test case 4: Mixed positive and negative values
    {
        std::vector<float> test_values = {-10.0f, 0.0f, 10.0f, 50.0f};

        // Naive would overflow due to exp(50)
        float k = *std::max_element(test_values.begin(), test_values.end());
        float sum_exp = 0.0f;
        for (float val : test_values) {
            sum_exp += std::exp(val - k);
        }
        float stable_result = k + std::log(sum_exp);

        EXPECT_TRUE(std::isfinite(stable_result)) << "Stable implementation should handle mixed values";
        // Result should be dominated by the largest value (50.0)
        EXPECT_GT(stable_result, 50.0f);
        EXPECT_LT(stable_result, 51.0f);
    }
}

// Test the mathematical properties of the stable algorithm
TEST(ReduceLogSumExpNumericalStabilityTest, MathematicalProperties) {
    // Property 1: LogSumExp should be >= max(inputs)
    {
        std::vector<float> values = {1.0f, 2.0f, 3.0f};
        float k = *std::max_element(values.begin(), values.end());
        float sum_exp = 0.0f;
        for (float val : values) {
            sum_exp += std::exp(val - k);
        }
        float result = k + std::log(sum_exp);

        EXPECT_GE(result, k) << "LogSumExp should be >= max(inputs)";
    }

    // Property 2: For identical values, LogSumExp(x, x, ..., x) = x + log(n)
    {
        float x = 42.0f;
        int n = 5;
        std::vector<float> values(n, x);

        float k = x;                         // max is x
        float sum_exp = n * std::exp(0.0f);  // all exp(x - x) = exp(0) = 1
        float result = k + std::log(sum_exp);

        float expected = x + std::log(static_cast<float>(n));
        EXPECT_NEAR(result, expected, 1e-5f) << "LogSumExp of identical values should be x + log(n)";
    }

    // Property 3: LogSumExp is monotonic
    {
        std::vector<float> values1 = {1.0f, 2.0f};
        std::vector<float> values2 = {1.0f, 2.0f, 3.0f};  // values2 contains values1 plus more

        auto compute_lse = [](const std::vector<float>& vals) {
            float k = *std::max_element(vals.begin(), vals.end());
            float sum_exp = 0.0f;
            for (float val : vals) {
                sum_exp += std::exp(val - k);
            }
            return k + std::log(sum_exp);
        };

        float lse1 = compute_lse(values1);
        float lse2 = compute_lse(values2);

        EXPECT_LT(lse1, lse2) << "Adding more positive values should increase LogSumExp";
    }
}
