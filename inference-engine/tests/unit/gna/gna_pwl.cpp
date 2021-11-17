// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "runtime/pwl.h"
#include "common_test_utils/data_utils.hpp"

namespace {
class PwlTest : public ::testing::Test {
protected:
    size_t range_search(double arg, const std::vector<pwl_t>& pwl) {
        size_t left = 0;
        size_t right = pwl.size() - 1;
        size_t mid = (right - left) / 2;
        while (left < right && (arg < pwl[mid].alpha || pwl[mid + 1].alpha < arg)) {
            if (arg < pwl[mid].alpha) {
                right = mid;
            } else {
                left = mid;
            }

            mid = left + (right - left) / 2;
        }

        return mid;
    }

    double pwl_function(double x, const std::vector<pwl_t>& pwl) {
        size_t segment_index = 0;
        if (x < pwl.front().alpha) {
            segment_index = 0;
        } else if (x > pwl.back().alpha) {
            segment_index = pwl.size() - 2;
        } else {
            auto index = range_search(x, pwl);
            segment_index = index == pwl.size() - 1 ? pwl.size() - 2 : index;
        }

        return pwl[segment_index].m * x + pwl[segment_index].b;
    }

    double function(double x, const DnnActivation& activation_type) {
        switch (activation_type) {
            case kActSigmoid:
                return 0.5 * (1.0 + tanh(x / 2.0));
            case kActTanh:
                return tanh(x);
            case kActSoftSign:
                return x / (1.0 + fabs(x));
            case kActExp:
                return exp(x);
            case kActLog:
                return log(x);
            case kActNegLog:
                return 1.0 * log(x);
            case kActNegHalfLog:
                return 0.5 * log(x);
            case kActPow:
                return pow(activation_type.args.pow.offset + activation_type.args.pow.scale * x, activation_type.args.pow.exponent);
            default:
                throw std::exception();
        }
    }

    bool GetPwl(const DnnActivation& activation_type,
                const double lower_bound,
                const double upper_bound,
                const double allowed_err_pct,
                std::vector<pwl_t>& pwl,
                const int max_iteration_number = 0) {
        return pwl_search(activation_type,
                          lower_bound,
                          upper_bound,
                          PWL_DESIGN_THRESHOLD,
                          allowed_err_pct,
                          PWL_DESIGN_SAMPLES,
                          pwl,
                          PWL_MAX_NUM_SEGMENTS,
                          max_iteration_number);
    }

    bool Check(const DnnActivation& activation_type,
               const size_t size,
               const double lower_bound,
               const double upper_bound,
               const double allowed_err_pct,
               std::vector<pwl_t>& pwl) {
        std::vector<float> data = CommonTestUtils::generate_float_numbers(size, lower_bound, upper_bound);
        for (auto x : data) {
            double delta = std::abs(pwl_function(x, pwl) - function(x, activation_type));
            if (delta > allowed_err_pct) {
                return false;
            }
        }
        return true;
    }
};

TEST_F(PwlTest, sigmoid) {
    std::vector<pwl_t> pwl;
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActSigmoid), -SIGMOID_DOMAIN, SIGMOID_DOMAIN, 1, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActSigmoid), 100, -SIGMOID_DOMAIN, SIGMOID_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActSigmoid), -SIGMOID_DOMAIN, SIGMOID_DOMAIN, 1, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActSigmoid), 100, -SIGMOID_DOMAIN, SIGMOID_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActSigmoid), -SIGMOID_DOMAIN, SIGMOID_DOMAIN, 0, pwl));
    double err_pct = 0.0046;
    ASSERT_TRUE(Check(DnnActivation::fromType(kActSigmoid), 100, -SIGMOID_DOMAIN, SIGMOID_DOMAIN, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActSigmoid), -SIGMOID_DOMAIN, SIGMOID_DOMAIN, err_pct, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActSigmoid), -SIGMOID_DOMAIN, SIGMOID_DOMAIN, err_pct, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActSigmoid), 100, -SIGMOID_DOMAIN, SIGMOID_DOMAIN, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActSigmoid), -SIGMOID_DOMAIN, SIGMOID_DOMAIN, err_pct/2, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActSigmoid), -SIGMOID_DOMAIN, SIGMOID_DOMAIN, err_pct*2, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActSigmoid), 100, -SIGMOID_DOMAIN, SIGMOID_DOMAIN, err_pct*2, pwl));
}

TEST_F(PwlTest, tanh) {
    std::vector<pwl_t> pwl;
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActTanh), -TANH_DOMAIN, TANH_DOMAIN, 1, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActTanh), 100, -TANH_DOMAIN, TANH_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActTanh), -TANH_DOMAIN, TANH_DOMAIN, 1, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActTanh), 100, -TANH_DOMAIN, TANH_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActTanh), -TANH_DOMAIN, TANH_DOMAIN, 0, pwl));
    double err_pct = 0.0046;
    ASSERT_TRUE(Check(DnnActivation::fromType(kActTanh), 100, -TANH_DOMAIN, TANH_DOMAIN, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActTanh), -TANH_DOMAIN, TANH_DOMAIN, err_pct, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActTanh), -TANH_DOMAIN, TANH_DOMAIN, err_pct, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActTanh), 100, -TANH_DOMAIN, TANH_DOMAIN, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActTanh), -TANH_DOMAIN, TANH_DOMAIN, err_pct/2, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActTanh), -TANH_DOMAIN, TANH_DOMAIN, err_pct*2, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActTanh), 100, -TANH_DOMAIN, TANH_DOMAIN, err_pct*2, pwl));
}

TEST_F(PwlTest, softsign) {
    std::vector<pwl_t> pwl;
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActSoftSign), -SOFTSIGN_DOMAIN, SOFTSIGN_DOMAIN, 1, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActSoftSign), 100, -SOFTSIGN_DOMAIN, SOFTSIGN_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActSoftSign), -SOFTSIGN_DOMAIN, SOFTSIGN_DOMAIN, 1, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActSoftSign), 100, -SOFTSIGN_DOMAIN, SOFTSIGN_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActSoftSign), -SOFTSIGN_DOMAIN, SOFTSIGN_DOMAIN, 0, pwl));
    double err_pct = 0.0071;
    ASSERT_TRUE(Check(DnnActivation::fromType(kActSoftSign), 100, -SOFTSIGN_DOMAIN, SOFTSIGN_DOMAIN, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActSoftSign), -SOFTSIGN_DOMAIN, SOFTSIGN_DOMAIN, err_pct, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActSoftSign), -SOFTSIGN_DOMAIN, SOFTSIGN_DOMAIN, err_pct, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActSoftSign), 100, -SOFTSIGN_DOMAIN, SOFTSIGN_DOMAIN, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActSoftSign), -SOFTSIGN_DOMAIN, SOFTSIGN_DOMAIN, err_pct/2, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActSoftSign), -SOFTSIGN_DOMAIN, SOFTSIGN_DOMAIN, err_pct*2, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActSoftSign), 100, -SOFTSIGN_DOMAIN, SOFTSIGN_DOMAIN, err_pct*2, pwl));
}

TEST_F(PwlTest, exp) {
    std::vector<pwl_t> pwl;
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActExp), -4.5, 4.5, 1, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActExp), 100, -4.5, 4.5, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActExp), -4.5, 4.5, 1, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActExp), 100, -4.5, 4.5, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActExp), -1, 1, 0, pwl));
    double err_pct = 0.0017;
    ASSERT_TRUE(Check(DnnActivation::fromType(kActExp), 100, -1.0, 1.0, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActExp), -1.0, 1.0, err_pct, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActExp), -1.0, 1.0, err_pct, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActExp), 100, -1.0, 1.0, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActExp), -1.0, 1.0, err_pct/2, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActExp), -1.0, 1.0, err_pct*2, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActExp), 100, -1.0, 1.0, err_pct*2, pwl));
}

TEST_F(PwlTest, log) {
    std::vector<pwl_t> pwl;
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActLog), 1e-10, LOG_DOMAIN, 1, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActLog), 100, 1e-10, LOG_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActLog), 1e-10, LOG_DOMAIN, 1, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActLog), 100, 1e-10, LOG_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActLog), 1e-10, LOG_DOMAIN, 0, pwl));
    double err_pct = 0.013;
    ASSERT_TRUE(Check(DnnActivation::fromType(kActLog), 100, 1e-10, LOG_DOMAIN, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActLog), 1e-10, LOG_DOMAIN, err_pct, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActLog), 1e-10, LOG_DOMAIN, err_pct, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActLog), 100, 1e-10, LOG_DOMAIN, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActLog), 1e-10, LOG_DOMAIN, err_pct/2, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActLog), 1e-10, LOG_DOMAIN, err_pct*2, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActLog), 100, 1e-10, LOG_DOMAIN, err_pct*2, pwl));
}

TEST_F(PwlTest, neglog) {
    std::vector<pwl_t> pwl;
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActNegLog), 1e-10, LOG_DOMAIN, 1, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActNegLog), 100, 1e-10, LOG_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActNegLog), 1e-10, LOG_DOMAIN, 1, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActNegLog), 100, 1e-10, LOG_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActNegLog), 1e-10, LOG_DOMAIN, 0, pwl));
    double err_pct = 0.013;
    ASSERT_TRUE(Check(DnnActivation::fromType(kActNegLog), 100, 1e-10, LOG_DOMAIN, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActNegLog), 1e-10, LOG_DOMAIN, err_pct, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActNegLog), 1e-10, LOG_DOMAIN, err_pct, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActNegLog), 100, 1e-10, LOG_DOMAIN, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActNegLog), 1e-10, LOG_DOMAIN, err_pct/2, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActNegLog), 1e-10, LOG_DOMAIN, err_pct*2, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActNegLog), 100, 1e-10, LOG_DOMAIN, err_pct*2, pwl));
}

TEST_F(PwlTest, neghalflog) {
    std::vector<pwl_t> pwl;
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActNegHalfLog), 1e-10, LOG_DOMAIN, 1, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActNegHalfLog), 100, 1e-10, LOG_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActNegHalfLog), 1e-10, LOG_DOMAIN, 1, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActNegHalfLog), 100, 1e-10, LOG_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActNegHalfLog), 1e-10, LOG_DOMAIN, 0, pwl));
    double err_pct = 0.013;
    ASSERT_TRUE(Check(DnnActivation::fromType(kActNegHalfLog), 100, 1e-10, LOG_DOMAIN, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActNegHalfLog), 1e-10, LOG_DOMAIN, err_pct, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActNegHalfLog), 1e-10, LOG_DOMAIN, err_pct, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActNegHalfLog), 100, 1e-10, LOG_DOMAIN, err_pct, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActNegHalfLog), 1e-10, LOG_DOMAIN, err_pct/2, pwl));
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActNegHalfLog), 1e-10, LOG_DOMAIN, err_pct*2, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActNegHalfLog), 100, 1e-10, LOG_DOMAIN, err_pct*2, pwl));
}

TEST_F(PwlTest, power) {
    std::vector<pwl_t> pwl;
    DnnActivation activation_type = DnnActivation::fromType(kActPow);
    activation_type.args.pow.offset = 0.5;
    activation_type.args.pow.scale = 0.5;
    activation_type.args.pow.exponent = 2;
    ASSERT_TRUE(GetPwl(activation_type, -POW_DOMAIN, POW_DOMAIN, 1, pwl, PWL_MAX_ITERATIONS_DEFAULT));
    ASSERT_TRUE(Check(activation_type, 100, -POW_DOMAIN, POW_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(activation_type, -POW_DOMAIN, POW_DOMAIN, 1, pwl));
    ASSERT_TRUE(Check(activation_type, 100, -POW_DOMAIN, POW_DOMAIN, 1, pwl));
    ASSERT_TRUE(GetPwl(activation_type, -POW_DOMAIN, POW_DOMAIN, 0, pwl));
    double err_pct = 0.004;
    ASSERT_TRUE(Check(activation_type, 100, -POW_DOMAIN, POW_DOMAIN, err_pct, pwl));
    ASSERT_TRUE(GetPwl(activation_type, -POW_DOMAIN, POW_DOMAIN, err_pct, pwl));
    ASSERT_TRUE(Check(activation_type, 100, -POW_DOMAIN, POW_DOMAIN, err_pct, pwl));
    EXPECT_FALSE(GetPwl(activation_type, -POW_DOMAIN, POW_DOMAIN, err_pct/2, pwl));
    ASSERT_TRUE(GetPwl(activation_type, -POW_DOMAIN, POW_DOMAIN, err_pct*2, pwl));
    ASSERT_TRUE(Check(activation_type, 100, -POW_DOMAIN, POW_DOMAIN, err_pct*2, pwl));
}

} // namespace
