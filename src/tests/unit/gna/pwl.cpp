// Copyright (C) 2018-2022 Intel Corporation
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
            case kActNegLog:
                return -1.0 * log(x);
            case kActNegHalfLog:
                return -0.5 * log(x);
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
        double err_pct = 0;
        try {
            pwl = pwl_search(activation_type,
                             lower_bound,
                             upper_bound,
                             PWL_DESIGN_THRESHOLD,
                             allowed_err_pct,
                             PWL_DESIGN_SAMPLES,
                             err_pct);
        } catch(...) {
            return false;
        }
        return true;
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

TEST_F(PwlTest, neglog) {
    std::vector<pwl_t> pwl;
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActNegLog), 1e-10, LOG_DOMAIN, 1, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActNegLog), 100, 1e-10, LOG_DOMAIN, 1, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActNegLog), 1e-10, LOG_DOMAIN, 0, pwl));
}

TEST_F(PwlTest, neghalflog) {
    std::vector<pwl_t> pwl;
    ASSERT_TRUE(GetPwl(DnnActivation::fromType(kActNegHalfLog), 1e-10, LOG_DOMAIN, 1, pwl));
    ASSERT_TRUE(Check(DnnActivation::fromType(kActNegHalfLog), 100, 1e-10, LOG_DOMAIN, 1, pwl));
    EXPECT_FALSE(GetPwl(DnnActivation::fromType(kActNegHalfLog), 1e-10, LOG_DOMAIN, 0, pwl));
}

} // namespace
