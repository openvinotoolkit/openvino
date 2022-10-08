// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//#include "backend/gna_limitations.hpp"
//#include "common/gna_target.hpp"

#include <gtest/gtest.h>

#include <cstdint>

#include "backend/dnn_types.h"
#include "backend/make_pwl.hpp"
#include "backend/pwl_input_params.hpp"
#include "backend/pwl_segments_creator_factory.hpp"
#include "backend/pwl_tools.hpp"
#include "round_float_define.hpp"
#include "runtime/pwl.h"

using namespace ov::intel_gna::backend;
using namespace ov::intel_gna::backend::pwl_tools;

namespace {

struct MakePWLIdentityTestParam {
    double in_scale_;
    double out_scale_;
    bool should_throw_;
    bool operator<(const MakePWLIdentityTestParam& rhs) const {
        if (in_scale_ < rhs.in_scale_) {
            return true;
        } else if (in_scale_ > rhs.in_scale_) {
            return false;
        } else if (out_scale_ < rhs.out_scale_) {
            return true;
        }
        return false;
    }
};

std::string SegmentToString(const gna_pwl_segment_t& segment) {
    std::stringstream stream;
    stream << "Segment(";
    stream << "xBase=" << static_cast<int32_t>(segment.xBase & XBASEMASK);
    stream << ", yBase=" << segment.yBase;
    stream << ", slope=" << segment.slope;
    stream << ", slope_scale=" << ComputeSlopeScale(segment.xBase);
    stream << ")";
    return stream.str();
}

size_t get_segment_index(const std::vector<gna_pwl_segment_t> pwl_segments, const int64_t x) {
    int j = 0;
    for (int i = 0; i < pwl_segments.size(); ++i) {
        if (static_cast<int32_t>(pwl_segments[i].xBase & XBASEMASK) <= x) {
            j = i;
        } else {
            break;
        }
    }
    return j;
}

class MakePWLIdentityTestFixture : public ::testing::TestWithParam<MakePWLIdentityTestParam> {
public:
    static std::string GetTestCaseName(const testing::TestParamInfo<MakePWLIdentityTestParam>& obj);

    DnnActivation identity_fun_{DnnActivation::fromType(kActIdentity)};
    std::vector<pwl_t> pwl_;
    double l_bound_{-1.0};
    double u_bound_{1.0};
    bool low_precision_ = false;

protected:
    std::set<int32_t> GenTestXValues(const std::vector<gna_pwl_segment_t>& segments);

    void ValidateSegments(const std::vector<gna_pwl_segment_t>& segments) {
        std::set<int32_t> test_values = GenTestXValues(segments);
        int64_t prev = std::numeric_limits<int64_t>::min();
        for (const auto& value : test_values) {
            const auto& segment = segments[get_segment_index(segments, value)];
            auto result = ComputePWL(segment, value);

            // check saturation
            if (result > std::numeric_limits<int16_t>::max() || result < std::numeric_limits<int16_t>::min()) {
                FAIL() << "PWL with saturation F(" << value << ")=" << result << ", for " << SegmentToString(segment);
            }
            // check check monotocity
            if (prev > result) {
                FAIL() << "PWL is not monotonic for F(" << value << ")=" << std::to_string(result)
                       << " for segment: " << SegmentToString(segment) << ", where F(" << value - 1
                       << ") = " << std::to_string(prev) << "!";
            }
            // check 0,0
            if (value == 0 && result != 0) {
                FAIL() << "PWL does not pass (0,0), but it is F(" << value << ")=" << result << ", for "
                       << SegmentToString(segment);
            }
            prev = result;
        }
    }
};

std::string MakePWLIdentityTestFixture::GetTestCaseName(const testing::TestParamInfo<MakePWLIdentityTestParam>& obj) {
    auto param = obj.param;
    std::ostringstream result;

    result << "_scale_in=" << std::to_string(param.in_scale_) << "_";
    result << "_scale_out=" << std::to_string(param.out_scale_);
    return result.str();
}

inline std::set<int32_t> MakePWLIdentityTestFixture::GenTestXValues(const std::vector<gna_pwl_segment_t>& segments) {
    std::set<int32_t> test_x_values;
    test_x_values.insert(std::numeric_limits<int32_t>::min());
    test_x_values.insert(std::numeric_limits<int32_t>::max());

    test_x_values.insert(-1);
    test_x_values.insert(0);
    test_x_values.insert(1);
    for (const auto& segment : segments) {
        auto x_base = static_cast<int32_t>(segment.xBase & XBASEMASK);
        if (x_base > std::numeric_limits<int32_t>::min()) {
            test_x_values.insert(x_base - 1);
        }
        test_x_values.insert(x_base);
        if (x_base < std::numeric_limits<int32_t>::max()) {
            test_x_values.insert(x_base + 1);
        }
    }
    return test_x_values;
}

// This test check if method available in the system returns proper segments.
TEST_P(MakePWLIdentityTestFixture, check_make_pwl) {
    auto input_params = GetParam();
    std::vector<gna_pwl_segment_t> output_pwl;

    try {
        make_gna_pwl(identity_fun_,
                     pwl_,
                     l_bound_,
                     u_bound_,
                     input_params.in_scale_,
                     input_params.out_scale_,
                     low_precision_,
                     false,
                     output_pwl);
        if (input_params.should_throw_) {
            FAIL() << "Should throw, but didn't";
            return;
        }
    } catch (const std::exception& e) {
        if (!input_params.should_throw_) {
            FAIL() << "Thrown but shouldn't due to: " << e.what();
        }
        return;
    }

    ASSERT_FALSE(output_pwl.empty());

    ValidateSegments(output_pwl);
}

// This test check segment creation for PWL Identity
TEST_P(MakePWLIdentityTestFixture, check_pwl_identity_create_segments) {
    auto input_params = GetParam();

    auto pwl_creator = PWLSegmentsCreatorFactory::CreateCreator(identity_fun_.type);
    ASSERT_NE(pwl_creator, nullptr);

    std::vector<gna_pwl_segment_t> output_pwl;

    try {
        PWLInputParams pwl_input(low_precision_,
                                 identity_fun_.fqParams,
                                 input_params.in_scale_,
                                 input_params.out_scale_);
        output_pwl = pwl_creator->CreateSegments(pwl_input);
        if (input_params.should_throw_) {
            FAIL() << "Should throw, but didn't";
            return;
        }
    } catch (const std::exception& e) {
        if (!input_params.should_throw_) {
            FAIL() << "Thrown but shouldn't due to: " << e.what();
        }
        return;
    }

    ASSERT_FALSE(output_pwl.empty());

    ValidateSegments(output_pwl);
}

// This test check segment creation with borders for PWL Identity
TEST_P(MakePWLIdentityTestFixture, check_pwl_identity_create_segments_with_borders) {
    auto input_params = GetParam();

    auto pwl_creator = PWLSegmentsCreatorFactory::CreateCreator(identity_fun_.type);
    ASSERT_NE(pwl_creator, nullptr);

    PWLSegmentsWithBorderValues output_values;

    try {
        PWLInputParams pwl_input(low_precision_,
                                 identity_fun_.fqParams,
                                 input_params.in_scale_,
                                 input_params.out_scale_);
        output_values = pwl_creator->CreateSegmentsWithBorders(pwl_input);
        if (input_params.should_throw_) {
            FAIL() << "Should throw, but didn't";
            return;
        }
    } catch (const std::exception& e) {
        if (!input_params.should_throw_) {
            FAIL() << "Thrown but shouldn't due to: " << e.what();
        }
        return;
    }

    auto& segments = output_values.segments;

    ASSERT_FALSE(segments.empty());

    ValidateSegments(segments);
}
/**
 * Create Input parameters. Sets should throw to true in case scale factors relation is inproper
 */
MakePWLIdentityTestParam createIdentityParamsForScales(double in, double out) {
    auto slope = ComputeSlopeForSegment(1.0, in, out);

    // in case there is risk of division of 0 for given in/out parameters exception should be thrown
    bool should_throw = false;

    // check if exception is thrown if division by zero is possible
    // check if exception is thrown if scale factor with too big difference are used
    const auto x_lower = FLOAT_TO_INT32(static_cast<double>(std::numeric_limits<int16_t>::min()) * in / out);

    if (slope.value == 0 || x_lower == 0) {
        should_throw = true;
    }
    return {in, out, should_throw};
}

std::set<MakePWLIdentityTestParam> GenerateParams() {
    // use set to avoid duplicates
    std::set<MakePWLIdentityTestParam> params;

    params.insert(createIdentityParamsForScales(1.0, 1.0));
    params.insert(createIdentityParamsForScales(1.0, 2049.0));
    params.insert(createIdentityParamsForScales(16777216.0, 2049.0));
    params.insert(createIdentityParamsForScales(17895698.0, 2049.0));
    params.insert(createIdentityParamsForScales(16384.0, 2049.0));
    params.insert(createIdentityParamsForScales(38347924.0, 2049.0));
    params.insert(createIdentityParamsForScales(33570816.0, 2049.0));
    params.insert(createIdentityParamsForScales(30720.0, 69905.0));
    params.insert(createIdentityParamsForScales(542115584.0, 2049.0));
    params.insert(createIdentityParamsForScales(273.0, 7866240.0));
    params.insert(createIdentityParamsForScales(2049.0, 273.0));
    params.insert(createIdentityParamsForScales(286331153.0, 7.0));
    params.insert(createIdentityParamsForScales(286331153.0, 2049.0));
    params.insert(createIdentityParamsForScales(6712372736.0, 2049.0));

    for (int64_t i = 1; i < std::numeric_limits<int32_t>::max(); i = i * 16 + 1) {
        params.insert(createIdentityParamsForScales(static_cast<double>(i), 2049.0));
    }

    for (int64_t i = 1; i < std::numeric_limits<int32_t>::max(); i = i * 16 + 1) {
        params.insert(createIdentityParamsForScales(2049.0, static_cast<double>(i)));
    }

    for (int64_t i = 1; i < std::numeric_limits<int32_t>::max(); i = i * 16 + 1) {
        params.insert(createIdentityParamsForScales(static_cast<double>(i),
                                                    static_cast<double>(std::numeric_limits<int32_t>::max() / i)));
    }

    for (int64_t i = 1; i < std::numeric_limits<int32_t>::max(); i = i * 16 + 1) {
        params.insert(createIdentityParamsForScales(static_cast<double>(std::numeric_limits<int32_t>::max() / i),
                                                    static_cast<double>(i)));
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(MakePWLIdentityTests,
                         MakePWLIdentityTestFixture,
                         ::testing::ValuesIn(GenerateParams()),
                         MakePWLIdentityTestFixture::GetTestCaseName);

}  // namespace
