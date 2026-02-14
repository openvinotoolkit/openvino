// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::test {

// --- ov::hint::num_requests (uint32_t property) ---

// Negative values from different signed types should be rejected
TEST(PropertyValidation, NumRequestsRejectsNegativeInt) {
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(-1), ov::Exception, testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(-100), ov::Exception, testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(std::numeric_limits<int>::min()),
                    ov::Exception,
                    testing::HasSubstr("negative"));
}

TEST(PropertyValidation, NumRequestsRejectsNegativeLong) {
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(-1L), ov::Exception, testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(static_cast<long>(-100)),
                    ov::Exception,
                    testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(std::numeric_limits<long>::min()),
                    ov::Exception,
                    testing::HasSubstr("negative"));
}

TEST(PropertyValidation, NumRequestsRejectsNegativeInt64) {
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(int64_t{-1}), ov::Exception, testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(std::numeric_limits<int64_t>::min()),
                    ov::Exception,
                    testing::HasSubstr("negative"));
}

TEST(PropertyValidation, NumRequestsRejectsNegativeShort) {
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(short{-1}), ov::Exception, testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(std::numeric_limits<short>::min()),
                    ov::Exception,
                    testing::HasSubstr("negative"));
}

TEST(PropertyValidation, NumRequestsRejectsNegativeSignedChar) {
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(static_cast<signed char>(-1)),
                    ov::Exception,
                    testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(std::numeric_limits<signed char>::min()),
                    ov::Exception,
                    testing::HasSubstr("negative"));
}

// Zero from signed types should be accepted
TEST(PropertyValidation, NumRequestsAcceptsZeroFromSignedTypes) {
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests(0);
        EXPECT_EQ(kv.second.as<uint32_t>(), 0u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests(0L);
        EXPECT_EQ(kv.second.as<uint32_t>(), 0u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests(int64_t{0});
        EXPECT_EQ(kv.second.as<uint32_t>(), 0u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests(short{0});
        EXPECT_EQ(kv.second.as<uint32_t>(), 0u);
    });
}

// Positive values from signed types should be accepted
TEST(PropertyValidation, NumRequestsAcceptsPositiveFromSignedTypes) {
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests(1);
        EXPECT_EQ(kv.second.as<uint32_t>(), 1u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests(100);
        EXPECT_EQ(kv.second.as<uint32_t>(), 100u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests(std::numeric_limits<int>::max());
        EXPECT_EQ(kv.second.as<uint32_t>(), static_cast<uint32_t>(std::numeric_limits<int>::max()));
    });
}

// Unsigned types should always be accepted
TEST(PropertyValidation, NumRequestsAcceptsUnsignedTypes) {
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests(uint32_t{0});
        EXPECT_EQ(kv.second.as<uint32_t>(), 0u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests(uint32_t{100});
        EXPECT_EQ(kv.second.as<uint32_t>(), 100u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests(std::numeric_limits<uint32_t>::max());
        EXPECT_EQ(kv.second.as<uint32_t>(), std::numeric_limits<uint32_t>::max());
    });
}

// --- ov::auto_batch_timeout (uint32_t property) ---

TEST(PropertyValidation, AutoBatchTimeoutRejectsNegativeInt) {
    OV_EXPECT_THROW(std::ignore = ov::auto_batch_timeout(-1), ov::Exception, testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::auto_batch_timeout(-1000), ov::Exception, testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::auto_batch_timeout(std::numeric_limits<int>::min()),
                    ov::Exception,
                    testing::HasSubstr("negative"));
}

TEST(PropertyValidation, AutoBatchTimeoutAcceptsZeroAndPositive) {
    OV_ASSERT_NO_THROW({
        auto kv = ov::auto_batch_timeout(0);
        EXPECT_EQ(kv.second.as<uint32_t>(), 0u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::auto_batch_timeout(1000);
        EXPECT_EQ(kv.second.as<uint32_t>(), 1000u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::auto_batch_timeout(uint32_t{5000});
        EXPECT_EQ(kv.second.as<uint32_t>(), 5000u);
    });
}

// --- ov::hint::dynamic_quantization_group_size (uint64_t property) ---

TEST(PropertyValidation, DynamicQuantizationGroupSizeRejectsNegativeInt) {
    OV_EXPECT_THROW(std::ignore = ov::hint::dynamic_quantization_group_size(-1),
                    ov::Exception,
                    testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::hint::dynamic_quantization_group_size(int64_t{-1}),
                    ov::Exception,
                    testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::hint::dynamic_quantization_group_size(std::numeric_limits<int64_t>::min()),
                    ov::Exception,
                    testing::HasSubstr("negative"));
}

TEST(PropertyValidation, DynamicQuantizationGroupSizeAcceptsZeroAndPositive) {
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::dynamic_quantization_group_size(0);
        EXPECT_EQ(kv.second.as<uint64_t>(), 0u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::dynamic_quantization_group_size(128);
        EXPECT_EQ(kv.second.as<uint64_t>(), 128u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::dynamic_quantization_group_size(uint64_t{256});
        EXPECT_EQ(kv.second.as<uint64_t>(), 256u);
    });
}

// --- ov::key_cache_group_size (uint64_t property) ---

TEST(PropertyValidation, KeyCacheGroupSizeRejectsNegativeInt) {
    OV_EXPECT_THROW(std::ignore = ov::key_cache_group_size(-1), ov::Exception, testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::key_cache_group_size(int64_t{-100}),
                    ov::Exception,
                    testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::key_cache_group_size(std::numeric_limits<int>::min()),
                    ov::Exception,
                    testing::HasSubstr("negative"));
}

TEST(PropertyValidation, KeyCacheGroupSizeAcceptsZeroAndPositive) {
    OV_ASSERT_NO_THROW({
        auto kv = ov::key_cache_group_size(0);
        EXPECT_EQ(kv.second.as<uint64_t>(), 0u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::key_cache_group_size(64);
        EXPECT_EQ(kv.second.as<uint64_t>(), 64u);
    });
}

// --- ov::value_cache_group_size (uint64_t property) ---

TEST(PropertyValidation, ValueCacheGroupSizeRejectsNegativeInt) {
    OV_EXPECT_THROW(std::ignore = ov::value_cache_group_size(-1), ov::Exception, testing::HasSubstr("negative"));
    OV_EXPECT_THROW(std::ignore = ov::value_cache_group_size(int64_t{-1}),
                    ov::Exception,
                    testing::HasSubstr("negative"));
}

TEST(PropertyValidation, ValueCacheGroupSizeAcceptsZeroAndPositive) {
    OV_ASSERT_NO_THROW({
        auto kv = ov::value_cache_group_size(0);
        EXPECT_EQ(kv.second.as<uint64_t>(), 0u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::value_cache_group_size(uint64_t{128});
        EXPECT_EQ(kv.second.as<uint64_t>(), 128u);
    });
}

// --- Signed properties: verify they still accept negative values ---

TEST(PropertyValidation, InferenceNumThreadsAcceptsNegative) {
    // int32_t property - negative values are valid
    OV_ASSERT_NO_THROW({
        auto kv = ov::inference_num_threads(-1);
        EXPECT_EQ(kv.second.as<int32_t>(), -1);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::inference_num_threads(std::numeric_limits<int32_t>::min());
        EXPECT_EQ(kv.second.as<int32_t>(), std::numeric_limits<int32_t>::min());
    });
}

TEST(PropertyValidation, CompilationNumThreadsAcceptsNegative) {
    // int32_t property - negative values are valid
    OV_ASSERT_NO_THROW({
        auto kv = ov::compilation_num_threads(-1);
        EXPECT_EQ(kv.second.as<int32_t>(), -1);
    });
}

// --- Overflow tests: values exceeding uint32_t max from int64_t ---

TEST(PropertyValidation, NumRequestsRejectsOverflowFromInt64) {
    // int64_t value larger than uint32_t max should be rejected
    int64_t overflow_value = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(overflow_value), ov::Exception, testing::HasSubstr("exceeds"));
}

TEST(PropertyValidation, NumRequestsAcceptsMaxUint32FromInt64) {
    // int64_t value equal to uint32_t max should be accepted
    int64_t max_value = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests(max_value);
        EXPECT_EQ(kv.second.as<uint32_t>(), std::numeric_limits<uint32_t>::max());
    });
}

TEST(PropertyValidation, AutoBatchTimeoutRejectsOverflowFromInt64) {
    int64_t overflow_value = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    OV_EXPECT_THROW(std::ignore = ov::auto_batch_timeout(overflow_value), ov::Exception, testing::HasSubstr("exceeds"));
}

// uint32_t properties: string "-1" should be rejected
TEST(PropertyValidation, NumRequestsRejectsNegativeString) {
    // String "-1" should behave the same as int -1
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests("-1"), ov::Exception, testing::HasSubstr(""));
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests(std::string{"-1"}), ov::Exception, testing::HasSubstr(""));
    OV_EXPECT_THROW(std::ignore = ov::hint::num_requests("-100"), ov::Exception, testing::HasSubstr(""));
}

TEST(PropertyValidation, AutoBatchTimeoutRejectsNegativeString) {
    OV_EXPECT_THROW(std::ignore = ov::auto_batch_timeout("-1"), ov::Exception, testing::HasSubstr(""));
    OV_EXPECT_THROW(std::ignore = ov::auto_batch_timeout(std::string{"-500"}), ov::Exception, testing::HasSubstr(""));
}

// Positive string values should be accepted
TEST(PropertyValidation, NumRequestsAcceptsPositiveString) {
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests("0");
        EXPECT_EQ(kv.second.as<uint32_t>(), 0u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests("42");
        EXPECT_EQ(kv.second.as<uint32_t>(), 42u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::hint::num_requests(std::string{"100"});
        EXPECT_EQ(kv.second.as<uint32_t>(), 100u);
    });
}

TEST(PropertyValidation, AutoBatchTimeoutAcceptsPositiveString) {
    OV_ASSERT_NO_THROW({
        auto kv = ov::auto_batch_timeout("0");
        EXPECT_EQ(kv.second.as<uint32_t>(), 0u);
    });
    OV_ASSERT_NO_THROW({
        auto kv = ov::auto_batch_timeout("1000");
        EXPECT_EQ(kv.second.as<uint32_t>(), 1000u);
    });
}

}  // namespace ov::test
