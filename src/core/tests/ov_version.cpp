// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/ov_version.hpp"

#include <gtest/gtest.h>

#include "openvino/core/except.hpp"

namespace ov::test {

// Test fixture for Version tests
class VersionTest : public ::testing::Test {};

// Test fixture for version compatibility tests
class VersionCompatibilityTest : public ::testing::Test {};

// ============================================================================
// Version Parsing Tests
// ============================================================================

TEST_F(VersionTest, ParseFullVersionString) {
    ov::util::Version version("2024.5.0-12345-abcdef");
    EXPECT_EQ(version.major, 2024);
    EXPECT_EQ(version.minor, 5);
    EXPECT_EQ(version.patch, 0);
    EXPECT_EQ(version.tweak, 0);
    EXPECT_EQ(version.build, 12345);
}

TEST_F(VersionTest, ParseVersionWithTweak) {
    ov::util::Version version("2024.5.0.1-12345-abcdef");
    EXPECT_EQ(version.major, 2024);
    EXPECT_EQ(version.minor, 5);
    EXPECT_EQ(version.patch, 0);
    EXPECT_EQ(version.tweak, 1);
    EXPECT_EQ(version.build, 12345);
}

TEST_F(VersionTest, ParseVersionWithLargeNumbers) {
    ov::util::Version version("2025.10.99.999-99999-something");
    EXPECT_EQ(version.major, 2025);
    EXPECT_EQ(version.minor, 10);
    EXPECT_EQ(version.patch, 99);
    EXPECT_EQ(version.tweak, 999);
    EXPECT_EQ(version.build, 99999);
}

TEST_F(VersionTest, ParseBuildNumberOnly) {
    ov::util::Version version("12345");
    EXPECT_EQ(version.major, 0);
    EXPECT_EQ(version.minor, 0);
    EXPECT_EQ(version.patch, 0);
    EXPECT_EQ(version.tweak, 0);
    EXPECT_EQ(version.build, 12345);
}

TEST_F(VersionTest, ParseVersionFromString) {
    std::string version_str = "2024.1.2-5000-test";
    ov::util::Version version(version_str);
    EXPECT_EQ(version.major, 2024);
    EXPECT_EQ(version.minor, 1);
    EXPECT_EQ(version.patch, 2);
    EXPECT_EQ(version.tweak, 0);
    EXPECT_EQ(version.build, 5000);
}

TEST_F(VersionTest, ParseVersionFromConstCharPtr) {
    const char* version_str = "2024.1.2-5000-test";
    ov::util::Version version(version_str);
    EXPECT_EQ(version.major, 2024);
    EXPECT_EQ(version.minor, 1);
    EXPECT_EQ(version.patch, 2);
    EXPECT_EQ(version.tweak, 0);
    EXPECT_EQ(version.build, 5000);
}

TEST_F(VersionTest, ParseVersionFromCharArray) {
    char version_str[] = "2024.1.2-5000-test";
    ov::util::Version version(version_str);
    EXPECT_EQ(version.major, 2024);
    EXPECT_EQ(version.minor, 1);
    EXPECT_EQ(version.patch, 2);
    EXPECT_EQ(version.tweak, 0);
    EXPECT_EQ(version.build, 5000);
}

TEST_F(VersionTest, ParseVersionFromStringView) {
    std::string_view version_str = "2024.1.2-5000-test";
    ov::util::Version version(version_str);
    EXPECT_EQ(version.major, 2024);
    EXPECT_EQ(version.minor, 1);
    EXPECT_EQ(version.patch, 2);
    EXPECT_EQ(version.tweak, 0);
    EXPECT_EQ(version.build, 5000);
}

TEST_F(VersionTest, ParseVersionFromTemporaryString) {
    ov::util::Version version(std::string("2024.1.2-5000-test"));
    EXPECT_EQ(version.major, 2024);
    EXPECT_EQ(version.minor, 1);
    EXPECT_EQ(version.patch, 2);
    EXPECT_EQ(version.tweak, 0);
    EXPECT_EQ(version.build, 5000);
}

TEST_F(VersionTest, ParseVersionWithZeros) {
    ov::util::Version version("0.0.0-0-test");
    EXPECT_EQ(version.major, 0);
    EXPECT_EQ(version.minor, 0);
    EXPECT_EQ(version.patch, 0);
    EXPECT_EQ(version.tweak, 0);
    EXPECT_EQ(version.build, 0);
}

TEST_F(VersionTest, ParseVersionWithComplexSuffix) {
    ov::util::Version version("2024.5.0-12345-rc1-something-else");
    EXPECT_EQ(version.major, 2024);
    EXPECT_EQ(version.minor, 5);
    EXPECT_EQ(version.patch, 0);
    EXPECT_EQ(version.build, 12345);
}

TEST_F(VersionTest, ThrowsOnInvalidFormat) {
    EXPECT_THROW(ov::util::Version("invalid"), ov::Exception);
    EXPECT_THROW(ov::util::Version("2024.5"), ov::Exception);
    EXPECT_THROW(ov::util::Version("2024.5.0"), ov::Exception);
    EXPECT_THROW(ov::util::Version("2024.5.0-"), ov::Exception);
    EXPECT_THROW(ov::util::Version("a.b.c-1-test"), ov::Exception);
    EXPECT_THROW(ov::util::Version("2024.5.0-abc-test"), ov::Exception);
}

TEST_F(VersionTest, ThrowsOnEmptyString) {
    EXPECT_THROW(ov::util::Version(""), ov::Exception);
}

// ============================================================================
// Version Comparison Tests
// ============================================================================

TEST_F(VersionTest, EqualityOperator) {
    ov::util::Version v1("2024.5.0-12345-test");
    ov::util::Version v2("2024.5.0-12345-test");
    ov::util::Version v3("2024.5.0-12346-test");

    EXPECT_TRUE(v1 == v2);
    EXPECT_FALSE(v1 == v3);
}

TEST_F(VersionTest, InequalityOperator) {
    ov::util::Version v1("2024.5.0-12345-test");
    ov::util::Version v2("2024.5.0-12345-test");
    ov::util::Version v3("2024.5.0-12346-test");

    EXPECT_FALSE(v1 != v2);
    EXPECT_TRUE(v1 != v3);
}

TEST_F(VersionTest, LessThanOperator) {
    ov::util::Version v1("2024.5.0-12345-test");
    ov::util::Version v2("2024.5.0-12346-test");
    ov::util::Version v3("2024.5.1-12345-test");
    ov::util::Version v4("2024.6.0-12345-test");
    ov::util::Version v5("2025.5.0-12345-test");

    EXPECT_TRUE(v1 < v2);
    EXPECT_TRUE(v1 < v3);
    EXPECT_TRUE(v1 < v4);
    EXPECT_TRUE(v1 < v5);
    EXPECT_FALSE(v2 < v1);
}

TEST_F(VersionTest, GreaterThanOperator) {
    ov::util::Version v1("2024.5.0-12345-test");
    ov::util::Version v2("2024.5.0-12346-test");

    EXPECT_TRUE(v2 > v1);
    EXPECT_FALSE(v1 > v2);
    EXPECT_FALSE(v1 > v1);
}

TEST_F(VersionTest, LessThanOrEqualOperator) {
    ov::util::Version v1("2024.5.0-12345-test");
    ov::util::Version v2("2024.5.0-12345-test");
    ov::util::Version v3("2024.5.0-12346-test");

    EXPECT_TRUE(v1 <= v2);
    EXPECT_TRUE(v1 <= v3);
    EXPECT_FALSE(v3 <= v1);
}

TEST_F(VersionTest, GreaterThanOrEqualOperator) {
    ov::util::Version v1("2024.5.0-12345-test");
    ov::util::Version v2("2024.5.0-12345-test");
    ov::util::Version v3("2024.5.0-12346-test");

    EXPECT_TRUE(v1 >= v2);
    EXPECT_TRUE(v3 >= v1);
    EXPECT_FALSE(v1 >= v3);
}

TEST_F(VersionTest, ComparisonWithTweak) {
    ov::util::Version v1("2024.5.0.0-12345-test");
    ov::util::Version v2("2024.5.0.1-12345-test");

    EXPECT_TRUE(v1 < v2);
    EXPECT_FALSE(v1 > v2);
}

TEST_F(VersionTest, ComparisonPriorityOrder) {
    // Major takes precedence
    ov::util::Version v1("2023.99.99.99-99999-test");
    ov::util::Version v2("2024.0.0.0-0-test");
    EXPECT_TRUE(v1 < v2);

    // Minor takes precedence over patch
    ov::util::Version v3("2024.5.99.99-99999-test");
    ov::util::Version v4("2024.6.0.0-0-test");
    EXPECT_TRUE(v3 < v4);

    // Patch takes precedence over tweak
    ov::util::Version v5("2024.5.0.99-99999-test");
    ov::util::Version v6("2024.5.1.0-0-test");
    EXPECT_TRUE(v5 < v6);

    // Tweak takes precedence over build
    ov::util::Version v7("2024.5.0.0-99999-test");
    ov::util::Version v8("2024.5.0.1-0-test");
    EXPECT_TRUE(v7 < v8);
}

// ============================================================================
// Version Compatibility Tests
// ============================================================================

TEST_F(VersionCompatibilityTest, IdenticalVersionsAreCompatible) {
    ov::util::Version v1("2024.5.0-12345-test");
    ov::util::Version v2("2024.5.0-12345-test");

    EXPECT_TRUE(ov::util::is_version_compatible(v1, v2));
}

TEST_F(VersionCompatibilityTest, NewerVersionIsNotCompatible) {
    ov::util::Version older("2024.5.0-12345-test");
    ov::util::Version newer("2024.6.0-12345-test");

    EXPECT_FALSE(ov::util::is_version_compatible(newer, older));
}

TEST_F(VersionCompatibilityTest, DefaultPolicy) {
    ov::util::Version older("2024.5.0-12345-test");
    ov::util::Version newer("2024.6.0-12345-test");

    // Default policy has all max_diff = 0, so only identical versions are compatible
    EXPECT_TRUE(ov::util::is_version_compatible(older, newer));
    EXPECT_FALSE(ov::util::is_version_compatible(newer, older));
    EXPECT_TRUE(ov::util::is_version_compatible(older, older));
}

TEST_F(VersionCompatibilityTest, AnyDifference) {
    ov::util::Version older("2020.0.0-0-test");
    ov::util::Version newer("2024.5.10-12345-test");

    ov::util::VersionCompatibilityPolicy policy{SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};

    EXPECT_TRUE(ov::util::is_version_compatible(older, newer, policy));
}

TEST_F(VersionCompatibilityTest, MajorVersionDifferenceRespected) {
    ov::util::Version older("2022.5.0-12345-test");
    ov::util::Version newer("2024.5.0-12345-test");

    EXPECT_FALSE(ov::util::is_version_compatible(older, newer, {1}));
    EXPECT_TRUE(ov::util::is_version_compatible(older, newer, {2}));
}

TEST_F(VersionCompatibilityTest, MinorVersionDifferenceRespected) {
    ov::util::Version older("2024.3.0-12345-test");
    ov::util::Version newer("2024.5.0-12345-test");

    EXPECT_FALSE(ov::util::is_version_compatible(older, newer, {0, 1}));
    EXPECT_TRUE(ov::util::is_version_compatible(older, newer, {0, 2}));
}

TEST_F(VersionCompatibilityTest, PatchVersionDifferenceRespected) {
    ov::util::Version older("2024.5.0-12345-test");
    ov::util::Version newer("2024.5.3-12345-test");

    EXPECT_FALSE(ov::util::is_version_compatible(older, newer, {0, 0, 2}));
    EXPECT_TRUE(ov::util::is_version_compatible(older, newer, {0, 0, 3}));
}

TEST_F(VersionCompatibilityTest, TweakVersionDifferenceRespected) {
    ov::util::Version older("2024.5.0.0-12345-test");
    ov::util::Version newer("2024.5.0.3-12345-test");

    EXPECT_FALSE(ov::util::is_version_compatible(older, newer, {0, 0, 0, 2}));
    EXPECT_TRUE(ov::util::is_version_compatible(older, newer, {0, 0, 0, 3}));
}

TEST_F(VersionCompatibilityTest, BuildDifferenceRespected) {
    ov::util::Version older("2024.5.0-12340-test");
    ov::util::Version newer("2024.5.0-12345-test");

    EXPECT_FALSE(ov::util::is_version_compatible(older, newer, {0, 0, 0, 0, 4}));
    EXPECT_TRUE(ov::util::is_version_compatible(older, newer, {0, 0, 0, 0, 5}));
}

TEST_F(VersionCompatibilityTest, NotSpecifiedDifferencesAreUnlimited) {
    ov::util::Version v1("2024.5.0-5-test");
    ov::util::Version v2("2024.6.5-0-test");

    // Only major and minor differences are specified. Difference in patch, tweak, build don't matter.
    EXPECT_TRUE(ov::util::is_version_compatible(v1, v2, {0, 1}));
    EXPECT_FALSE(ov::util::is_version_compatible(v2, v1, {0, 1}));
}

TEST_F(VersionCompatibilityTest, StrictPolicyNoToleranceExceptExact) {
    ov::util::Version older("2024.5.0-12345-test");
    ov::util::Version newer("2024.5.0-12345-test");

    ov::util::VersionCompatibilityPolicy strict_policy{0, 0, 0, 0, 0};

    EXPECT_TRUE(ov::util::is_version_compatible(older, newer, strict_policy));

    ov::util::Version older2("2024.5.0-12344-test");
    EXPECT_FALSE(ov::util::is_version_compatible(older2, newer, strict_policy));
}

TEST_F(VersionCompatibilityTest, ComplexPolicyMultipleConstraints) {
    ov::util::Version older("2024.3.0-12340-test");
    ov::util::Version newer("2024.5.2-12345-test");

    ov::util::VersionCompatibilityPolicy policy{0, 2, 2, SIZE_MAX, 5};

    EXPECT_TRUE(ov::util::is_version_compatible(older, newer, policy));

    // Exceeds minor difference
    ov::util::Version older2("2024.2.0-12340-test");
    EXPECT_FALSE(ov::util::is_version_compatible(older2, newer, policy));

    // Different major version
    ov::util::Version older3("2023.5.0-12340-test");
    EXPECT_FALSE(ov::util::is_version_compatible(older3, newer, policy));
}

TEST_F(VersionCompatibilityTest, BuildOnlyVersions) {
    ov::util::Version older("12340");
    ov::util::Version newer("12345");

    EXPECT_TRUE(ov::util::is_version_compatible(older, newer, {0, 0, 0, 0, 5}));
    EXPECT_FALSE(ov::util::is_version_compatible(older, newer, {0, 0, 0, 0, 4}));
}
}  // namespace ov::test
