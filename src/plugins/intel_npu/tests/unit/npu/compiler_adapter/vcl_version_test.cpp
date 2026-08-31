// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>

#include "intel_npu/utils/vcl/vcl.h"
#include "openvino/core/except.hpp"
#include "vcl_version_utils.hpp"

namespace {

using ::intel_npu::vcl_version_utils::checkVclVersion;
using ::intel_npu::vcl_version_utils::getUsedVclVersion;
using ::intel_npu::vcl_version_utils::UsedVersion;

// The floor the plugin was built against.
constexpr uint16_t kFloorMajor = VCL_COMPILER_VERSION_MAJOR;
constexpr uint16_t kFloorMinor = VCL_COMPILER_VERSION_MINOR;

// Convenience wrapper: negotiate against the real plugin floor, then validate.
void negotiateAndCheck(uint16_t loadedMajor, uint16_t loadedMinor) {
    const UsedVersion used = getUsedVclVersion(kFloorMajor, kFloorMinor, loadedMajor, loadedMinor);
    checkVclVersion(used, loadedMajor, loadedMinor, kFloorMajor, kFloorMinor);
}

//
// getUsedVclVersion — version negotiation
//

TEST(VclVersionTests, SameMajorTakesTheLowerMinor) {
    // Plugin minor is the lower one.
    EXPECT_EQ(getUsedVclVersion(7, 10, 7, 20).Major, 7);
    EXPECT_EQ(getUsedVclVersion(7, 10, 7, 20).Minor, 10);

    // Loaded minor is the lower one.
    EXPECT_EQ(getUsedVclVersion(7, 20, 7, 10).Major, 7);
    EXPECT_EQ(getUsedVclVersion(7, 20, 7, 10).Minor, 10);
}

TEST(VclVersionTests, SameMajorAndMinorIsUnchanged) {
    const UsedVersion used = getUsedVclVersion(7, 10, 7, 10);
    EXPECT_EQ(used.Major, 7);
    EXPECT_EQ(used.Minor, 10);
}

TEST(VclVersionTests, PluginMajorNewerThanLibraryDowngradesToLibrary) {
    // Plugin 9.3 against library 7.20 -> speak the library's 7.20, minor included.
    const UsedVersion used = getUsedVclVersion(9, 3, 7, 20);
    EXPECT_EQ(used.Major, 7);
    EXPECT_EQ(used.Minor, 20);
}

TEST(VclVersionTests, PluginMajorOlderThanLibraryKeepsPluginVersion) {
    // Plugin 7.10 against library 9.3 -> keep 7.10; the library is expected to be
    // backward compatible.
    const UsedVersion used = getUsedVclVersion(7, 10, 9, 3);
    EXPECT_EQ(used.Major, 7);
    EXPECT_EQ(used.Minor, 10);
}

//
// checkVclVersion — floor enforcement
//

TEST(VclVersionTests, VersionAtTheFloorIsAccepted) {
    EXPECT_NO_THROW(checkVclVersion(UsedVersion{7, 10}, 7, 10, 7, 10));
}

TEST(VclVersionTests, VersionAboveTheFloorIsAccepted) {
    EXPECT_NO_THROW(checkVclVersion(UsedVersion{7, 11}, 7, 11, 7, 10));
    EXPECT_NO_THROW(checkVclVersion(UsedVersion{8, 0}, 8, 0, 7, 10));
}

TEST(VclVersionTests, MinorBelowTheFloorThrows) {
    EXPECT_THROW(checkVclVersion(UsedVersion{7, 9}, 7, 9, 7, 10), ov::Exception);
}

TEST(VclVersionTests, MajorBelowTheFloorThrows) {
    EXPECT_THROW(checkVclVersion(UsedVersion{6, 99}, 6, 99, 7, 10), ov::Exception);
}

TEST(VclVersionTests, ThrowMessageReportsLoadedAndRequiredVersions) {
    try {
        checkVclVersion(UsedVersion{6, 1}, 6, 1, 7, 10);
        FAIL() << "Expected checkVclVersion to throw";
    } catch (const ov::Exception& error) {
        const std::string what = error.what();
        EXPECT_NE(what.find("Unsupported VCL version: 6.1"), std::string::npos) << what;
        EXPECT_NE(what.find("please use VCL 7.10 or later"), std::string::npos) << what;
    }
}

//
// Negotiation and enforcement together, against the real plugin floor
//

TEST(VclVersionTests, LibraryOlderThanTheFloorIsRejected) {
    // Negotiation downgrades to the library's version, which then fails the floor check.
    ASSERT_GT(kFloorMinor, 0u) << "test assumes a non-zero plugin floor minor";
    EXPECT_THROW(negotiateAndCheck(kFloorMajor, static_cast<uint16_t>(kFloorMinor - 1)), ov::Exception);
    EXPECT_THROW(negotiateAndCheck(static_cast<uint16_t>(kFloorMajor - 1), 99), ov::Exception);
}

TEST(VclVersionTests, LibraryAtOrNewerThanTheFloorIsAccepted) {
    EXPECT_NO_THROW(negotiateAndCheck(kFloorMajor, kFloorMinor));
    EXPECT_NO_THROW(negotiateAndCheck(kFloorMajor, static_cast<uint16_t>(kFloorMinor + 1)));
    EXPECT_NO_THROW(negotiateAndCheck(static_cast<uint16_t>(kFloorMajor + 1), 0));
}

}  // namespace
