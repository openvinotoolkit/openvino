// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/core.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/file_utils.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define RETURN_STATIC_
#    undef OPENVINO_STATIC_LIBRARY  // Skip include of "ie_plugins.hpp" during include of "ie_core.cpp"
#endif  // OPENVINO_STATIC_LIBRARY

#include "ie_core.cpp"

#ifdef RETURN_STATIC_
#    define OPENVINO_STATIC_LIBRARY
#endif  // STATIC_

using namespace testing;
using namespace ov::util;

TEST(CoreTests_getPluginPathFromXML, UseAbsPathAsIs) {
    auto libPath = ov::util::get_absolute_file_path("test_name.ext", false);
    auto absPath = from_file_path(ov::getPluginPathFromXML(libPath));
    EXPECT_TRUE(is_absolute_file_path(absPath));
    EXPECT_STREQ(absPath.c_str(), libPath.c_str());
}

TEST(CoreTests_getPluginPathFromXML, CovertRelativePathAsRelativeToLibDir) {
    auto libPath = "test_name.ext";
    auto absPath = from_file_path(ov::getPluginPathFromXML(libPath));
    EXPECT_TRUE(is_absolute_file_path(absPath));

    auto refPath =
        from_file_path(FileUtils::makePath(InferenceEngine::getInferenceEngineLibraryPath(), to_file_path(libPath)));
    EXPECT_STREQ(absPath.c_str(), refPath.c_str());
}

TEST(CoreTests_getPluginPath, UseAbsPathAsIs) {
    auto libName = from_file_path(FileUtils::makePluginLibraryName({}, to_file_path("test_name")));  // libtest_name.so
    auto libPath = ov::util::get_absolute_file_path(libName, false);
    auto absPath = from_file_path(ov::getPluginPath(libPath));
    EXPECT_TRUE(is_absolute_file_path(absPath));
    EXPECT_STREQ(absPath.c_str(), libPath.c_str());
}

TEST(CoreTests_getPluginPath, RelativePathIsFromWorkDir) {
    auto libName = from_file_path(FileUtils::makePluginLibraryName({}, to_file_path("test_name")));  // libtest_name.so
    auto absPath = from_file_path(ov::getPluginPath(libName));
    EXPECT_TRUE(is_absolute_file_path(absPath));
    EXPECT_STREQ(absPath.c_str(), get_absolute_file_path(libName, false).c_str());
}

TEST(CoreTests_getPluginPath, ConvertNameToAbsPathFromLibDir) {
    auto libName = "test_name.ext";
    auto absPath = from_file_path(ov::getPluginPath(libName));  // libtestname.ext.so
    EXPECT_TRUE(is_absolute_file_path(absPath));

    auto refPath = from_file_path(
        FileUtils::makePluginLibraryName(InferenceEngine::getInferenceEngineLibraryPath(), to_file_path(libName)));
    EXPECT_STREQ(absPath.c_str(), refPath.c_str());
}

TEST(CoreTests_getPluginPathFromLibDir, PathIsFromLibDir) {
    auto libName = "test_name";
    auto absPath = from_file_path(ov::getFilePathFromLibDir(libName));
    EXPECT_TRUE(is_absolute_file_path(absPath));

    auto refPath =
        from_file_path(FileUtils::makePath(InferenceEngine::getInferenceEngineLibraryPath(), to_file_path(libName)));
    EXPECT_STREQ(absPath.c_str(), refPath.c_str());
}