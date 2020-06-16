// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined _WIN32
// Avoidance of Windows.h to define min/max.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif  // _WIN32

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <string>
#include <fstream>

#include "ie_plugin_dispatcher.hpp"

#include "unit_test_utils/mocks/mock_plugin_dispatcher.hpp"

using namespace InferenceEngine;
using namespace ::testing;

// Add DISABLED_ prefix to test name when building with -fsanitize=address
#if defined(__SANITIZE_ADDRESS__) || (defined(__clang__) && __has_feature(address_sanitizer))
#define DISABLE_IF_SANITIZER(TEST_NAME) DISABLED_ ## TEST_NAME
#else
#define DISABLE_IF_SANITIZER(TEST_NAME) TEST_NAME
#endif

class PluginDispatcherTests : public ::testing::Test {
public:
    const std::string nameExt(const std::string& name) { return name + IE_BUILD_POSTFIX;}
};

// The test is disabled for SANITIZER builds due to known issue in the test code:
// The module unloaded before static holder object (BuiltInShapeInferHolder::ImplsHolder) was destroyed.
// IShapeInferExtension mechanics is deprecated.
IE_SUPPRESS_DEPRECATED_START
TEST_F(PluginDispatcherTests, DISABLE_IF_SANITIZER(canLoadMockPlugin)) {
    PluginDispatcher dispatcher({ "", "./", "./lib" });
    ASSERT_NO_THROW(dispatcher.getPluginByName(nameExt("mock_engine")));
}

#if defined _WIN32

class SetDllDirectoryCaller {
public:
    /// Call SetDllDirectory if not called before
    SetDllDirectoryCaller(const char* path) {
        // Check if user already called SetDllDirectory with acctual directory
        call_setdlldirectory = (1 >= GetDllDirectory(0, nullptr));
        if (call_setdlldirectory) {
            SetDllDirectory(path);
        }
    }
    /// Restore serch path order to default
    ~SetDllDirectoryCaller() {
        if (call_setdlldirectory)
            SetDllDirectory(nullptr);
    }

    bool call_setdlldirectory;

    // Non copyable or movable
    SetDllDirectoryCaller(const SetDllDirectoryCaller&) = delete;
    SetDllDirectoryCaller& operator=(const SetDllDirectoryCaller&) = delete;
};

TEST_F(PluginDispatcherTests, canLoadMockPluginAndRetainSetDllDirectory) {
    // a test pre-requisite that SetDllDirectory is not configured
    ASSERT_EQ(1, GetDllDirectory(0, nullptr));

    // try modify DLL search order with SetDllDirectory
    const char *set_dir = "12345";
    char get_dir[6] = {0};
    SetDllDirectoryCaller set_dll_directory_caller(set_dir);

    PluginDispatcher dispatcher({ "", "./", "./lib" });
    ASSERT_NO_THROW(dispatcher.getPluginByName(nameExt("mock_engine")));

    // verify DLL search order retained
    ASSERT_EQ(sizeof(get_dir), GetDllDirectory(0, nullptr));
    ASSERT_NE(0, GetDllDirectory(sizeof(get_dir), get_dir));
    ASSERT_EQ(std::string(get_dir), std::string(set_dir));
}

TEST_F(PluginDispatcherTests, canLoadMockPluginAndKeepDefaultDLLSearch) {
    // a test pre-requisite that SetDllDirectory is not configured
    ASSERT_EQ(1, GetDllDirectory(0, nullptr));

    PluginDispatcher dispatcher({ "", "./", "./lib" });
    ASSERT_NO_THROW(dispatcher.getPluginByName(nameExt("mock_engine")));

    // verify DLL search order is still default
    ASSERT_EQ(1, GetDllDirectory(0, nullptr));
}
#endif

TEST_F(PluginDispatcherTests, throwsOnUnknownPlugin) {
    PluginDispatcher dispatcher({ "./", "./lib" });
    ASSERT_THROW(dispatcher.getPluginByName(nameExt("unknown_plugin")), InferenceEngine::details::InferenceEngineException);
}

ACTION(ThrowException) {
    THROW_IE_EXCEPTION << "Exception!";
}

#if defined(ENABLE_MKL_DNN)
TEST_F(PluginDispatcherTests, returnsIfLoadSuccessfull) {
    MockDispatcher disp({ "./", "./lib" });
    PluginDispatcher dispatcher({ "", "./", "./lib" });
    auto ptr = dispatcher.getPluginByName(nameExt("mock_engine"));

    EXPECT_CALL(disp, getPluginByName(_)).WillOnce(Return(ptr));
    ASSERT_NO_THROW(disp.getPluginByName(nameExt("MKLDNNPlugin")));
}

#if defined ENABLE_MKL_DNN && !defined _WIN32 && !defined __CYGWIN__ && !defined __APPLE__
TEST_F(PluginDispatcherTests, libMKLDNNPluginSymbolsExposure) {
    std::vector<std::string> locations = {"/libMKLDNNPlugin.so", "/lib/libMKLDNNPlugin.so"};
    char path[PATH_MAX];
    if (readlink("/proc/self/exe", path, sizeof(path)/sizeof(path[0])) < 0) {
        return;
    }

    std::string Path = path;
    for (auto location : locations) {
        std::string fullPath = Path.substr(0, Path.find_last_of("/")) + location;
        if (std::ifstream(fullPath.c_str()).good()) {
            std::string command = "readelf --dyn-syms ";
            command += fullPath;
            command += " | grep UNIQUE | c++filt";
            char buffer[128];
            std::string result;
            std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
            if (pipe) {
                while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
                    result += buffer;
                }
            }
            if (result.size())
                FAIL() << " Visibility is not hidden and there are symbols exposure:\n" << result << std::endl;
        }
    }
}
#endif

#endif

IE_SUPPRESS_DEPRECATED_END
