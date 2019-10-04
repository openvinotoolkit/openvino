// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined _WIN32
// Avoidance of Windows.h to include winsock library.
#define _WINSOCKAPI_
// Avoidance of Windows.h to define min/max.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif  // _WIN32

#include "tests_common.hpp"
#include "mock_plugin_dispatcher.hpp"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <string>
#include "ie_plugin_dispatcher.hpp"
#include "ie_plugin_ptr.hpp"
#include "ie_device.hpp"
#include <fstream>

using namespace InferenceEngine;
using namespace ::testing;

class PluginDispatcherTests : public ::testing::Test {
public:
    const std::string nameExt(const std::string& name) { return name + IE_BUILD_POSTFIX;}
};

TEST_F(PluginDispatcherTests, canLoadMockPlugin) {
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

ACTION(ThrowException)
{
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
            std::array<char, 128> buffer;
            std::string result;
            std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
            if (pipe) {
                while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
                    result += buffer.data();
                }
            }
            if (result.size())
                FAIL() << " Visibility is not hidden and there are symbols exposure:\n" << result << std::endl;
        }
    }

}
#endif

#endif
