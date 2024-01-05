// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <thread>
#include <future>
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>

class ThreadLocalTest : public ::testing::Test, public ::testing::WithParamInterface<std::string>
{
public:
    void SetUp()
    {
        target_device = GetParam();
    }

    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj)
    {
        const auto layerName = obj.param;
        std::ostringstream result;

        result << layerName;
        return result.str();
    }

public:
    std::string target_device = "";
};

TEST(LoadLibraryTest, load_free_library)
{
    std::promise<void> free_promise;
    std::future<void> free_future = free_promise.get_future();
    std::promise<void> thread_exit_promise;
    std::future<void> thread_exit_future = thread_exit_promise.get_future();
    HMODULE shared_object = LoadLibraryA("openvino.dll");

    std::thread sub_thread = std::thread([&]
                                         {
        free_promise.set_value();
        thread_exit_future.get(); });
    free_future.get();
    FreeLibrary(shared_object);

    thread_exit_promise.set_value();
    if (sub_thread.joinable())
    {
        sub_thread.join();
    }
}

typedef void (*TEST_FUNC)(std::string);

TEST_P(ThreadLocalTest, get_property_test)
{
    HMODULE shared_object = LoadLibraryA("ov_thread_local.dll");
    TEST_FUNC procAddr = reinterpret_cast<TEST_FUNC>(GetProcAddress(shared_object, "core_get_property_test"));
    procAddr(target_device);
    FreeLibrary(shared_object);
}

TEST_P(ThreadLocalTest, infer_test)
{
    HMODULE shared_object = LoadLibraryA("ov_thread_local.dll");
    TEST_FUNC procAddr = reinterpret_cast<TEST_FUNC>(GetProcAddress(shared_object, "core_infer_test"));
    procAddr(target_device);
    FreeLibrary(shared_object);
}

std::vector<std::string> test_device = {"CPU", "GPU"};

INSTANTIATE_TEST_SUITE_P(OV_ThreadLocalTests,
                         ThreadLocalTest, ::testing::ValuesIn(test_device),
                         ThreadLocalTest::getTestCaseName);
