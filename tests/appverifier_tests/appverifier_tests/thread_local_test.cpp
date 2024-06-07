// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <windows.h>

#include <future>
#include <thread>

typedef void (*TestFunc)(const std::string&);

class ThreadLocalTest : public ::testing::Test, public ::testing::WithParamInterface<std::string> {
public:
    void SetUp() {
        target_device = GetParam();
    }

    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
        return obj.param;
    }

public:
    std::string target_device = "";
};

TEST(LoadLibraryTest, load_free_library) {
    std::promise<void> free_promise;
    std::future<void> free_future = free_promise.get_future();
    std::promise<void> thread_exit_promise;
    std::future<void> thread_exit_future = thread_exit_promise.get_future();
    auto shared_object = LoadLibraryA("openvino.dll");
    if (!shared_object) {
        std::cout << "LoadLibrary openvino.dll fail" << std::endl;
        return;
    }

    std::thread sub_thread = std::thread([&] {
        free_promise.set_value();
        thread_exit_future.get();
    });
    free_future.get();
    FreeLibrary(shared_object);

    thread_exit_promise.set_value();
    if (sub_thread.joinable()) {
        sub_thread.join();
    }
}

TEST_P(ThreadLocalTest, get_property_test) {
    auto shared_object = LoadLibraryA("ov_thread_local.dll");
    if (!shared_object) {
        std::cout << "LoadLibrary ov_thread_local.dll fail" << std::endl;
        return;
    }
    auto procAddr = reinterpret_cast<TestFunc>(GetProcAddress(shared_object, "core_get_property_test"));
    procAddr(target_device);
    FreeLibrary(shared_object);
}

TEST_P(ThreadLocalTest, infer_test) {
    auto shared_object = LoadLibraryA("ov_thread_local.dll");
    if (!shared_object) {
        std::cout << "LoadLibrary ov_thread_local.dll fail" << std::endl;
        return;
    }
    auto procAddr = reinterpret_cast<TestFunc>(GetProcAddress(shared_object, "core_infer_test"));
    procAddr(target_device);
    FreeLibrary(shared_object);
}

void process_sub_thread(const std::string& func_name, const std::string& target_device) {
    auto shared_object = LoadLibraryA("ov_thread_local.dll");
    if (!shared_object) {
        std::cout << "LoadLibrary ov_thread_local.dll fail" << std::endl;
        return;
    }
    auto procAddr = reinterpret_cast<TestFunc>(GetProcAddress(shared_object, func_name.c_str()));
    std::promise<void> free_promise;
    std::future<void> free_future = free_promise.get_future();
    std::promise<void> thread_exit_promise;
    std::future<void> thread_exit_future = thread_exit_promise.get_future();
    std::thread sub_thread = std::thread([&] {
        procAddr(target_device);
        free_promise.set_value();
        thread_exit_future.get();
    });

    free_future.get();
    FreeLibrary(shared_object);

    thread_exit_promise.set_value();
    if (sub_thread.joinable()) {
        sub_thread.join();
    }
}

TEST_P(ThreadLocalTest, get_property_test_subthread) {
    process_sub_thread("core_get_property_test", target_device);
}

TEST_P(ThreadLocalTest, infer_test_subthread) {
    process_sub_thread("core_infer_test", target_device);
}

INSTANTIATE_TEST_SUITE_P(OV_ThreadLocalTests,
                         ThreadLocalTest,
                         ::testing::Values("CPU", "GPU"),
                         ThreadLocalTest::getTestCaseName);
