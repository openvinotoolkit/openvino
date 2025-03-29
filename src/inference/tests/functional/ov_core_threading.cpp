// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <thread>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "functional_test_utils/test_model/test_model.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/util/file_util.hpp"

#ifdef __GLIBC__
#    include <gnu/libc-version.h>
#    if __GLIBC_MINOR__ < 34
#        define OV_TEST_GLIBC_VERSION_LESS_2_34
#    endif
#endif

class CoreThreadingTests : public ::testing::Test {
protected:
    std::string modelName = "CoreThreadingTests.xml", weightsName = "CoreThreadingTests.bin";

public:
    void SetUp() override {
        auto prefix = ov::test::utils::generateTestFilePrefix();
        modelName = prefix + modelName;
        weightsName = prefix + weightsName;
        ov::test::utils::generate_test_model(modelName, weightsName);
    }

    void TearDown() override {
        ov::test::utils::removeIRFiles(modelName, weightsName);
    }

    void runParallel(std::function<void(void)> func,
                     const unsigned int iterations = 100,
                     const unsigned int threadsNum = 8) {
        std::vector<std::thread> threads(threadsNum);

        for (auto& thread : threads) {
            thread = std::thread([&]() {
                for (unsigned int i = 0; i < iterations; ++i) {
                    func();
                }
            });
        }

        for (auto& thread : threads) {
            if (thread.joinable())
                thread.join();
        }
    }

    void safeAddExtension(ov::Core& core) {
        try {
            auto extension = ov::detail::load_extensions(
                ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                   std::string("openvino_template_extension") + OV_BUILD_POSTFIX));
            core.add_extension(extension);
        } catch (const ov::Exception& ex) {
            ASSERT_STR_CONTAINS(ex.what(), "name: custom_opset. Opset");
        }
    }
};

// tested function: SetConfig
TEST_F(CoreThreadingTests, SetConfigPluginDoesNotExist) {
    ov::Core core;

    runParallel(
        [&]() {
            core.set_property(ov::enable_profiling(true));
        },
        10000);
}

// TODO: CVS-68982
#ifndef OPENVINO_STATIC_LIBRARY

// tested function: RegisterPlugin
TEST_F(CoreThreadingTests, RegisterPlugin) {
    ov::Core core;
    std::atomic<int> index{0};
    runParallel(
        [&]() {
            const std::string deviceName = std::to_string(index++);
            core.register_plugin(ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                                    std::string("mock_engine") + OV_BUILD_POSTFIX),
                                 deviceName);
            core.get_versions(deviceName);
            core.unload_plugin(deviceName);
        },
        4000);
}

// tested function: RegisterPlugins
TEST_F(CoreThreadingTests, RegisterPlugins) {
    ov::Core core;
    std::atomic<unsigned int> index{0};

    auto getPluginXml = [&]() -> std::tuple<std::string, std::string> {
        std::string indexStr = std::to_string(index++);
        std::string pluginsXML = "test_plugins" + indexStr + ".xml";
        std::ofstream file(pluginsXML);

        file << "<ie><plugins><plugin location=\"";
        file << ov::test::utils::getExecutableDirectory();
        file << ov::util::FileTraits<char>::file_separator;
        file << ov::util::FileTraits<char>::library_prefix();
        file << "mock_engine";
        file << OV_BUILD_POSTFIX;
        file << ov::util::FileTraits<char>::dot_symbol;
        file << ov::util::FileTraits<char>::library_ext();
        file << "\" name=\"";
        file << indexStr;
        file << "\"></plugin></plugins></ie>";
        file.flush();
        file.close();

        return std::tie(pluginsXML, indexStr);
    };

    runParallel(
        [&]() {
            std::string fileName, deviceName;
            std::tie(fileName, deviceName) = getPluginXml();
            core.register_plugins(fileName);
            core.get_versions(deviceName);
            ASSERT_EQ(0, std::remove(fileName.c_str()));
        },
        1000);
}

#endif  // !OPENVINO_STATIC_LIBRARY

// tested function: get_available_devices, unload_plugin
// TODO: some initialization (e.g. thread/dlopen) sporadically fails during such stress-test scenario
TEST_F(CoreThreadingTests, GetAvailableDevices) {
#ifdef OV_TEST_GLIBC_VERSION_LESS_2_34
    GTEST_SKIP();
#endif
    ov::Core core;
    runParallel(
        [&]() {
            std::vector<std::string> devices = core.get_available_devices();

            // unregister all the devices
            for (auto&& deviceName : devices) {
                try {
                    core.unload_plugin(deviceName);
                } catch (const ov::Exception& ex) {
                    // if several threads unload plugin at once, the first thread does this
                    // while all others will throw an exception that plugin is not registered
                    ASSERT_STR_CONTAINS(ex.what(), "name is not registered in the");
                }
            }
        },
        30);
}

#if defined(ENABLE_OV_IR_FRONTEND)

namespace ov {
namespace test {
namespace util {
class Barrier {
private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    size_t m_count;
    const size_t m_expected;
    size_t m_wait_id;

public:
    explicit Barrier(std::size_t count) : m_count{count}, m_expected{count}, m_wait_id{} {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (--m_count == 0) {
            ++m_wait_id;
            m_count = m_expected;
            m_cv.notify_all();
        } else {
            const auto wait_id = m_wait_id;
            m_cv.wait(lock, [this, wait_id] {
                return wait_id != m_wait_id;
            });
        }
    }
};
}  // namespace util
}  // namespace test
}  // namespace ov

// tested function: read_model and add_extension
TEST_F(CoreThreadingTests, ReadModel) {
    ov::Core core;
    auto model = core.read_model(modelName, weightsName);

    constexpr size_t threads_num = 12;
    ov::test::util::Barrier sync_point(threads_num);

    runParallel(
        [&]() {
            safeAddExtension(core);
            // Add the extension and read model are thread-safe when use separately.
            // The barrier is required here to wait until all threads add extensions to core before read model.
            // The read_model loads Frontend which check extension vector and assume it want change. If extension vector
            // is expanded then all iterators are invalidated and can result in segfault when frontend check extensions
            // to be added in frontend.
            sync_point.arrive_and_wait();
            std::ignore = core.read_model(modelName, weightsName);

            // sync before next iteration (modification of extensions vector)
            sync_point.arrive_and_wait();
        },
        100,
        threads_num);
}
#endif  // defined(ENABLE_OV_IR_FRONTEND)
