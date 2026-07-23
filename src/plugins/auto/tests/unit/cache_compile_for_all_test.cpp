// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <common_test_utils/test_constants.hpp>
#include <thread>

#include "include/auto_unit_test.hpp"

using namespace ov::mock_auto_plugin;

// Verify the cache pre-compilation logic in AutoSchedule::init():
// while CPU_HELP + ACTUALDEVICE are compiled in parallel, the remaining candidate devices
// (excluding CPU and the actual device) are also compiled in the background to populate the
// cache blobs, and their compiled models are released right after compilation.
// The pre-compilation only happens when ov::intel_auto::compile_for_all is enabled and a cache
// directory is configured.
using CacheWarmupParams = std::tuple<bool,   // whether cache_dir is enabled
                                     bool>;  // whether ov::intel_auto::compile_for_all is enabled

class AutoCacheCompileForAllTest : public tests::AutoTest, public ::testing::TestWithParam<CacheWarmupParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CacheWarmupParams>& obj) {
        const auto& [cacheEnabled, compileForAll] = obj.param;
        std::ostringstream result;
        result << (cacheEnabled ? "cacheEnabled" : "cacheDisabled") << "_"
               << (compileForAll ? "compileForAll" : "compileForActual");
        return result.str();
    }
};

TEST_P(AutoCacheCompileForAllTest, compileForAllCompilesOtherDevicesWhenEnabled) {
    const auto& [cacheEnabled, compileForAll] = this->GetParam();
    plugin->set_device_name("AUTO");

    const std::string actualDevice = "GPU.1";
    const std::string cpuHelpDevice = ov::test::utils::DEVICE_CPU;
    const std::string otherDevice = "GPU.0";

    // A dedicated mock compiled model for the "other" (warm-up) device so its lifetime can be checked.
    auto mockIExeNetOther = std::make_shared<NiceMock<ov::MockICompiledModel>>(model, mock_plugin_gpu);
    ov::SoPtr<ov::MockICompiledModel> mockExeNetworkOther = {mockIExeNetOther, {}};
    ON_CALL(*mockIExeNetOther.get(), inputs()).WillByDefault(ReturnRefOfCopy(model->inputs()));
    ON_CALL(*mockIExeNetOther.get(), outputs()).WillByDefault(ReturnRefOfCopy(model->outputs()));

    // cache_dir: enabled -> non-empty path, disabled -> empty string.
    const std::string cacheDir = cacheEnabled ? "test_cache_dir" : "";
    ON_CALL(*core, get_property(StrEq(""), StrEq(ov::cache_dir.name()), _))
        .WillByDefault(Return(ov::Any(cacheDir)));

    std::atomic<int> compileCount{0};
    // ACTUALDEVICE (GPU.1)
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq(actualDevice)),
                          _))
        .WillByDefault(InvokeWithoutArgs([this]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return mockExeNetworkActual;
        }));
    // CPU_HELP (CPU)
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq(cpuHelpDevice)),
                          _))
        .WillByDefault(Return(mockExeNetwork));
    // Other device (GPU.0) - the warm-up target.
    ON_CALL(*core,
            compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                          ::testing::Matcher<const std::string&>(StrEq(otherDevice)),
                          _))
        .WillByDefault(InvokeWithoutArgs([&compileCount, mockExeNetworkOther]() {
            compileCount++;
            return mockExeNetworkOther;
        }));

    metaDevices = {{cpuHelpDevice, {}, -1}, {otherDevice, {}, -1}, {actualDevice, {}, -1}};
    ON_CALL(*plugin, parse_meta_devices(_, _)).WillByDefault(Return(metaDevices));
    ON_CALL(*plugin, get_valid_device)
        .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
            std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
            return devices;
        });
    // Always select GPU.1 as the actual device.
    ON_CALL(*plugin, select_device(_, _, _)).WillByDefault(Return(metaDevices[2]));

    config.insert(ov::device::priorities(cpuHelpDevice + std::string(",") + otherDevice + std::string(",") + actualDevice));
    config.insert(ov::intel_auto::compile_for_all(compileForAll));

    const auto otherBaselineCount = mockIExeNetOther.use_count();

    std::shared_ptr<ov::ICompiledModel> exeNetwork;
    OV_ASSERT_NO_THROW(exeNetwork = plugin->compile_model(model, config));

    // Give the background warm-up task time to finish compiling and releasing.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    if (cacheEnabled && compileForAll) {
        EXPECT_GE(compileCount.load(), 1);
        // The warm-up compiled model must be released, so no extra reference is kept.
        EXPECT_EQ(mockIExeNetOther.use_count(), otherBaselineCount);
    } else {
        EXPECT_EQ(compileCount.load(), 0);
    }
}

const std::vector<CacheWarmupParams> testConfigs = {CacheWarmupParams{true, true},
                                                    CacheWarmupParams{true, false},
                                                    CacheWarmupParams{false, true},
                                                    CacheWarmupParams{false, false}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         AutoCacheCompileForAllTest,
                         ::testing::ValuesIn(testConfigs),
                         AutoCacheCompileForAllTest::getTestCaseName);
