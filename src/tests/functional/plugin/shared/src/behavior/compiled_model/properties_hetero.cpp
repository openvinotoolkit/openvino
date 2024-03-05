// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties_hetero.hpp"

namespace ov {
namespace test {
namespace behavior {

void OVClassHeteroCompiledModelGetMetricTest::SetCpuAffinity(ov::Core& core, std::vector<std::string>& expectedTargets) {
#ifdef ENABLE_INTEL_CPU
    auto layermap = core.query_model(actualNetwork, heteroDeviceName);
    for (auto &iter : layermap) {
        if (iter.first.find("Concat") != std::string::npos)
            layermap[iter.first] = ov::test::utils::DEVICE_CPU;
    }
    for (auto& node : actualNetwork->get_ops()) {
        auto affinity = layermap[node->get_friendly_name()];
        node->get_rt_info()["affinity"] = affinity;
    }
    if (target_device.find(ov::test::utils::DEVICE_CPU) == std::string::npos)
        expectedTargets = {target_device, ov::test::utils::DEVICE_CPU};
#endif
}

TEST_P(OVClassHeteroCompiledModelGetMetricTest_SUPPORTED_CONFIG_KEYS, GetMetricNoThrow) {
    ov::Core ie = ov::test::utils::create_core();

    auto heteroExeNetwork = ie.compile_model(actualNetwork, heteroDeviceName);
    auto deviceExeNetwork = ie.compile_model(actualNetwork, target_device);

    std::vector<ov::PropertyName> heteroConfigValues, deviceConfigValues;
    OV_ASSERT_NO_THROW(heteroConfigValues = heteroExeNetwork.get_property(ov::supported_properties));
    OV_ASSERT_NO_THROW(deviceConfigValues = deviceExeNetwork.get_property(ov::supported_properties));

    std::cout << "Supported config keys: " << std::endl;
    for (auto&& conf : heteroConfigValues) {
        std::cout << conf << std::endl;
        ASSERT_LT(0, conf.size());
    }
    ASSERT_LE(0, heteroConfigValues.size());

    // check that all device config values are present in hetero case
    for (auto&& deviceConf : deviceConfigValues) {
        auto it = std::find(heteroConfigValues.begin(), heteroConfigValues.end(), deviceConf);
        ASSERT_TRUE(it != heteroConfigValues.end());

        ov::Any heteroConfigValue = heteroExeNetwork.get_property(deviceConf);
        ov::Any deviceConfigValue = deviceExeNetwork.get_property(deviceConf);

        if (ov::internal::exclusive_async_requests.name() != deviceConf &&
            ov::supported_properties.name() != deviceConf) {
            std::stringstream strm;
            deviceConfigValue.print(strm);
            strm << " ";
            heteroConfigValue.print(strm);
            ASSERT_EQ(deviceConfigValue, heteroConfigValue) << deviceConf << " " << strm.str();
        }
    }
}

TEST_P(OVClassHeteroCompiledModelGetMetricTest_TARGET_FALLBACK, GetMetricNoThrow) {
    ov::Core ie = ov::test::utils::create_core();

    setHeteroNetworkAffinity(target_device);

    auto compiled_model = ie.compile_model(actualNetwork, heteroDeviceName);

    std::string targets;
    OV_ASSERT_NO_THROW(targets = compiled_model.get_property(ov::device::priorities));
    auto expectedTargets = target_device;

    std::cout << "Compiled model fallback targets: " << targets << std::endl;
    ASSERT_EQ(expectedTargets, targets);
}

TEST_P(OVClassHeteroCompiledModelGetMetricTest_EXEC_DEVICES, GetMetricNoThrow) {
    ov::Core ie = ov::test::utils::create_core();
    std::vector<std::string> expectedTargets = {target_device};

    SetCpuAffinity(ie, expectedTargets);

    auto compiled_model = ie.compile_model(actualNetwork, heteroDeviceName);

    std::vector<std::string> exeTargets;
    OV_ASSERT_NO_THROW(exeTargets = compiled_model.get_property(ov::execution_devices));

    ASSERT_EQ(expectedTargets, exeTargets);
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
