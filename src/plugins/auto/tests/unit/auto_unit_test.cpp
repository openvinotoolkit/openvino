// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"
#include "ie_plugin_config.hpp"

std::shared_ptr<ov::Model> ov::mock_auto_plugin::tests::AutoTest::create_model() {
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::Shape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto subtract = std::make_shared<ov::opset11::Subtract>(add, const_value);
    subtract->set_friendly_name("sub");
    auto result = std::make_shared<ov::opset11::Result>(subtract);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

ov::mock_auto_plugin::tests::AutoTest::AutoTest() {
    set_log_level("LOG_NONE");
    // prepare mockicore and cnnNetwork for loading
    core = std::make_shared<NiceMock<MockICore>>();
    NiceMock<MockAutoPlugin>* mock_multi = new NiceMock<MockAutoPlugin>();
    plugin.reset(mock_multi);
    // replace core with mock Icore
    plugin->set_core(core);
    // mock execNetwork can work
    model = create_model();
    // prepare mockExeNetwork
    mockIExeNet = std::make_shared<NiceMock<ov::MockCompiledModel>>(model, plugin);
    mockExeNetwork = {mockIExeNet, {}};

    mockIExeNetActual = std::make_shared<NiceMock<ov::MockCompiledModel>>(model, plugin);
    mockExeNetworkActual = {mockIExeNetActual, {}};
    inferReqInternal = std::make_shared<ov::MockSyncInferRequest>(mockIExeNet);
    ON_CALL(*mockIExeNet.get(), create_sync_infer_request()).WillByDefault(Return(inferReqInternal));
    optimalNum = (uint32_t)1;
    ON_CALL(*mockIExeNet.get(), get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
        .WillByDefault(Return(optimalNum));
    inferReqInternalActual = std::make_shared<ov::MockSyncInferRequest>(mockIExeNetActual);
    ON_CALL(*mockIExeNetActual.get(), create_sync_infer_request()).WillByDefault(Return(inferReqInternalActual));
    ON_CALL(*mockIExeNetActual.get(), get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
        .WillByDefault(Return(optimalNum));
    ON_CALL(*mockIExeNet.get(), create_infer_request()).WillByDefault([this]() {
                return mockIExeNet->ICompiledModel::create_infer_request();
            });
    ON_CALL(*mockIExeNetActual.get(), create_infer_request()).WillByDefault([this]() {
                return mockIExeNetActual->ICompiledModel::create_infer_request();
            });
    std::vector<ov::PropertyName> supported_props = {ov::hint::num_requests};
    ON_CALL(*mockIExeNet.get(), get_property(StrEq(ov::supported_properties.name())))
        .WillByDefault(Return(ov::Any(supported_props)));
    ON_CALL(*mockIExeNetActual.get(), get_property(StrEq(ov::supported_properties.name())))
        .WillByDefault(Return(ov::Any(supported_props)));
    unsigned int num = 1;
    ON_CALL(*mockIExeNet.get(), get_property(StrEq(ov::hint::num_requests.name())))
        .WillByDefault(Return(ov::Any(num)));
    ON_CALL(*mockIExeNetActual.get(), get_property(StrEq(ov::hint::num_requests.name())))
        .WillByDefault(Return(ov::Any(num)));
    std::vector<std::string> supportConfigs = {"SUPPORTED_CONFIG_KEYS", "NUM_STREAMS"};
    ON_CALL(*core, get_property(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _))
        .WillByDefault(Return(ov::Any(supportConfigs)));
    std::vector<ov::PropertyName> supportedProps = {ov::compilation_num_threads};
    ON_CALL(*core, get_property(_, StrEq(ov::supported_properties.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(supportedProps));
    ON_CALL(*core, get_property(_, StrEq(ov::compilation_num_threads.name()), _)).WillByDefault(Return(12));
    std::vector<std::string> cpuCability =  {"FP32", "FP16", "INT8", "BIN"};
    std::vector<std::string> gpuCability =  {"FP32", "FP16", "BATCHED_BLOB", "BIN", "INT8"};
    std::vector<std::string> othersCability =  {"FP32", "FP16"};
    std::string igpuArchitecture = "GPU: vendor=0x8086 arch=0";
    std::string dgpuArchitecture = "GPU: vendor=0x8086 arch=1";
    auto iGpuType = ov::device::Type::INTEGRATED;
    auto dGpuType = ov::device::Type::DISCRETE;
    ON_CALL(*core, get_property(StrEq(ov::test::utils::DEVICE_CPU),
                   StrEq(ov::device::capabilities.name()), _)).WillByDefault(RETURN_MOCK_VALUE(cpuCability));
    ON_CALL(*core, get_property(HasSubstr("GPU"),
                StrEq(ov::device::capabilities.name()), _)).WillByDefault(RETURN_MOCK_VALUE(gpuCability));
    ON_CALL(*core, get_property(StrEq("OTHERS"),
                   StrEq(ov::device::capabilities.name()), _)).WillByDefault(RETURN_MOCK_VALUE(othersCability));
    ON_CALL(*core, get_property(StrEq("GPU"),
                StrEq(ov::device::architecture.name()), _)).WillByDefault(RETURN_MOCK_VALUE(igpuArchitecture));
    ON_CALL(*core, get_property(StrEq("GPU.0"),
                StrEq(ov::device::architecture.name()), _)).WillByDefault(RETURN_MOCK_VALUE(igpuArchitecture));
    ON_CALL(*core, get_property(StrEq("GPU.1"),
                StrEq(ov::device::architecture.name()), _)).WillByDefault(RETURN_MOCK_VALUE(dgpuArchitecture));
    ON_CALL(*core, get_property(StrEq("GPU"),
                StrEq(ov::device::type.name()), _)).WillByDefault(RETURN_MOCK_VALUE(iGpuType));
    ON_CALL(*core, get_property(StrEq("GPU.0"),
                StrEq(ov::device::type.name()), _)).WillByDefault(RETURN_MOCK_VALUE(iGpuType));
    ON_CALL(*core, get_property(StrEq("GPU.1"),
                StrEq(ov::device::type.name()), _)).WillByDefault(RETURN_MOCK_VALUE(dGpuType));
    const std::vector<std::string> metrics = {METRIC_KEY(SUPPORTED_CONFIG_KEYS), ov::device::full_name.name(), ov::device::id.name()};
    const char igpuFullDeviceName[] = "Intel(R) Gen9 HD Graphics (iGPU)";
    const char dgpuFullDeviceName[] = "Intel(R) Iris(R) Xe MAX Graphics (dGPU)";
    ON_CALL(*core, get_property(_, StrEq(METRIC_KEY(SUPPORTED_METRICS)), _))
           .WillByDefault(RETURN_MOCK_VALUE(metrics));
    ON_CALL(*core, get_property(_, ov::supported_properties.name(), _))
           .WillByDefault(Return(ov::Any(supported_props)));
    ON_CALL(*core, get_property(StrEq("GPU"),
                StrEq(ov::device::full_name.name()), _)).WillByDefault(RETURN_MOCK_VALUE(igpuFullDeviceName));
    ON_CALL(*core, get_property(StrEq("GPU"),
                StrEq(ov::device::id.name()), _)).WillByDefault(Return(ov::Any("0")));
    ON_CALL(*core, get_property(StrEq("GPU.0"),
                StrEq(ov::device::full_name.name()), _)).WillByDefault(RETURN_MOCK_VALUE(igpuFullDeviceName));
    ON_CALL(*core, get_property(StrEq("GPU.1"),
                StrEq(ov::device::full_name.name()), _)).WillByDefault(RETURN_MOCK_VALUE(dgpuFullDeviceName));
    const std::vector<std::string>  availableDevs = {"CPU", "GPU.0", "GPU.1"};
    ON_CALL(*core, get_available_devices()).WillByDefault(Return(availableDevs));
    ON_CALL(*plugin, parse_meta_devices)
    .WillByDefault(
        [this](const std::string& priorityDevices, const ov::AnyMap& config) {
            return plugin->Plugin::parse_meta_devices(priorityDevices, config);
        });

    ON_CALL(*plugin, select_device)
        .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices,
                                const std::string& netPrecision,
                                unsigned int priority) {
            return plugin->Plugin::select_device(metaDevices, netPrecision, priority);
        });

    ON_CALL(*plugin, get_valid_device)
        .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
            std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
            return devices;
        });

    ON_CALL(*core, get_supported_property).WillByDefault([](const std::string& device, const ov::AnyMap& fullConfigs) {
        auto item = fullConfigs.find(ov::device::properties.name());
        ov::AnyMap deviceConfigs;
        if (item != fullConfigs.end()) {
            ov::AnyMap devicesProperties;
            std::stringstream strConfigs(item->second.as<std::string>());
            // Parse the device properties to common property into deviceConfigs.
            ov::util::Read<ov::AnyMap>{}(strConfigs, devicesProperties);
            auto it = devicesProperties.find(device);
            if (it != devicesProperties.end()) {
                std::stringstream strConfigs(it->second.as<std::string>());
                ov::util::Read<ov::AnyMap>{}(strConfigs, deviceConfigs);
            }
        }
        for (auto&& item : fullConfigs) {
            if (item.first != ov::device::properties.name()) {
                // primary property
                // override will not happen here if the property already present in the device config list.
                deviceConfigs.insert(item);
            }
        }
        return deviceConfigs;
    });

    ON_CALL(*plugin, get_device_list).WillByDefault([this](const ov::AnyMap& config) {
        return plugin->Plugin::get_device_list(config);
    });
}

ov::mock_auto_plugin::tests::AutoTest::~AutoTest() {
    testing::Mock::AllowLeak(plugin.get());
    testing::Mock::AllowLeak(mockIExeNet.get());
    testing::Mock::AllowLeak(mockIExeNetActual.get());
    core.reset();
    plugin.reset();
    mockExeNetwork = {};
    mockExeNetworkActual = {};
    config.clear();
    metaDevices.clear();
    inferReqInternal.reset();
    inferReqInternalActual.reset();
}