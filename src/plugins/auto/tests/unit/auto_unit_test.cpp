// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

std::shared_ptr<ov::Model> ov::mock_auto_plugin::tests::BaseTest::create_model() {
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

ov::mock_auto_plugin::tests::BaseTest::BaseTest() {
    set_log_level("LOG_NONE");
    model = create_model();
    // construct mock auto plugin
    NiceMock<MockAutoPlugin>* mock_auto = new NiceMock<MockAutoPlugin>();
    plugin.reset(mock_auto);
    // construct  mock plugin
    mock_plugin_cpu = std::make_shared<NiceMock<ov::MockPluginBase>>();
    mock_plugin_gpu = std::make_shared<NiceMock<ov::MockPluginBase>>();
    // prepare mockExeNetwork
    mockIExeNet = std::make_shared<NiceMock<ov::MockCompiledModel>>(model, mock_plugin_cpu);
    mockExeNetwork = {mockIExeNet, {}};

    mockIExeNetActual = std::make_shared<NiceMock<ov::MockCompiledModel>>(model, mock_plugin_gpu);
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
    ON_CALL(*plugin, get_device_list).WillByDefault([this](const ov::AnyMap& config) {
        return plugin->Plugin::get_device_list(config);
    });
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
}

ov::mock_auto_plugin::tests::BaseTest::~BaseTest() {
    testing::Mock::AllowLeak(mockIExeNet.get());
    testing::Mock::AllowLeak(mockIExeNetActual.get());
    testing::Mock::AllowLeak(mock_plugin_cpu.get());
    testing::Mock::AllowLeak(mock_plugin_gpu.get());
    testing::Mock::AllowLeak(plugin.get());
    mockExeNetwork = {};
    mockExeNetworkActual = {};
    config.clear();
    metaDevices.clear();
    inferReqInternal.reset();
    inferReqInternalActual.reset();
    mock_plugin_cpu.reset();
    mock_plugin_gpu.reset();
    plugin.reset();
}

ov::mock_auto_plugin::tests::AutoTest::AutoTest() {
    // prepare mockicore and cnnNetwork for loading
    core = std::make_shared<NiceMock<MockICore>>();
    // replace core with mock Icore
    plugin->set_core(core);
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
           .WillByDefault(Return(ov::Any(supportedProps)));
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
}

ov::mock_auto_plugin::tests::AutoTest::~AutoTest() {
    core.reset();
}

namespace {

std::string get_mock_engine_path() {
    std::string mockEngineName("mock_engine");
    return ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                              mockEngineName + IE_BUILD_POSTFIX);
}

template <class T>
std::function<T> make_std_function(const std::shared_ptr<void> so, const std::string& functionName) {
    std::function<T> ptr(reinterpret_cast<T*>(ov::util::get_symbol(so, functionName.c_str())));
    return ptr;
}

ov::PropertyName RO_property(const std::string& propertyName) {
    return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
}

ov::PropertyName RW_property(const std::string& propertyName) {
    return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
}

}  // namespace

ov::mock_auto_plugin::tests::AutoTestWithRealCore::AutoTestWithRealCore() {
    register_plugin_simple(core, "MOCK_CPU", {});
    // validate the mock plugin, to ensure the order as well
    core.get_property("MOCK_CPU", ov::supported_properties);
    register_plugin_support_batch_and_context(core, "MOCK_GPU", {});
    // validate the mock plugin
    core.get_property("MOCK_GPU", ov::supported_properties);
    ov::Any optimalNum = (uint32_t)1;
    ON_CALL(*mock_plugin_cpu.get(), compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_), _))
                    .WillByDefault(Return(mockIExeNet));
    ON_CALL(*mock_plugin_cpu.get(), compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_), _))
                    .WillByDefault(Return(mockIExeNetActual));
}

void ov::mock_auto_plugin::tests::AutoTestWithRealCore::reg_plugin(ov::Core& core,
                                              std::shared_ptr<ov::IPlugin> plugin,
                                              const std::string& device_name,
                                              const ov::AnyMap& properties) {
    std::string libraryPath = get_mock_engine_path();
    if (!m_so)
        m_so = ov::util::load_shared_object(libraryPath.c_str());
    if (device_name.find("MULTI") == std::string::npos && device_name.find("AUTO") == std::string::npos)
        plugin->set_device_name(device_name);
    std::function<void(ov::IPlugin*)> inject_mock_plugin = make_std_function<void(ov::IPlugin*)>(m_so, "InjectPlugin");

    inject_mock_plugin(plugin.get());
    core.register_plugin(ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                            std::string("mock_engine") + IE_BUILD_POSTFIX),
                         device_name,
                         properties);
}

// test
void ov::mock_auto_plugin::tests::AutoTestWithRealCore::register_plugin_support_batch_and_context(ov::Core& core,
                                                                   const std::string& device_name,
                                                                   const ov::AnyMap& properties) {
    auto remote_context = std::make_shared<ov::MockRemoteContext>(mock_plugin_gpu->get_device_name());
    m_mock_contexts.push_back(remote_context);
    ON_CALL(*mock_plugin_gpu, compile_model(_, _)).WillByDefault(Return(mockIExeNetActual));
    ON_CALL(*mock_plugin_gpu, create_context).WillByDefault(Return(ov::SoPtr<ov::IRemoteContext>(remote_context, nullptr)));
    ON_CALL(*mock_plugin_gpu, get_default_context).WillByDefault(Return(ov::SoPtr<ov::IRemoteContext>(remote_context, nullptr)));
    ON_CALL(*mock_plugin_gpu, get_property).WillByDefault([](const std::string& name, const ov::AnyMap& property) -> ov::Any {
            const std::vector<ov::PropertyName> roProperties{
                RO_property(ov::supported_properties.name()),
                RO_property(ov::optimal_batch_size.name()),
                RO_property(ov::optimal_number_of_infer_requests.name()),
                RO_property(ov::device::capabilities.name()),
                RO_property(ov::device::type.name()),
                RO_property(ov::device::uuid.name()),
            };
            // the whole config is RW before network is loaded.
            const std::vector<ov::PropertyName> rwProperties{
                RW_property(ov::num_streams.name()),
                RW_property(ov::enable_profiling.name()),
                RW_property(ov::compilation_num_threads.name()),
                RW_property(ov::hint::performance_mode.name()),
                RW_property(ov::hint::num_requests.name())
            };
            if (name == ov::supported_properties) {
                std::vector<ov::PropertyName> supportedProperties;
                supportedProperties.reserve(roProperties.size() + rwProperties.size());
                supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
                supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

                return decltype(ov::supported_properties)::value_type(supportedProperties);
            } else if (name == ov::optimal_number_of_infer_requests.name()) {
                return decltype(ov::optimal_number_of_infer_requests)::value_type(1);
            } else if (name == ov::optimal_batch_size.name()) {
                return decltype(ov::optimal_batch_size)::value_type(4);
            } else if (name == ov::device::capabilities.name()) {
                return decltype(ov::device::capabilities)::value_type({"FP32", "FP16", "BATCHED_BLOB", "BIN", "INT8"});
            } else if (name == ov::device::type.name()) {
                return decltype(ov::device::type)::value_type(ov::device::Type::INTEGRATED);
            } else if (name == ov::loaded_from_cache.name()) {
                return false;
            } else if (name == ov::enable_profiling.name()) {
                return decltype(ov::enable_profiling)::value_type{false};
            } else if (name == ov::streams::num.name()) {
                return decltype(ov::streams::num)::value_type{2};
            } else if (name == ov::compilation_num_threads.name()) {
                return decltype(ov::compilation_num_threads)::value_type{4};
            } else if (name == "SUPPORTED_CONFIG_KEYS") {  // TODO: Remove this key
                std::vector<std::string> configs;
                for (const auto& property : rwProperties) {
                    configs.emplace_back(property);
                }
                return configs;
            } else if (name == "SUPPORTED_METRICS") {  // TODO: Remove this key
                std::vector<std::string> configs;
                for (const auto& property : roProperties) {
                    configs.emplace_back(property);
                }
                return configs;
            } else if (name == ov::internal::supported_properties) {
                return decltype(ov::internal::supported_properties)::value_type({});
            }
            OPENVINO_NOT_IMPLEMENTED;
    });
    std::shared_ptr<ov::IPlugin> base_plugin = mock_plugin_gpu;
    reg_plugin(core, base_plugin, device_name, properties);
}

void ov::mock_auto_plugin::tests::AutoTestWithRealCore::register_plugin_simple(ov::Core& core,
                                                                                const std::string& device_name,
                                                                                const ov::AnyMap& properties) {
    ON_CALL(*mock_plugin_cpu, compile_model(_, _)).WillByDefault(Return(mockIExeNet));
    ON_CALL(*mock_plugin_cpu, create_context).WillByDefault(Throw(ov::Exception{"NotImplemented"}));
    ON_CALL(*mock_plugin_cpu, get_default_context).WillByDefault(Throw(ov::Exception{"NotImplemented"}));
    ON_CALL(*mock_plugin_cpu, get_property).WillByDefault([](const std::string& name, const ov::AnyMap& property) -> ov::Any {
            const std::vector<ov::PropertyName> roProperties{
                RO_property(ov::supported_properties.name()),
                RO_property(ov::device::uuid.name()),
            };
            // the whole config is RW before network is loaded.
            const std::vector<ov::PropertyName> rwProperties{
                RW_property(ov::num_streams.name()),
                RW_property(ov::enable_profiling.name()),
                RW_property(ov::hint::performance_mode.name())
            };
            if (name == ov::supported_properties) {
                std::vector<ov::PropertyName> supportedProperties;
                supportedProperties.reserve(roProperties.size() + rwProperties.size());
                supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
                supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

                return decltype(ov::supported_properties)::value_type(supportedProperties);
            } else if (name == ov::loaded_from_cache.name()) {
                return false;
            } else if (name == ov::enable_profiling.name()) {
                return decltype(ov::enable_profiling)::value_type{false};
            } else if (name == ov::streams::num.name()) {
                return decltype(ov::streams::num)::value_type{2};
            } else if (name == "SUPPORTED_CONFIG_KEYS") {  // TODO: Remove this key
                std::vector<std::string> configs;
                for (const auto& property : rwProperties) {
                    configs.emplace_back(property);
                }
                return configs;
            } else if (name == "SUPPORTED_METRICS") {  // TODO: Remove this key
                std::vector<std::string> configs;
                for (const auto& property : roProperties) {
                    configs.emplace_back(property);
                }
                return configs;
            } else if (name == ov::internal::supported_properties) {
                return decltype(ov::internal::supported_properties)::value_type({});
            }
            OPENVINO_NOT_IMPLEMENTED;
    });
    std::shared_ptr<ov::IPlugin> base_plugin = mock_plugin_cpu;

    reg_plugin(core, base_plugin, device_name, properties);
}