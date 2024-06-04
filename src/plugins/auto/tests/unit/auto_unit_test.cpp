// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"

#include "common_test_utils/file_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/file_util.hpp"

namespace testing {
namespace internal {
template <>
void PrintTo<ov::Any>(const ov::Any& a, std::ostream* os) {
    *os << "using custom PrintTo ov::Any";
}
}  // namespace internal
}  // namespace testing

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

std::shared_ptr<ov::Model> ov::mock_auto_plugin::tests::BaseTest::create_dynamic_output_model() {
    auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
    boxes->set_friendly_name("param_1");
    boxes->get_output_tensor(0).set_names({"input_tensor_1"});
    auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 2});
    scores->set_friendly_name("param_2");
    scores->get_output_tensor(0).set_names({"input_tensor_2"});
    auto max_output_boxes_per_class = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {10});
    auto iou_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.75});
    auto score_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.7});
    auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold);
    auto res = std::make_shared<ov::op::v0::Result>(nms);
    res->set_friendly_name("output_dynamic");
    return std::make_shared<ov::Model>(ov::NodeVector{nms}, ov::ParameterVector{boxes, scores});
}

ov::mock_auto_plugin::tests::BaseTest::BaseTest(const MODELTYPE modelType) {
    set_log_level("LOG_NONE");
    if (modelType == MODELTYPE::DYNAMIC)
        model = create_dynamic_output_model();
    else
        model = create_model();
    // construct mock auto plugin
    NiceMock<MockAutoPlugin>* mock_auto = new NiceMock<MockAutoPlugin>();
    plugin.reset(mock_auto);
    // construct  mock plugin
    mock_plugin_cpu = std::make_shared<NiceMock<ov::MockIPlugin>>();
    mock_plugin_gpu = std::make_shared<NiceMock<ov::MockIPlugin>>();
    // prepare mockExeNetwork
    mockIExeNet = std::make_shared<NiceMock<ov::MockICompiledModel>>(model, mock_plugin_cpu);
    mockExeNetwork = {mockIExeNet, {}};

    mockIExeNetActual = std::make_shared<NiceMock<ov::MockICompiledModel>>(model, mock_plugin_gpu);
    mockExeNetworkActual = {mockIExeNetActual, {}};
    ON_CALL(*mockIExeNet.get(), inputs()).WillByDefault(ReturnRefOfCopy(model->inputs()));
    ON_CALL(*mockIExeNet.get(), outputs()).WillByDefault(ReturnRefOfCopy(model->outputs()));
    ON_CALL(*mockIExeNetActual.get(), inputs()).WillByDefault(ReturnRefOfCopy(model->inputs()));
    ON_CALL(*mockIExeNetActual.get(), outputs()).WillByDefault(ReturnRefOfCopy(model->outputs()));
    inferReqInternal = std::make_shared<ov::mock_auto_plugin::MockISyncInferRequest>(mockIExeNet);

    ON_CALL(*mockIExeNet.get(), create_sync_infer_request()).WillByDefault([this]() {
        return inferReqInternal;
    });
    optimalNum = (uint32_t)1;
    ON_CALL(*mockIExeNet.get(), get_property(StrEq(ov::optimal_number_of_infer_requests.name())))
        .WillByDefault(Return(optimalNum));
    inferReqInternalActual = std::make_shared<ov::mock_auto_plugin::MockISyncInferRequest>(mockIExeNetActual);

    ON_CALL(*mockIExeNetActual.get(), create_sync_infer_request()).WillByDefault([this]() {
        return inferReqInternalActual;
    });
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
    ON_CALL(*mockIExeNet.get(), get_property(StrEq(ov::hint::num_requests.name()))).WillByDefault(Return(ov::Any(num)));
    ON_CALL(*mockIExeNetActual.get(), get_property(StrEq(ov::hint::num_requests.name())))
        .WillByDefault(Return(ov::Any(num)));
    ON_CALL(*plugin, get_device_list).WillByDefault([this](const ov::AnyMap& config) {
        return plugin->Plugin::get_device_list(config);
    });
    ON_CALL(*plugin, parse_meta_devices)
        .WillByDefault([this](const std::string& priorityDevices, const ov::AnyMap& config) {
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
    plugin->get_executor_manager()->clear();
    plugin.reset();
}

ov::mock_auto_plugin::tests::AutoTest::AutoTest(const MODELTYPE modelType) : BaseTest(modelType) {
    // prepare mockicore and cnnNetwork for loading
    core = std::make_shared<NiceMock<MockICore>>();
    // replace core with mock Icore
    plugin->set_core(core);
    std::vector<ov::PropertyName> supportedProps = {ov::compilation_num_threads};
    ON_CALL(*core, get_property(_, StrEq(ov::supported_properties.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(supportedProps));
    ON_CALL(*core, get_property(_, StrEq(ov::compilation_num_threads.name()), _)).WillByDefault(Return(12));
    std::vector<std::string> cpuCability = {"FP32", "FP16", "INT8", "BIN"};
    std::vector<std::string> gpuCability = {"FP32", "FP16", "BIN", "INT8"};
    std::vector<std::string> othersCability = {"FP32", "FP16"};
    std::string igpuArchitecture = "GPU: vendor=0x8086 arch=0";
    std::string dgpuArchitecture = "GPU: vendor=0x8086 arch=1";
    auto iGpuType = ov::device::Type::INTEGRATED;
    auto dGpuType = ov::device::Type::DISCRETE;
    ON_CALL(*core, get_property(StrEq(ov::test::utils::DEVICE_CPU), StrEq(ov::device::capabilities.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(cpuCability));
    ON_CALL(*core, get_property(HasSubstr("GPU"), StrEq(ov::device::capabilities.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(gpuCability));
    ON_CALL(*core, get_property(StrEq("OTHERS"), StrEq(ov::device::capabilities.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(othersCability));
    ON_CALL(*core, get_property(StrEq("GPU"), StrEq(ov::device::architecture.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(igpuArchitecture));
    ON_CALL(*core, get_property(StrEq("GPU.0"), StrEq(ov::device::architecture.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(igpuArchitecture));
    ON_CALL(*core, get_property(StrEq("GPU.1"), StrEq(ov::device::architecture.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(dgpuArchitecture));
    ON_CALL(*core, get_property(StrEq("GPU"), StrEq(ov::device::type.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(iGpuType));
    ON_CALL(*core, get_property(StrEq("GPU.0"), StrEq(ov::device::type.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(iGpuType));
    ON_CALL(*core, get_property(StrEq("GPU.1"), StrEq(ov::device::type.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(dGpuType));
    const char igpuFullDeviceName[] = "Intel(R) Gen9 HD Graphics (iGPU)";
    const char dgpuFullDeviceName[] = "Intel(R) Iris(R) Xe MAX Graphics (dGPU)";
    ON_CALL(*core, get_property(_, ov::supported_properties.name(), _)).WillByDefault(Return(ov::Any(supportedProps)));
    ON_CALL(*core, get_property(StrEq("GPU"), StrEq(ov::device::full_name.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(igpuFullDeviceName));
    ON_CALL(*core, get_property(StrEq("GPU"), StrEq(ov::device::id.name()), _)).WillByDefault(Return(ov::Any("0")));
    ON_CALL(*core, get_property(StrEq("GPU.0"), StrEq(ov::device::full_name.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(igpuFullDeviceName));
    ON_CALL(*core, get_property(StrEq("GPU.1"), StrEq(ov::device::full_name.name()), _))
        .WillByDefault(RETURN_MOCK_VALUE(dgpuFullDeviceName));
    const std::vector<std::string> availableDevs = {"CPU", "GPU.0", "GPU.1"};
    ON_CALL(*core, get_available_devices()).WillByDefault(Return(availableDevs));
    ON_CALL(*core, get_supported_property).WillByDefault([](const std::string& device, const ov::AnyMap& fullConfigs, const bool keep_core_property = true) {
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

void ov::mock_auto_plugin::MockISyncInferRequest::allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor,
                                                                       const element::Type& element_type,
                                                                       const Shape& shape) {
    if (!tensor || tensor->get_element_type() != element_type) {
        tensor = ov::make_tensor(element_type, shape);
    } else {
        tensor->set_shape(shape);
    }
}

ov::mock_auto_plugin::MockISyncInferRequest::MockISyncInferRequest(
    const std::shared_ptr<const ov::ICompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model) {
    OPENVINO_ASSERT(compiled_model);
    // Allocate input/output tensors
    for (const auto& input : get_inputs()) {
        allocate_tensor(input, [this, input](ov::SoPtr<ov::ITensor>& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor,
                                 input.get_element_type(),
                                 input.get_partial_shape().is_dynamic() ? ov::Shape{0} : input.get_shape());
        });
    }
    for (const auto& output : get_outputs()) {
        allocate_tensor(output, [this, output](ov::SoPtr<ov::ITensor>& tensor) {
            // Can add a check to avoid double work in case of shared tensors
            allocate_tensor_impl(tensor,
                                 output.get_element_type(),
                                 output.get_partial_shape().is_dynamic() ? ov::Shape{0} : output.get_shape());
        });
    }
}