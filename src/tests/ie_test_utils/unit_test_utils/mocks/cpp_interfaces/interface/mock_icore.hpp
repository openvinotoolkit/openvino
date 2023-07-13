// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include "ie_icore.hpp"
#include "openvino/runtime/icompiled_model.hpp"

class MockICore : public InferenceEngine::ICore {
public:
    MOCK_CONST_METHOD3(ReadNetwork,
                       InferenceEngine::CNNNetwork(const std::string&, const InferenceEngine::Blob::CPtr&, bool));
    MOCK_CONST_METHOD2(ReadNetwork, InferenceEngine::CNNNetwork(const std::string&, const std::string&));

    MOCK_METHOD3(LoadNetwork,
                 InferenceEngine::SoExecutableNetworkInternal(const InferenceEngine::CNNNetwork&,
                                                              const std::string&,
                                                              const std::map<std::string, std::string>&));
    MOCK_METHOD3(LoadNetwork,
                 InferenceEngine::SoExecutableNetworkInternal(const InferenceEngine::CNNNetwork&,
                                                              const std::shared_ptr<InferenceEngine::RemoteContext>&,
                                                              const std::map<std::string, std::string>&));
    MOCK_METHOD4(
        LoadNetwork,
        InferenceEngine::SoExecutableNetworkInternal(const std::string&,
                                                     const std::string&,
                                                     const std::map<std::string, std::string>&,
                                                     const std::function<void(const InferenceEngine::CNNNetwork&)>&));
    MOCK_METHOD5(
        LoadNetwork,
        InferenceEngine::SoExecutableNetworkInternal(const std::string&,
                                                     const InferenceEngine::Blob::CPtr&,
                                                     const std::string&,
                                                     const std::map<std::string, std::string>&,
                                                     const std::function<void(const InferenceEngine::CNNNetwork&)>&));

    MOCK_METHOD3(ImportNetwork,
                 InferenceEngine::SoExecutableNetworkInternal(std::istream&,
                                                              const std::string&,
                                                              const std::map<std::string, std::string>&));
    MOCK_METHOD3(ImportNetwork,
                 InferenceEngine::SoExecutableNetworkInternal(std::istream&,
                                                              const std::shared_ptr<InferenceEngine::RemoteContext>&,
                                                              const std::map<std::string, std::string>&));

    MOCK_METHOD2(CreateContext,
                 InferenceEngine::RemoteContext::Ptr(const std::string& deviceName,
                                                     const InferenceEngine::ParamMap& params));

    MOCK_CONST_METHOD3(QueryNetwork,
                       InferenceEngine::QueryNetworkResult(const InferenceEngine::CNNNetwork&,
                                                           const std::string&,
                                                           const std::map<std::string, std::string>&));

    MOCK_CONST_METHOD3(GetMetric, ov::Any(const std::string&, const std::string&, const ov::AnyMap&));
    MOCK_CONST_METHOD2(GetConfig, ov::Any(const std::string&, const std::string&));
    MOCK_CONST_METHOD3(get_property, ov::Any(const std::string&, const std::string&, const ov::AnyMap&));
    MOCK_CONST_METHOD0(GetAvailableDevices, std::vector<std::string>());
    MOCK_CONST_METHOD1(DeviceSupportsModelCaching, bool(const std::string&));  // NOLINT not a cast to bool
    MOCK_METHOD2(GetSupportedConfig,
                 std::map<std::string, std::string>(const std::string&, const std::map<std::string, std::string>&));
    MOCK_CONST_METHOD2(get_supported_property, ov::AnyMap(const std::string&, const ov::AnyMap&));
    MOCK_CONST_METHOD0(isNewAPI, bool());
    MOCK_METHOD1(GetDefaultContext, InferenceEngine::RemoteContext::Ptr(const std::string&));

    MOCK_CONST_METHOD0(is_new_api, bool());
    MOCK_CONST_METHOD2(create_context, ov::RemoteContext(const std::string& deviceName, const ov::AnyMap& params));
    MOCK_CONST_METHOD0(get_available_devices, std::vector<std::string>());
    MOCK_CONST_METHOD3(query_model,
                       ov::SupportedOpsMap(const std::shared_ptr<const ov::Model>&,
                                           const std::string&,
                                           const ov::AnyMap&));
    MOCK_CONST_METHOD3(import_model,
                       ov::SoPtr<ov::ICompiledModel>(std::istream&, const std::string&, const ov::AnyMap&));
    MOCK_CONST_METHOD3(compile_model,
                       ov::SoPtr<ov::ICompiledModel>(const std::shared_ptr<const ov::Model>&,
                                                     const std::string&,
                                                     const ov::AnyMap&));
    MOCK_CONST_METHOD3(compile_model,
                       ov::SoPtr<ov::ICompiledModel>(const std::shared_ptr<const ov::Model>&,
                                                     const ov::RemoteContext&,
                                                     const ov::AnyMap&));
    MOCK_CONST_METHOD3(compile_model,
                       ov::SoPtr<ov::ICompiledModel>(const std::string&, const std::string&, const ov::AnyMap&));
    MOCK_CONST_METHOD4(
        compile_model,
        ov::SoPtr<ov::ICompiledModel>(const std::string&, const ov::Tensor&, const std::string&, const ov::AnyMap&));
    MOCK_CONST_METHOD3(read_model, std::shared_ptr<ov::Model>(const std::string&, const ov::Tensor&, bool));
    MOCK_CONST_METHOD2(read_model, std::shared_ptr<ov::Model>(const std::string&, const std::string&));
    MOCK_CONST_METHOD1(get_default_context, ov::RemoteContext(const std::string&));
    MOCK_CONST_METHOD3(import_model,
                       ov::SoPtr<ov::ICompiledModel>(std::istream&, const ov::RemoteContext&, const ov::AnyMap&));
    MOCK_METHOD2(set_property, void(const std::string& device_name, const ov::AnyMap& properties));

    ~MockICore() = default;
};
