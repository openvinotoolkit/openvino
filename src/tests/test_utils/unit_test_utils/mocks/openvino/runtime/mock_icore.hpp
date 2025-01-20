// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/icore.hpp"

namespace ov {

class MockICore : public ov::ICore {
public:
    MOCK_METHOD(ov::Any, get_property, (const std::string&, const std::string&, const ov::AnyMap&), (const));
    MOCK_METHOD(ov::Any, get_property, (const std::string&, const std::string&), (const));
    MOCK_METHOD(ov::AnyMap, get_supported_property, (const std::string&, const ov::AnyMap&, const bool), (const));
    MOCK_METHOD(ov::AnyMap, create_compile_config, (const std::string&, const ov::AnyMap&), (const));
    MOCK_METHOD(ov::SoPtr<ov::IRemoteContext>,
                create_context,
                (const std::string& deviceName, const ov::AnyMap& params),
                (const));
    MOCK_METHOD(std::vector<std::string>, get_available_devices, (), (const));
    MOCK_METHOD(ov::SupportedOpsMap,
                query_model,
                (const std::shared_ptr<const ov::Model>&, const std::string&, const ov::AnyMap&),
                (const));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>,
                import_model,
                (std::istream&, const std::string&, const ov::AnyMap&),
                (const));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>,
                compile_model,
                (const std::shared_ptr<const ov::Model>&, const std::string&, const ov::AnyMap&),
                (const));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>,
                compile_model,
                (const std::shared_ptr<const ov::Model>&, const ov::SoPtr<ov::IRemoteContext>&, const ov::AnyMap&),
                (const));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>,
                compile_model,
                (const std::string&, const std::string&, const ov::AnyMap&),
                (const));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>,
                compile_model,
                (const std::string&, const ov::Tensor&, const std::string&, const ov::AnyMap&),
                (const));
    MOCK_METHOD(std::shared_ptr<ov::Model>, read_model, (const std::string&, const ov::Tensor&, bool), (const));
    MOCK_METHOD(std::shared_ptr<ov::Model>,
                read_model,
                (const std::string&, const std::string&, const ov::AnyMap&),
                (const));
    MOCK_METHOD(std::shared_ptr<ov::Model>,
                read_model,
                (const std::shared_ptr<AlignedBuffer>&, const std::shared_ptr<AlignedBuffer>&),
                (const));
    MOCK_METHOD(ov::SoPtr<ov::IRemoteContext>, get_default_context, (const std::string&), (const));
    MOCK_METHOD(ov::SoPtr<ov::ICompiledModel>,
                import_model,
                (std::istream&, const ov::SoPtr<ov::IRemoteContext>&, const ov::AnyMap&),
                (const));
    MOCK_METHOD(bool, device_supports_model_caching, (const std::string&), (const));
    MOCK_METHOD(void, set_property, (const std::string& device_name, const ov::AnyMap& properties));

    ~MockICore() = default;
};

}  // namespace ov
