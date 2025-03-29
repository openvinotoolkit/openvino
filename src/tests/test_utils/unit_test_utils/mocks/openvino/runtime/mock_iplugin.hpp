// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <map>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/version.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"

namespace ov {

class MockIPlugin : public ov::IPlugin {
public:
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>,
                compile_model,
                (const std::shared_ptr<const ov::Model>&, const ov::AnyMap&),
                (const));
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>, compile_model, (const std::string&, const ov::AnyMap&), (const));
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>,
                compile_model,
                (const std::shared_ptr<const ov::Model>&, const ov::AnyMap&, const ov::SoPtr<ov::IRemoteContext>&),
                (const));
    MOCK_METHOD(void, set_property, (const ov::AnyMap&));
    MOCK_METHOD(ov::Any, get_property, (const std::string&, const ov::AnyMap&), (const));
    MOCK_METHOD(ov::SoPtr<ov::IRemoteContext>, create_context, (const ov::AnyMap&), (const));
    MOCK_METHOD(ov::SoPtr<ov::IRemoteContext>, get_default_context, (const ov::AnyMap&), (const));

    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>, import_model, (std::istream&, const ov::AnyMap&), (const));
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>,
                import_model,
                (std::istream&, const ov::SoPtr<ov::IRemoteContext>&, const ov::AnyMap&),
                (const));
    MOCK_METHOD(ov::SupportedOpsMap,
                query_model,
                (const std::shared_ptr<const ov::Model>&, const ov::AnyMap&),
                (const));
};

}  // namespace ov
