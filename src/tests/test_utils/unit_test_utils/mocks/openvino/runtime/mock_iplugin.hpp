// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <map>
#include <string>

#include "common_test_utils/file_utils.hpp"
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
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>, import_model, (const ov::Tensor&, const ov::AnyMap&), (const));
    MOCK_METHOD(std::shared_ptr<ov::ICompiledModel>,
                import_model,
                (const ov::Tensor&, const ov::SoPtr<ov::IRemoteContext>&, const ov::AnyMap&),
                (const));
    MOCK_METHOD(ov::SupportedOpsMap,
                query_model,
                (const std::shared_ptr<const ov::Model>&, const ov::AnyMap&),
                (const));
};

namespace test::utils {
template <class TPlugin>
struct MockIPluginInjector {
    template <class... Args>
    MockIPluginInjector(Args&&... args)
        : m_so{ov::util::load_shared_object(get_mock_engine_path().c_str())},
          m_plugin_impl{std::make_unique<TPlugin>(std::forward<Args>(args)...)} {}

    void inject_plugin() const {
        const auto inject = make_std_function<void(ov::IPlugin*)>(m_so, "InjectPlugin");
        inject(m_plugin_impl.get());
    }

private:
    std::shared_ptr<void> m_so;
    std::unique_ptr<TPlugin> m_plugin_impl;
};
}  // namespace test::utils

}  // namespace ov
