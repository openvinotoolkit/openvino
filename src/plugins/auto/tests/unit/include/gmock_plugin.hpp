// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>

#include <iostream>

#include "openvino/runtime/core.hpp"
#include "plugin.hpp"

using namespace ov::mock_auto_plugin;
namespace ov {
namespace mock_auto_plugin {

class MockAutoPlugin : public Plugin {
public:
    MOCK_METHOD((std::string), get_device_list, ((const ov::AnyMap&)), (const, override));
    MOCK_METHOD((bool), is_meta_device, ((const std::string&)), (const, override));
    MOCK_METHOD((std::list<DeviceInformation>),
                get_valid_device,
                ((const std::vector<DeviceInformation>&), const std::string&),
                (const, override));
    MOCK_METHOD(DeviceInformation,
                select_device,
                ((const std::vector<DeviceInformation>&), const std::string&, unsigned int),
                (override));
    MOCK_METHOD((std::vector<DeviceInformation>),
                parse_meta_devices,
                (const std::string&, const ov::AnyMap&),
                (const, override));
};

class MockISyncInferRequest : public ISyncInferRequest {
public:
    MockISyncInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model);
    MOCK_METHOD(std::vector<ov::ProfilingInfo>, get_profiling_info, (), (const, override));
    MOCK_METHOD(void, infer, (), (override));
    MOCK_METHOD(std::vector<ov::SoPtr<IVariableState>>, query_state, (), (const, override));
    ~MockISyncInferRequest() = default;

private:
    void allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor,
                              const ov::element::Type& element_type,
                              const ov::Shape& shape);
};

class MockAsyncInferRequest : public IAsyncInferRequest {
public:
    MockAsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                          const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                          const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor,
                          bool ifThrow)
        : IAsyncInferRequest(request, task_executor, callback_executor),
          m_throw(ifThrow) {
        m_pipeline = {};
        m_pipeline.push_back({task_executor, [this] {
                                  if (m_throw)
                                      OPENVINO_THROW("runtime inference failure");
                              }});
    }

private:
    bool m_throw;
};
}  // namespace mock_auto_plugin
}  // namespace ov
