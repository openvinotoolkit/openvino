// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>
#include <ie_metric_helpers.hpp>
#include "openvino/runtime/iplugin.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

#define IE_SET_METRIC(key, name,  ...)                                                            \
    typename ::InferenceEngine::Metrics::MetricType<::InferenceEngine::Metrics::key>::type name = \
        __VA_ARGS__;

#define RETURN_MOCK_VALUE(value) \
    InvokeWithoutArgs([value](){return ov::Any(value);})

//  getMetric will return a fake ov::Any, gmock will call ostreamer << ov::Any
//  it will cause core dump, so add this special implemented
namespace testing {
namespace internal {
    template<>
    void PrintTo<ov::Any>(const ov::Any& a, std::ostream* os);
}
}

#define ENABLE_LOG_IN_MOCK() \
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) { \
            std::cout << stream.str() << std::endl; \
            });

namespace ov {
class MockCompiledModel : public ICompiledModel {
public:
    MockCompiledModel(const std::shared_ptr<const ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin)
        : ICompiledModel(model, plugin) {}
    MOCK_METHOD(std::shared_ptr<ISyncInferRequest>, create_sync_infer_request, (), (const, override));
    MOCK_METHOD(Any, get_property, (const std::string&), (const, override));
    MOCK_METHOD(void, set_property, (const AnyMap&), (override));
    MOCK_METHOD(void, export_model, (std::ostream&), (const, override));
    MOCK_METHOD(std::shared_ptr<const Model>, get_runtime_model, (), (const, override));
    MOCK_METHOD(std::shared_ptr<IAsyncInferRequest>, create_infer_request, (), (const, override));
};

class MockAsyncInferRequest : public IAsyncInferRequest {
public:
    MockAsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                          const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                          const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor,
                          bool ifThrow);
private:
    bool m_throw;
};

class MockSyncInferRequest : public ISyncInferRequest {
public:
    MockSyncInferRequest(const std::shared_ptr<const MockCompiledModel>& compiled_model);
    MOCK_METHOD(std::vector<ov::ProfilingInfo>, get_profiling_info, (), (const, override));
    //MOCK_METHOD(Tensor, get_tensor, (const Output<const Node>&), (const, override));
    //MOCK_METHOD(void, set_tensor, (const Output<const Node>&, const Tensor&), (override));
    //MOCK_METHOD(std::vector<Tensor>, get_tensors, (const Output<const Node>&), (const, override));
    //MOCK_METHOD(void, set_tensors, (const Output<const Node>&, const std::vector<Tensor>&), (override));
    MOCK_METHOD(void, infer, (), (override));
    MOCK_METHOD(std::vector<ov::SoPtr<IVariableState>>, query_state, (), (const, override));
    //MOCK_METHOD(const std::shared_ptr<const ICompiledModel>&, get_compiled_model, (), (const, override));
    //MOCK_METHOD(const std::vector<Output<const Node>>&, get_inputs, (), (const, override));
    //MOCK_METHOD(const std::vector<Output<const Node>>&, get_outputs, (), (const, override));
    //MOCK_METHOD(void, check_tensors, (), (const, override));
    ~MockSyncInferRequest() = default;

private:
    void allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor, const ov::element::Type& element_type, const ov::Shape& shape);
};
} // namespace ov
