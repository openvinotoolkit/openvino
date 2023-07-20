// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/mock_common.hpp"
#include "openvino/runtime/make_tensor.hpp"

//  getMetric will return a fake ov::Any, gmock will call ostreamer << ov::Any
//  it will cause core dump, so add this special implemented
namespace testing {
namespace internal {
    template<>
    void PrintTo<ov::Any>(const ov::Any& a, std::ostream* os) {
        *os << "using custom PrintTo ov::Any";
    }
}
}

namespace ov {
MockAsyncInferRequest::MockAsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                          const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                          const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor,
                          bool ifThrow)
    : IAsyncInferRequest(request, task_executor, callback_executor), m_throw(ifThrow) {
    m_pipeline = {};
    m_pipeline.push_back({task_executor,
                [this] {
                    if (m_throw)
                        OPENVINO_THROW("runtime inference failure");
                } });
}

void MockSyncInferRequest::allocate_tensor_impl(ov::SoPtr<ov::ITensor>& tensor, const element::Type& element_type, const Shape& shape) {
    if (!tensor || tensor->get_element_type() != element_type) {
        tensor = ov::make_tensor(element_type, shape);
    } else {
        tensor->set_shape(shape);
    }
}

MockSyncInferRequest::MockSyncInferRequest(const std::shared_ptr<const MockCompiledModel>& compiled_model)
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
} //namespace ov
