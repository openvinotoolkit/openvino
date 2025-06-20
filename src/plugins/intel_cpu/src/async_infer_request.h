// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "infer_request.h"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

namespace ov::intel_cpu {

class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    AsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor,
                      bool is_optimized_single_stream = false);
    ~AsyncInferRequest() override;

    void infer() override;

    void setSubInferRequest(const std::vector<std::shared_ptr<IAsyncInferRequest>>& requests);

    std::vector<std::shared_ptr<ov::IAsyncInferRequest>> getSubInferRequest() const {
        return m_sub_infer_requests;
    }

    void setSubInfer(bool has_sub_infer) {
        m_has_sub_infers = has_sub_infer;
    }

    void throw_if_canceled() const;

    std::vector<std::shared_ptr<ov::IAsyncInferRequest>> m_sub_infer_requests;
    bool m_has_sub_infers = false;
    std::shared_ptr<IInferRequest> m_internal_request;
    std::shared_ptr<ov::threading::IStreamsExecutor> m_stream_executor;
    std::function<void()> m_infer_func;
};

}  // namespace ov::intel_cpu
