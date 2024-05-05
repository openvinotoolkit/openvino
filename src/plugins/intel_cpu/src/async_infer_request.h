// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "infer_request.h"
#include "openvino/runtime/iasync_infer_request.hpp"

namespace ov {
namespace intel_cpu {

class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    AsyncInferRequest(const std::shared_ptr<IInferRequest>& request,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);
    ~AsyncInferRequest();

    void setSubInferRequest(const std::vector<std::shared_ptr<IAsyncInferRequest>>& requests);

    std::vector<std::shared_ptr<ov::IAsyncInferRequest>> getSubInferRequest() const {
        return m_sub_infer_requests;
    }

    void throw_if_canceled() const;
private:
    std::vector<std::shared_ptr<ov::IAsyncInferRequest>> m_sub_infer_requests;
};

}  // namespace intel_cpu
}  // namespace ov
