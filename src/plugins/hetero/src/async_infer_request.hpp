// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "sync_infer_request.hpp"

namespace ov {
namespace hetero {

class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    AsyncInferRequest(const std::shared_ptr<InferRequest>& request,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);

    ~AsyncInferRequest();

    void cancel() override;

private:
    std::shared_ptr<InferRequest> m_infer_request;
};

}  // namespace hetero
}  // namespace ov