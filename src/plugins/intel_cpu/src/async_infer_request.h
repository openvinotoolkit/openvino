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

    void setSubInfer(bool sub_infer) {
        m_sub_infers = sub_infer;
    }

    void throw_if_canceled() const;
    bool m_sub_infers = false;
};

}  // namespace intel_cpu
}  // namespace ov
