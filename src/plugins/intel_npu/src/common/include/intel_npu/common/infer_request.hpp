// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/iinfer_request.hpp"

namespace intel_npu {

class InferRequest : public ov::IInferRequest {
public:
    virtual void infer_async() = 0;
    virtual void get_result() = 0;

    virtual ~InferRequest() = default;
};

}  // namespace intel_npu
