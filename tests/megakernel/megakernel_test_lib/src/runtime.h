// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mk_params.h"

namespace mk {
class TestRuntime : public IMegakernelRuntime {
public:
    ~TestRuntime() override;
    TErrorcode Init(const IConstantParams* weights, const IPlatformParams* platform) override;
    TErrorcode Execute(const IRuntimeParams* io) override;
    TErrorcode Destroy() override;

private:
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_command_queue queue = nullptr;
    clSetKernelArgMemPointerINTEL_fn set_pointer = nullptr;
};
}