#pragma once

#include "megakernelImpl.h"

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