// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "nodes/executors/executor.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

enum ACLArgs {
    ACL_SRC_0,
    ACL_SRC_1,
    ACL_SRC_2,
    ACL_BIAS,
    ACL_WEI,
    ACL_DST,
    COUNT_OF_ARGS
};

using ACLInfo          = std::shared_ptr<arm_compute::TensorInfo>;
using ACLMemory        = std::shared_ptr<arm_compute::Tensor>;
using ACLFunction      = std::unique_ptr<arm_compute::IFunction>;
using ACLMemoryShapes  = std::array<arm_compute::TensorShape, ACLArgs::COUNT_OF_ARGS>;
using ACLMemoryInfo    = std::array<ACLInfo, ACLArgs::COUNT_OF_ARGS>;
using ACLMemoryTensors = std::array<ACLMemory, ACLArgs::COUNT_OF_ARGS>;

struct ACLTensorAttrs {
    bool hasLayoutTypeNHWC = false;
    size_t maxDimsShape = arm_compute::MAX_DIMS;
};

class ACLCommonExecutor : public Executor {
public:
    virtual void updateTensorsShapes(ACLMemoryShapes& aclMemoryShapes) {}
    virtual arm_compute::Status validateTensorsInfo(const ACLMemoryInfo & aclMemoryInfos) {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'updateTensorsInfo' method is not implemented by executor");
    }
    virtual ACLFunction configureFunction(const ACLMemoryTensors& aclMemoryTensors) {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'configureFunction' method is not implemented by executor");
    }
    impl_desc_type implType() const override {
        return impl_desc_type::acl;
    }
    void execute(const MemoryArgs& memory) final;
    bool update(const MemoryArgs& memory) final;
    ~ACLCommonExecutor();

protected:
    ACLTensorAttrs aclTensorAttrs;

private:
    ACLMemoryTensors aclMemoryTensors;
    ACLFunction iFunction = nullptr;
};

using ACLCommonExecutorPtr = std::shared_ptr<ACLCommonExecutor>;

}  // namespace intel_cpu
}  // namespace ov
