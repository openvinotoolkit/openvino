// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "nodes/executors/executor.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

using ACLMemoryInfo = std::shared_ptr<arm_compute::TensorInfo>;
using ACLMemory     = std::shared_ptr<arm_compute::Tensor>;
using ACLMemoryMap  = std::unordered_map<int, ACLMemory>;
using ACLFunction   = std::unique_ptr<arm_compute::IFunction>;

struct ACLTensorAttrs {
    bool hasLayoutTypeNHWC = false;
    size_t maxDimsShape = arm_compute::MAX_DIMS;
};

class ACLCommonExecutor : public Executor {
public:
    virtual arm_compute::Status updateTensorsInfo(const ACLMemoryMap& acl_memory) {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'updateTensorsInfo' method is not implemented by executor");
    }
    virtual ACLFunction configureFunction(const ACLMemoryMap& acl_memory) {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'configureFunction' method is not implemented by executor");
    }
    impl_desc_type implType() const override {
        return impl_desc_type::acl;
    }
    static arm_compute::ITensorInfo* getACLInfo(const ACLMemory& aclMemory);
    void execute(const MemoryArgs& memory) final;
    bool update(const MemoryArgs& memory) final;
    ~ACLCommonExecutor();

protected:
    ACLTensorAttrs aclTensorAttrs;

private:
    ACLMemoryMap aclMemoryMap;
    ACLFunction iFunction = nullptr;
};

using ACLCommonExecutorPtr = std::shared_ptr<ACLCommonExecutor>;

}  // namespace intel_cpu
}  // namespace ov
