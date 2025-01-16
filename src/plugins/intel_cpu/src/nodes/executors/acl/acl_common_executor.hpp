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

using ACLFunction = std::unique_ptr<arm_compute::IFunction>;
using ACLShapes   = std::array<arm_compute::TensorShape, ACLArgs::COUNT_OF_ARGS>;
using ACLInfos    = std::array<std::shared_ptr<arm_compute::TensorInfo>, ACLArgs::COUNT_OF_ARGS>;
using ACLTensors  = std::array<std::shared_ptr<arm_compute::Tensor>, ACLArgs::COUNT_OF_ARGS>;

struct ACLTensorAttrs {
    bool hasLayoutTypeNHWC = false;
    size_t maxDimsShape = arm_compute::MAX_DIMS;
    std::array<bool, ACLArgs::COUNT_OF_ARGS> memoryUsageIndicator;
};

class ACLCommonExecutor : public Executor {
public:
    ACLCommonExecutor();
    virtual void updateTensorsShapes(ACLShapes& aclMemoryShapes) = 0;
    virtual arm_compute::Status validateTensorsInfo(const ACLInfos& aclMemoryInfos) = 0;
    virtual ACLFunction configureFunction(const ACLTensors& aclMemoryTensors) = 0;
    impl_desc_type implType() const override {
        return impl_desc_type::acl;
    }
    void execute(const MemoryArgs& memory) override;
    bool update(const MemoryArgs& memory) override;
    arm_compute::TensorInfo& getTensorInfo(ACLArgs index) {
        return *aclMemoryInfos[index].get();
    }
    ~ACLCommonExecutor();

protected:
    ACLTensorAttrs aclTensorAttrs;
private:
    ACLTensors aclMemoryTensors;
    ACLInfos aclMemoryInfos;
    ACLFunction iFunction = nullptr;
};

using ACLCommonExecutorPtr = std::shared_ptr<ACLCommonExecutor>;

}  // namespace intel_cpu
}  // namespace ov
