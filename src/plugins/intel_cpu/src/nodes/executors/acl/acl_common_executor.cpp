// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_common_executor.hpp"
#include "acl_utils.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

static const std::unordered_map<int, ACLArgs> argConvert = {
    {ARG_SRC_0, ACL_SRC_0},
    {ARG_SRC_1, ACL_SRC_1},
    {ARG_SRC_2, ACL_SRC_2},
    {ARG_BIAS,  ACL_BIAS},
    {ARG_WEI,   ACL_WEI},
    {ARG_DST,   ACL_DST},
};

using ACLTypes   = std::array<arm_compute::DataType,   ACLArgs::COUNT_OF_ARGS>;
using ACLLayouts = std::array<arm_compute::DataLayout, ACLArgs::COUNT_OF_ARGS>;

static void initACLTensorParams(const MemoryPtr& memoryPtr,
                                const ACLTensorAttrs& attrs,
                                arm_compute::TensorShape& tensorShape,
                                arm_compute::DataType& dataType,
                                arm_compute::DataLayout& dataLayout) {
    dataType = precisionToAclDataType(memoryPtr->getPrecision());
    dataLayout = getAclDataLayoutByMemoryDesc(memoryPtr->getDescPtr());
    if (dataType != arm_compute::DataType::UNKNOWN) {
        auto collapsed_dims = collapse_dims_to_max_rank(memoryPtr->getStaticDims(), attrs.maxDimsShape);
        tensorShape = shapeCast(collapsed_dims);
        if (attrs.hasLayoutTypeNHWC) {
            changeLayoutToNH_C({&tensorShape});
        }
    }
}

static std::shared_ptr<arm_compute::TensorInfo> initTensorInfo(const arm_compute::TensorShape& tensorShape,
                              const arm_compute::DataType& dataType,
                              const arm_compute::DataLayout& dataLayout) {
    std::shared_ptr<arm_compute::TensorInfo> aclMemoryInfo = nullptr;
    if (dataType != arm_compute::DataType::UNKNOWN) {
        aclMemoryInfo = std::make_shared<arm_compute::TensorInfo>(
                tensorShape, 1,
                dataType,
                dataLayout);
    }
    return aclMemoryInfo;
}

static std::shared_ptr<arm_compute::Tensor> initTensor(const std::shared_ptr<arm_compute::TensorInfo>& aclMemoryInfo) {
    std::shared_ptr<arm_compute::Tensor> aclMemory = nullptr;
    if (aclMemoryInfo) {
        aclMemory = std::make_shared<arm_compute::Tensor>();
        aclMemory->allocator()->init(*aclMemoryInfo);
    }
    return aclMemory;
}

ACLCommonExecutor::ACLCommonExecutor() {
    for (int i = 0; i < ACLArgs::COUNT_OF_ARGS; ++i) {
        aclTensorAttrs.memoryUsageIndicator[i] = false;
    }
}

bool ACLCommonExecutor::update(const MemoryArgs &memory) {
    // Initialize ACL tensors params
    ACLShapes  aclMemoryShapes;
    ACLTypes   aclDataType{};
    ACLLayouts aclDataLayout{};
    for (auto& cpu_mem_ptr : memory) {
        const ACLArgs index = argConvert.at(cpu_mem_ptr.first);
        initACLTensorParams(cpu_mem_ptr.second, aclTensorAttrs,
                            aclMemoryShapes[index],
                            aclDataType[index],
                            aclDataLayout[index]);
    }

    // Update ACL tensors shapes
    updateTensorsShapes(aclMemoryShapes);

    // Initialize arm_compute::TensorInfo objects
    for (int i = 0; i < ACLArgs::COUNT_OF_ARGS; i++) {
        aclMemoryInfos[i] = initTensorInfo(aclMemoryShapes[i], aclDataType[i], aclDataLayout[i]);
    }

    // Validate arm_compute::TensorInfo objects for specific ACL function
    auto tensorsInfoValidateStatus = validateTensorsInfo(aclMemoryInfos);
    if (!tensorsInfoValidateStatus) {
        DEBUG_LOG("ACL operator validation failed: ", tensorsInfoValidateStatus.error_description());
        return false;
    }

    // Initialize arm_compute::Tensor objects
    for (int i = 0; i < ACLArgs::COUNT_OF_ARGS; i++) {
        aclMemoryTensors[i] = initTensor(aclMemoryInfos[i]);
        // Indicate that arm_compute::Tensor object can use import_memory function
        if (aclMemoryTensors[i]) {
            aclTensorAttrs.memoryUsageIndicator[i] = true;
        }
    }

    // Configure arm_compute::IFunction object
    configureThreadSafe([&] {
        iFunction = configureFunction(aclMemoryTensors);
    });
    return true;
}

void ACLCommonExecutor::execute(const MemoryArgs &memory) {
    // TODO: Move import_memory() to update() function - CVS-145871
    for (auto& cpu_mem_ptr : memory) {
        const ACLArgs index = argConvert.at(cpu_mem_ptr.first);
        if (aclTensorAttrs.memoryUsageIndicator[index]) {
            aclMemoryTensors[index]->allocator()->import_memory(memory.at(cpu_mem_ptr.first)->getData());
        }
    }
    iFunction->run();
}

ACLCommonExecutor::~ACLCommonExecutor() {
    for (int i = 0; i < ACLArgs::COUNT_OF_ARGS; i++) {
        if (aclTensorAttrs.memoryUsageIndicator[i]) {
            aclMemoryTensors[i]->allocator()->free();
        }
    }
}

}   // namespace intel_cpu
}   // namespace ov
