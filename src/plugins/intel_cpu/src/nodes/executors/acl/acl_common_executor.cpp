// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_common_executor.hpp"

#include <ostream>

#include "acl_utils.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

const std::unordered_map<int, ACLArgs> argConvert = {
    {ARG_SRC_0, ACL_SRC_0},
    {ARG_SRC_1, ACL_SRC_1},
    {ARG_SRC_1, ACL_SRC_2},
    {ARG_BIAS,  ACL_BIAS},
    {ARG_WEI,   ACL_WEI},
    {ARG_DST,   ACL_DST},
};

using ACLMemoryTypes   = std::array<arm_compute::DataType,   ACLArgs::COUNT_OF_ARGS>;
using ACLMemoryLayouts = std::array<arm_compute::DataLayout, ACLArgs::COUNT_OF_ARGS>;

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

ACLInfo ACLCommonExecutor::initTensorInfo(const arm_compute::TensorShape& tensorShape,
                                          const arm_compute::DataType& dataType,
                                          const arm_compute::DataLayout& dataLayout) {
    ACLInfo aclMemoryInfo = nullptr;
    if (dataType != arm_compute::DataType::UNKNOWN) {
        aclMemoryInfo = std::make_shared<arm_compute::TensorInfo>(
                tensorShape, 1,
                dataType,
                dataLayout);
    }
    return aclMemoryInfo;
}

static ACLMemory initTensor(const ACLInfo& aclMemoryInfo) {
    ACLMemory aclMemory = nullptr;
    if (aclMemoryInfo) {
        aclMemory = std::make_shared<arm_compute::Tensor>();
        aclMemory->allocator()->init(*aclMemoryInfo);
    }
    return aclMemory;
}

bool ACLCommonExecutor::update(const MemoryArgs &memory) {
    // Initialize ACL tensors params
    ACLMemoryShapes  aclMemoryShapes;
    ACLMemoryTypes   aclDataType{};
    ACLMemoryLayouts aclDataLayout{};
    for (auto& cpu_mem_ptr : memory) {
        if (cpu_mem_ptr.second->getSize() == 0) {
            continue;
        }
        const ACLArgs index = argConvert.at(cpu_mem_ptr.first);
        initACLTensorParams(cpu_mem_ptr.second, aclTensorAttrs,
                            aclMemoryShapes[index],
                            aclDataType[index],
                            aclDataLayout[index]);
    }

    // Update ACL tensors shapes
    updateTensorsShapes(aclMemoryShapes);

    // Initialize arm_compute::TensorInfo objects
    ACLMemoryInfo aclMemoryInfos;
    for (int i = 0; i < ACLArgs::COUNT_OF_ARGS; i++) {
        aclMemoryInfos[i] = initTensorInfo(aclMemoryShapes[i], aclDataType[i], aclDataLayout[i]);
    }

    // Validate arm_compute::TensorInfo objects for specific ACL function
    auto tensorsInfoValidateStatus = validateTensorsInfo(aclMemoryInfos);
    if (!tensorsInfoValidateStatus) {
        DEBUG_LOG("ACL operator validation was failed: ", tensorsInfoValidateStatus.error_description());
        return false;
    }

    // Initialize arm_compute::Tensor objects
    for (int i = 0; i < ACLArgs::COUNT_OF_ARGS; i++) {
        aclMemoryTensors[i] = initTensor(aclMemoryInfos[i]);
    }

    // Configure arm_compute::IFunction object
    configureThreadSafe([&] {
        iFunction = configureFunction(aclMemoryTensors);
    });

    for (auto& cpu_mem_ptr : memory) {
        const ACLArgs index = argConvert.at(cpu_mem_ptr.first);
        if (aclMemoryTensors[index]) {
            aclMemoryTensors[index]->allocator()->import_memory(memory.at(cpu_mem_ptr.first)->getData());
        }
    }
    return true;
}

//namespace {
//std::ostream& operator<<(std::ostream& os, const arm_compute::ITensorInfo* tensor_info) {
//    const auto data_type = tensor_info->data_type();
//    switch (data_type) {
//        case arm_compute::DataType::S8: {
//            return os << "S8";
//        }
//        case arm_compute::DataType::QSYMM8: {
//            return os << "QSYMM8";
//        }
//        case arm_compute::DataType::QASYMM8: {
//            return os << "QASYMM8";
//        }
//        case arm_compute::DataType::QASYMM8_SIGNED: {
//            return os << "QASYMM8_SIGNED";
//        }
//        case arm_compute::DataType::S32: {
//            return os << "S32";
//        }
//        case arm_compute::DataType::F32: {
//            return os << "F32";
//        }
//        default: {
//            return os << "[UNKNOWN]";
//        }
//    }
//}
//} // namespace

void ACLCommonExecutor::execute(const MemoryArgs &memory) {
//    for (auto& cpu_mem_ptr : memory) {
//        const ACLArgs index = argConvert.at(cpu_mem_ptr.first);
//        if (aclMemoryTensors[index]) {
//            aclMemoryTensors[index]->allocator()->import_memory(memory.at(cpu_mem_ptr.first)->getData());
//        }
//    }

//    for (auto index = 0; index < aclMemoryTensors.size(); ++index) {
//        const auto& tensor = aclMemoryTensors[index];
//        if ((tensor == nullptr) || (index == ACLArgs::ACL_DST)) {
//            continue;
//        }
//
//        if (index == ACLArgs::ACL_SRC_0) {
//            std::cout << "src0 ";
//        } else if (index == ACLArgs::ACL_WEI) {
//            std::cout << "src1 ";
//        } else if (index == ACLArgs::ACL_BIAS) {
//            std::cout << "biases ";
//        } else {
//            std::cout << "[UNKNOWN] ";
//        }
//        std::cout << tensor->info() << ":" << std::endl;
//        tensor->print(std::cout);
//    }

    iFunction->run();

//    {
//        std::shared_ptr<arm_compute::Tensor> tensor = aclMemoryTensors[ACLArgs::ACL_DST];
//        std::cout << "dst " << tensor->info() << ":" << std::endl;
//        tensor->print(std::cout);
//    }
}

ACLCommonExecutor::~ACLCommonExecutor() {
    for (int i = 0; i < ACLArgs::COUNT_OF_ARGS; i++) {
        if (aclMemoryTensors[i]) {
            aclMemoryTensors[i]->allocator()->free();
        }
    }
}

}   // namespace intel_cpu
}   // namespace ov
