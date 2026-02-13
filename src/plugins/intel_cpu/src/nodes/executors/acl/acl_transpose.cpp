// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_transpose.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/runtime/NEON/functions/NEPermute.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "acl_utils.hpp"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/transpose.hpp"
#include "nodes/executors/transpose_config.hpp"
#include "openvino/core/except.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {
namespace {

struct ACLTransposeConfig {
    arm_compute::PermutationVector order;
    arm_compute::TensorInfo srcTensorInfo;
    arm_compute::TensorInfo dstTensorInfo;
};

std::optional<ACLTransposeConfig> prepareAclTransposeConfig(const TransposeParams& transposeParams,
                                                            const MemoryDescPtr& srcDesc,
                                                            const MemoryDescPtr& dstDesc) {
    if ((srcDesc->hasLayoutType(LayoutType::ncsp) || dstDesc->hasLayoutType(LayoutType::ncsp)) &&
        (srcDesc->hasLayoutType(LayoutType::nspc) || dstDesc->hasLayoutType(LayoutType::nspc))) {
        DEBUG_LOG("NEPermute does not support layout:",
                  " src: ",
                  srcDesc->serializeFormat(),
                  " dst: ",
                  dstDesc->serializeFormat());
        return std::nullopt;
    }
    if (srcDesc->getShape().getRank() > 4) {
        DEBUG_LOG("NEPermute supports up to 4D input tensor. Passed tensor rank: ", srcDesc->getShape().getRank());
        return std::nullopt;
    }
    if (srcDesc->getPrecision() != dstDesc->getPrecision()) {
        DEBUG_LOG("NEPermute requires the same input and output precisions");
        return std::nullopt;
    }

    auto inputOrder = transposeParams.permuteParams.order;
    if (inputOrder.empty()) {
        inputOrder.resize(srcDesc->getShape().getRank());
        std::iota(inputOrder.begin(), inputOrder.end(), 0);
    }

    std::vector<size_t> vec;
    if (srcDesc->hasLayoutType(LayoutType::nspc)) {
        auto changeLayoutToNhwc = [](VectorDims shape) -> VectorDims {
            std::swap(shape[1], shape[2]);
            std::swap(shape[2], shape[3]);
            return shape;
        };
        auto srcDims = changeLayoutToNhwc(srcDesc->getShape().getStaticDims());
        auto dstDims = changeLayoutToNhwc(dstDesc->getShape().getStaticDims());
        for (int i = inputOrder.size() - 1; i >= 0; --i) {
            auto it = std::find(srcDims.rbegin(), srcDims.rend(), dstDims[i]);
            auto index = it - srcDims.rbegin();
            vec.push_back(index);
        }
    } else {
        for (unsigned int i = 0; i < inputOrder.size(); ++i) {
            vec.push_back(axisCast(inputOrder[i], inputOrder.size()));
        }
        std::reverse(vec.begin(), vec.end());
    }
    arm_compute::PermutationVector order;
    for (unsigned int i = 0; i < inputOrder.size(); ++i) {
        order.set(i, vec[i]);
    }

    auto srcDims = shapeCast(srcDesc->getShape().getDims());
    auto dstDims = shapeCast(dstDesc->getShape().getDims());

    if (srcDesc->hasLayoutType(LayoutType::nspc) && dstDesc->hasLayoutType(LayoutType::nspc)) {
        changeLayoutToNH_C({&srcDims, &dstDims});
    }

    ACLTransposeConfig config{
        order,
        arm_compute::TensorInfo(srcDims,
                                1,
                                precisionToAclDataType(srcDesc->getPrecision()),
                                getAclDataLayoutByMemoryDesc(srcDesc)),
        arm_compute::TensorInfo(dstDims,
                                1,
                                precisionToAclDataType(dstDesc->getPrecision()),
                                getAclDataLayoutByMemoryDesc(dstDesc)),
    };

    arm_compute::Status status =
        arm_compute::NEPermute::validate(&config.srcTensorInfo, &config.dstTensorInfo, config.order);
    if (!status) {
        DEBUG_LOG("NEPermute validation failed: ", status.error_description());
        return std::nullopt;
    }

    return config;
}

}  // namespace

ACLTransposeExecutor::ACLTransposeExecutor(const TransposeParams& transposeParams,
                                           const MemoryDescPtr& srcDesc,
                                           const MemoryDescPtr& dstDesc,
                                           ExecutorContext::CPtr context)
    : TransposeExecutor(std::move(context)),
      acl_permute(std::make_unique<arm_compute::NEPermute>()) {
    auto aclConfig = prepareAclTransposeConfig(transposeParams, srcDesc, dstDesc);
    OPENVINO_ASSERT(aclConfig, "ACLTransposeExecutor does not support provided configuration");

    srcTensor.allocator()->init(aclConfig->srcTensorInfo);
    dstTensor.allocator()->init(aclConfig->dstTensorInfo);
    configureThreadSafe([&] {
        acl_permute->configure(&srcTensor, &dstTensor, aclConfig->order);
    });
}

bool ACLTransposeExecutor::supports(const TransposeConfig& config) {
    return prepareAclTransposeConfig(config.attrs.params, config.descs.at(ARG_SRC), config.descs.at(ARG_DST))
        .has_value();
}

ExecutorPtr ACLTransposeExecutor::create(const TransposeAttrs& attrs,
                                         const MemoryArgs& memory,
                                         const ExecutorContext::CPtr& context) {
    const auto& descs = attrs.descs.empty() ? memoryDescsFromMemory(memory) : attrs.descs;
    return std::make_shared<ACLTransposeExecutor>(attrs.params, descs.at(ARG_SRC), descs.at(ARG_DST), context);
}

void ACLTransposeExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    OPENVINO_ASSERT(acl_permute, "ACL transpose executor is not initialized");
    srcTensor.allocator()->import_memory(src[0]->getData());
    dstTensor.allocator()->import_memory(dst[0]->getData());

    acl_permute->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}  // namespace ov::intel_cpu
