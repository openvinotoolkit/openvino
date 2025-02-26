// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <optional>

#include "acl_common_executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"

namespace ov::intel_cpu {

struct ACLFCAttrs {
    ov::element::Type inputPrecision;
    bool isConvertedWeights = false;
    bool isWeightsRepacked = false;
    bool weightsNonTransposed;
};

namespace acl_fc_executor {

VectorDims makeDummyInputDims(const Shape& inShape, const Shape& wShape);

VectorDims makeDummyOutputDims(const VectorDims& inShape, const VectorDims& wShape, const size_t out_rank);

DnnlMemoryDescPtr makeTransposedWeightDescriptor(const DnnlMemoryDescPtr& srcDesc, const DnnlMemoryDescPtr& dstDesc);

std::optional<MemoryPtr> convertWeightPrecision(const MemoryPtr& input,
                                                const MemoryPtr& output,
                                                ov::element::Type weightPrecision);

std::optional<MemoryPtr> reorderDataFallback(const MemoryPtr& input,
                                             const MemoryPtr& output,
                                             const ExecutorContext::CPtr& context);

MemoryPtr reorderData(const DnnlMemoryDescPtr& srcWeightDesc,
                      const DnnlMemoryDescPtr& dstWeightDesc,
                      const MemoryCPtr& weightsMem,
                      const ExecutorContext::CPtr& context);

MemoryPtr reorderWeights(const MemoryArgs& memory,
                         const ExecutorContext::CPtr context,
                         ACLFCAttrs& aclfcAttrs,
                         DnnlMemoryDescPtr dnnlSrcDesc,
                         DnnlMemoryDescPtr dnnlDstDesc);

MemoryPtr prepareWeightMemory(const MemoryArgs& memory,
                              const ExecutorContext::CPtr& context,
                              const FCAttrs& attrs,
                              ACLFCAttrs& aclfcAttrs,
                              const PostOps& postOps,
                              arm_compute::WeightFormat& expectedWeightFormat,
                              arm_compute::TensorInfo& weiTensorInfo);

arm_compute::TensorShape normalizeDimsTo2D(const arm_compute::TensorShape shape);

void updateFCTensorsShapes(ACLShapes& aclMemoryShapes);

class ACLWeightsConverter : public ACLCommonExecutor {
public:
    ACLWeightsConverter() = default;
    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override {}
    arm_compute::Status validateTensorsInfo(const ACLInfos& aclMemoryInfos) override;
    ACLFunction configureFunction(const ACLTensors& aclMemoryTensors) override;
};

class ACLWeightFormatGenerator : public ACLCommonExecutor {
public:
    ACLWeightFormatGenerator(const FCAttrs& attrs, const PostOps& postOps, const MemoryArgs& memory);
    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override;
    arm_compute::Status validateTensorsInfo(const ACLInfos& aclMemoryInfos) override;
    ACLFunction configureFunction(const ACLTensors& aclMemoryTensors) override;
    arm_compute::WeightFormat getOptImplWeightFormat() {
        return expectedWeightFormat;
    }

private:
    arm_compute::FullyConnectedLayerInfo fullyConnectedLayerInfo;
    arm_compute::WeightsInfo weightsInfo;
    ACLFCAttrs aclfcAttrs;
    arm_compute::WeightFormat expectedWeightFormat;
};

}  // namespace acl_fc_executor
}  // namespace ov::intel_cpu
