// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "acl_common_executor.hpp"
#include "ov_optional.hpp"
#include "nodes/executors/fullyconnected_config.hpp"

namespace ov {
namespace intel_cpu {

struct ACLFCAttrs {
    ov::element::Type inputPrecision;
    bool isConvertedWeights = false;
    bool isWeightsRepacked = false;
    bool weightsNonTransposed;
};

namespace acl_fc_executor {

VectorDims makeDummyInputDims(const Shape& inShape, const Shape& wShape);

VectorDims makeDummyOutputDims(const VectorDims& inShape, const VectorDims& wShape, const size_t out_rank);

DnnlMemoryDescPtr makeTransposedWeightDescriptor(const DnnlMemoryDescPtr srcDesc,
                                                        const DnnlMemoryDescPtr dstDesc);

ov::optional<MemoryPtr> convertWeightPrecision(MemoryPtr input,
                                                      MemoryPtr output,
                                                      ov::element::Type weightPrecision);

ov::optional<MemoryPtr> reorderDataFallback(MemoryPtr input,
                                                   MemoryPtr output,
                                                   ExecutorContext::CPtr context);

MemoryPtr reorderData(DnnlMemoryDescPtr srcWeightDesc,
                             DnnlMemoryDescPtr dstWeightDesc,
                             MemoryCPtr weightsMem,
                             ExecutorContext::CPtr context);

MemoryPtr reorderWeights(const MemoryArgs &memory,
                                const ExecutorContext::CPtr context,
                                ACLFCAttrs& aclfcAttrs,
                                DnnlMemoryDescPtr dnnlSrcDesc,
                                DnnlMemoryDescPtr dnnlDstDesc);

MemoryPtr prepareWeightMemory(const MemoryArgs &memory,
                                     const ExecutorContext::CPtr context,
                                     const FCAttrs &attrs,
                                     ACLFCAttrs& aclfcAttrs,
                                     const PostOps &postOps,
                                     arm_compute::WeightFormat& expectedWeightFormat,
                                     arm_compute::TensorInfo& weiTensorInfo);

arm_compute::TensorShape normalizeDimsTo2D(const arm_compute::TensorShape shape);

void updateFCTensorsShapes(ACLShapes& aclMemoryShapes);

class ACLWeightsConverter : public ACLCommonExecutor {
public:
    ACLWeightsConverter() = default;
    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override {}
    arm_compute::Status validateTensorsInfo(const ACLInfos & aclMemoryInfos) override;
    ACLFunction configureFunction(const ACLTensors & aclMemoryTensors) override;
};

class ACLWeightFormatGenerator : public ACLCommonExecutor {
public:
    ACLWeightFormatGenerator(const FCAttrs& attrs,
                             const PostOps& postOps,
                             const MemoryArgs& memory);
    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override;
    arm_compute::Status validateTensorsInfo(const ACLInfos & aclMemoryInfos) override;
    ACLFunction configureFunction(const ACLTensors & aclMemoryTensors) override;
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
}  // namespace intel_cpu
}  // namespace ov