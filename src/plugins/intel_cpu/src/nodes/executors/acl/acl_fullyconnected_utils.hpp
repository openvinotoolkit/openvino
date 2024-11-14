// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "acl_common_executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"

namespace ov {
namespace intel_cpu {

struct ACLFCAttrs {
    ov::element::Type inputPrecision;
    bool isConvertedWeights = false;
    bool weightsNonTransposed;
};

namespace acl_fc_executor {

VectorDims makeDummyInputDims(const Shape& inShape, const Shape& wShape);

VectorDims makeDummyOutputDims(const VectorDims& inShape, const VectorDims& wShape, const size_t out_rank);

MemoryPtr prepareWeightMemory(const MemoryArgs &memory,
                              const ExecutorContext::CPtr context,
                              const FCAttrs &attrs,
                              const ACLFCAttrs& aclfcAttrs,
                              const PostOps &postOps);

void updateFCTensorsShapes(ACLShapes& aclMemoryShapes);

class ACLWeightsConverter : public ACLCommonExecutor {
public:
    ACLWeightsConverter() = default;
    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override {}
    arm_compute::Status validateTensorsInfo(const ACLInfos & aclMemoryInfos) override;
    ACLFunction configureFunction(const ACLTensors & aclMemoryTensors) override;
};

class ACLWeightsTranspose : public ACLCommonExecutor {
public:
    ACLWeightsTranspose() = default;
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

class ACLWeightsReorder : public ACLCommonExecutor {
public:
    ACLWeightsReorder(arm_compute::WeightFormat inWeightFormat,
                      arm_compute::WeightFormat outWeightFormat)
                      : inWeightFormat(inWeightFormat), outWeightFormat(outWeightFormat) {}
    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override {}
    arm_compute::Status validateTensorsInfo(const ACLInfos & aclMemoryInfos) override;
    ACLFunction configureFunction(const ACLTensors & aclMemoryTensors) override;
private:
    arm_compute::WeightFormat inWeightFormat;
    arm_compute::WeightFormat outWeightFormat;
};

}  // namespace acl_fc_executor
}  // namespace intel_cpu
}  // namespace ov
