// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include "kernels/gather_uni_kernel.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class Gather : public Node {
public:
    Gather(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    void createOrUpdateJitKernelIfNeeded();
    void initializePerThreadParams();
    void executeDynamicImpl(dnnl::stream strm) override;
    bool needPrepareParams() const override;
    void prepareParams() override;
    std::vector<VectorDims> shapeInfer() const override;

private:
    void execReference();

    bool isDataShapeStat = false;
    bool isIdxShapeStat = false;
    bool isAxisInputConst = false;
    bool reverseIndexing = false;
    int axis = 0;
    int batchDims = 0;
    int dataSrcRank = 1;
    GatherShapeParameters shapeParameters;

    std::vector<GatherShapeParameters::PerThread> execParamsPerThread;

    static constexpr size_t GATHER_DATA = 0;
    static constexpr size_t GATHER_INDICES = 1;
    static constexpr size_t GATHER_AXIS = 2;

    std::shared_ptr<jitGatherKernelInterface> jitKernel;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
