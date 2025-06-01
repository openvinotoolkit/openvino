// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "executors/interpolate.hpp"
#include "executors/interpolate_config.hpp"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "executors/executor_factory.hpp"

namespace ov::intel_cpu::node {

class Interpolate : public Node {
public:
    Interpolate(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool canBeInPlace() const override {
        return false;
    }
    bool canFuse(const NodePtr& node) const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;

    inline int get_scale_id() const;
    inline int get_axis_id() const;

private:
    bool is_version11 = true;
    InterpolateAttrs interpAttrs;
    size_t dataRank = 0;
    std::shared_ptr<legacy::InterpolateExecutorBaseLegacy> execPtr = nullptr;

    void setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims);

    static VectorDims getPaddedInputShape(const VectorDims& srcDims,
                                          const std::vector<int>& padBegin,
                                          const std::vector<int>& padEnd);
    std::vector<float> getScales(const VectorDims& srcDimPad, const VectorDims& dstDim);

    bool hasPad = false;

    bool isAxesSpecified = false;
    std::vector<int> axes;
    std::vector<float> scales;
    bool isScaleConstant = false;

    // 6 ptrs for each quantization, 2 ptrs for each depth_wise
    std::vector<const void*> postOpsDataPtrs;

    std::vector<float> lastScales;
    std::vector<int32_t> lastSizes;

    VectorDims lastOutputDims;

    bool canUseAclExecutor = false;
    std::shared_ptr<InterpolateExecutor> aclExecPtr = nullptr;
//////////////////////
    MemoryArgs memory;
    ExecutorFactoryPtr<InterpolateAttrs> factory;
    ExecutorPtr executor = nullptr;
};

}  // namespace ov::intel_cpu::node
