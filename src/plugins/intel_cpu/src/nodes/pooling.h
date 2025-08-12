// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/dnnl_executor.h"
#include "cpu_shape.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/executors/pooling.hpp"
#include "nodes/node_config.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class Pooling : public Node {
public:
    Pooling(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    std::vector<dnnl::memory::format_tag> getAvailableFormatsForDims(const Shape& dims) const override;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initDescriptor(const NodeConfig& config) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    void prepareParams() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    AttrPtr initPrimitiveAttr() override;

private:
    using executorPtr = std::shared_ptr<DnnlExecutorLegacy>;
    executorPtr dnnlExecPtr = nullptr;

    void setPostOps(dnnl::primitive_attr& attr);

    PoolingAttrs poolingAttrs;

    std::shared_ptr<PoolingExecutor> execPtr = nullptr;

    void initEffectiveAttributes(const Shape& inShape, const Shape& outShape);
    dnnl::algorithm getPoolingAlgorithm() const;
    dnnl::pooling_forward::primitive_desc createDescriptorInternal(const dnnl::memory::desc& in_candidate,
                                                                   const dnnl::memory::desc& out_candidate,
                                                                   dnnl::algorithm alg);

    AttrPtr pAttr;

    Shape inShape;

    bool isNotMaxPool1 = false;
    bool useACL = false;
};

}  // namespace ov::intel_cpu::node
