// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_postops_composer_legacy.h"
#include "graph_context.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class Eltwise : public Node {
public:
    Eltwise(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    bool canBeInPlace() const override;
    bool canFuseConvert(const NodePtr& convertNode);
    bool canFuseParent(const NodePtr& parentNode) const;
    bool canFuse(const NodePtr& node) const override;

    void appendPostOps(dnnl::post_ops& ops,
                       const VectorDims& postOpDims,
                       std::unordered_map<int, MemoryPtr>& postOpsMem,
                       int channelAxis) override;
    void appendPostOps(dnnl::post_ops& ops,
                       const VectorDims& postOpDims,
                       std::vector<const void*>& postOpsMem,
                       int channelAxis) override;
    bool appendAttrPostOps(DnnlPostOpsComposerLegacy& dnnlpoc,
                           bool isLastPostOp,
                           dnnl::memory::data_type outDataType,
                           bool allowBinary = true);
    void fuseInto(NodePtr& parentNode) override;
    ov::element::Type getRuntimePrecision() const override;

    float getAlpha() const {
        return m_attrs.data.alpha;
    }
    float getBeta() const {
        return m_attrs.data.beta;
    }
    float getGamma() const {
        return m_attrs.data.gamma;
    }
    const std::vector<float>& getScales() const {
        return m_attrs.scales;
    }
    const std::vector<float>& getShifts() const {
        return m_attrs.shifts;
    }

    const EltwiseAttrs& attrs() const {
        return m_attrs;
    }

    dnnl::algorithm getOneDnnAlgorithm() const {
        return m_attrs.data.onednnAlgorithm;
    }

    bool isWithBroadcast();
    bool isSpecialConvolutionAddFusing() const {
        return m_attrs.specialConvolutionAddFusing;
    }

    void prepareParams() override;
    void createPrimitive() override;

    void executeDynamicImpl(const dnnl::stream& strm) override;

    EltwiseBroadcastingPolicy getBroadcastingPolicy() const {
        return m_attrs.broadcastingPolicy;
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    using Initializer = std::function<void(const std::shared_ptr<ov::Node>&, Eltwise& node)>;
    static const std::map<const ov::DiscreteTypeInfo, Initializer>& getInitializers();
    static EltwiseBroadcastingPolicy determineBroadcastingPolicy(const std::shared_ptr<ov::Node>& op);

    size_t getOpInputsNum() const;
    void init() override;

    void appendMemory(const std::vector<float>& data, MemoryPtr& memPtr, std::vector<MemoryPtr>& postOpsMem);
    static void appendMemory(const std::vector<float>& data,
                             [[maybe_unused]] MemoryPtr& memPtr,
                             std::vector<const void*>& postOpsMem);
    template <typename T>
    void appendPostOpsImpl(dnnl::post_ops& ops,
                           const VectorDims& postOpDims,
                           std::vector<T>& postOpsMem,
                           int channelAxis = 1);

    ExecutorFactoryPtr<EltwiseAttrs> m_factory;
    ExecutorPtr m_executor;
    EltwiseAttrs m_attrs;
    MemoryArgs m_memory;

    std::vector<float> m_depthwiseData;
    MemoryPtr m_depthwiseMemory;
    size_t m_depthwiseDataSize = 0;
};

}  // namespace ov::intel_cpu::node
