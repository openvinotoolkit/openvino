// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_shape.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/executors/convert.hpp"
#include "nodes/node_config.h"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class Convert : public Node {
public:
    Convert(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    Convert(const Shape& shape,
            const ov::element::Type& inPrc,
            const ov::element::Type& outPrc,
            const std::string& nodeName,
            const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void prepareParams() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;
    [[nodiscard]] bool canBeInPlace() const override {
        return false;
    }

    // This is the interface extension designed to provide inp and output tensor descriptors without the CNNLayer.
    // In that case the Convert node is instantiated with default CNNLayer and inp/out tensor descriptors are set via
    // this method. This is useful if the Convert node is added to the graph as an auxiliary operation at the Graph
    // initialization stage.
    void setDescs(const MemoryDesc& input, const MemoryDesc& output) {
        this->input = input.clone();
        this->output = output.clone();
    }

    [[nodiscard]] const MemoryDesc& getInput() const {
        return *input;
    }
    [[nodiscard]] const MemoryDesc& getOutput() const {
        return *output;
    }

    [[nodiscard]] bool needPrepareParams() const override {
        return inputShapesModified();
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    static bool isSupportedDesc(const MemoryDesc& desc);

private:
    MemoryDescPtr input;
    MemoryDescPtr output;
    ConvertParams convertParams;
    std::shared_ptr<ConvertExecutor> execPtr = nullptr;
    NodeConfig config;
};

}  // namespace ov::intel_cpu::node
