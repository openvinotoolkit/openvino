// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executors/convert_list.hpp"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

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
    bool created() const override;
    bool canBeInPlace() const override {
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

    const MemoryDesc& getInput() const {
        return *input;
    }
    const MemoryDesc& getOutput() const {
        return *output;
    }

    bool needPrepareParams() const override {
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

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
