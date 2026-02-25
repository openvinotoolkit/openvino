// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <openvino/op/constant.hpp>
#include <string>

#include "cpu_memory.h"
#include "cpu_shape.h"
#include "edge.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class Input : public Node {
public:
    struct InputConfig {
        MemoryDescPtr desc;
        bool inPlace;
    };

    struct OutputConfig {
        OutputConfig() = default;
        OutputConfig(bool useParentMemoryDesc_, bool inPlace_)
            : useParentMemoryDescForOutput(useParentMemoryDesc_),
              inPlace(inPlace_) {}

        OutputConfig(MemoryDescPtr desc_, bool inPlace_) : desc(std::move(desc_)), inPlace(inPlace_) {}

        // @todo better to use memory desc with any layout and undefined precision
        MemoryDescPtr desc = nullptr;
        bool useParentMemoryDescForOutput = false;
        bool inPlace = false;
    };

    Input(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    Input(const Shape& shape,
          const ov::element::Type& prc,
          const std::string& name,
          const std::string& type,
          const GraphContext::CPtr& context);

    Input(const MemoryDescPtr& memDesc,
          const std::string& name,
          const std::string& type,
          const GraphContext::CPtr& context);

    Input(const MemoryPtr& mem, const std::string& name, const std::string& type, const GraphContext::CPtr& context);

    Input(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context, const InputConfig& config);

    Input(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context, const OutputConfig& config);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void selectOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    bool created() const override;
    void resolveInPlaceEdges(Edge::LOOK look) override;

    void withMeanImage();
    MemoryCPtr getMemoryPtr() const;

    void execute(const dnnl::stream& strm) override {}
    void executeDynamicImpl(const dnnl::stream& strm) override {}

    bool neverExecute() const override {
        return true;
    }
    bool isExecutable() const override {
        return false;
    }

    bool needShapeInfer() const override {
        return false;
    }
    bool needPrepareParams() const override {
        return false;
    }

private:
    void cloneBlobIfRequired();
    void initSupportedPdDefault();
    void initSupportedPdFromMemDesc();

    std::shared_ptr<ov::op::v0::Constant> m_constOp;
    MemoryCPtr memoryPtr;
    bool isMeanImage = false;
    MemoryDescPtr extMemDesc = nullptr;
    bool m_useParentMemoryDescForOutput = false;
    bool m_isInPlace = false;
};

}  // namespace ov::intel_cpu::node
