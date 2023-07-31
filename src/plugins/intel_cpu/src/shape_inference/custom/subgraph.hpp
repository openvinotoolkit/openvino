// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>
#include "shape_inference/shape_inference_cpu.hpp"

#pragma once
namespace ov {
namespace intel_cpu {
namespace node {
using Result = IShapeInfer::Result;

class SnippetShapeInfer : public ShapeInferEmptyPads {
public:
    explicit SnippetShapeInfer(const std::shared_ptr<snippets::op::Subgraph>& s) : m_subgraph(s) {}
    Result infer(
            const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
            const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        return {m_subgraph->shape_infer(input_shapes).dims, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    std::shared_ptr<snippets::op::Subgraph> m_subgraph;
};

class SnippetShapeInferFactory : public ShapeInferFactory {
public:
    explicit SnippetShapeInferFactory(const std::shared_ptr<ov::Node>& op) {
        m_subgraph = ov::as_type_ptr<snippets::op::Subgraph>(op);
        OPENVINO_ASSERT(m_subgraph, "Invalid node type detected in SnippetShapeInferFactory");
    }
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<SnippetShapeInfer>(m_subgraph);
    }

private:
    std::shared_ptr<snippets::op::Subgraph> m_subgraph = nullptr;
};
} // namespace node
} // namespace intel_cpu
} // namespace ov

