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
    explicit SnippetShapeInfer(const std::shared_ptr<snippets::op::Subgraph>& s) : m_subgraph(s) {
        m_status_map[snippets::ShapeInferStatus::success] = ov::intel_cpu::ShapeInferStatus::success;
        m_status_map[snippets::ShapeInferStatus::skip] = ov::intel_cpu::ShapeInferStatus::skip;
    }
    Result infer(
            const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
            const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const auto& snippets_result = m_subgraph->shape_infer(input_shapes);
        OPENVINO_ASSERT(m_status_map.count(snippets_result.status) != 0, "Failed to map snippets shapeInfer status to the plugin one");
        return {snippets_result.dims, m_status_map.at(snippets_result.status)};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    std::shared_ptr<snippets::op::Subgraph> m_subgraph;
    std::map<snippets::ShapeInferStatus, ov::intel_cpu::ShapeInferStatus> m_status_map;
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

