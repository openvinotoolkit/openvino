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
/* This class implementation is a temporal WA
   TODO: revise the implementation to remove the node reference*/
class SnippetShapeInfer : public ShapeInferEmptyPads {
public:
    SnippetShapeInfer(Snippet* node) : m_node(node) {}
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        return {m_node->shapeInfer(), ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    Snippet* m_node;
};

class SnippetShapeInferFactory : public ShapeInferFactory {
public:
    SnippetShapeInferFactory(Snippet* node) : m_node(node) {}
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<SnippetShapeInfer>(m_node);
    }

private:
    Snippet* m_node;
};
} // namespace node
} // namespace intel_cpu
} // namespace ov

