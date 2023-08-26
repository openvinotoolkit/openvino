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
    SnippetShapeInfer(std::shared_ptr<ov::Model> body) : m_body(body) {}
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        auto broadcast_merge = [](VectorDims& dst, const VectorDims& src) {
            // Ranks are both static.
            auto dst_rank = dst.size();
            auto src_rank = src.size();
            const auto new_rank = std::max(dst_rank, src_rank);
            dst.insert(dst.begin(), new_rank - dst_rank, 1);
            for (size_t i = 0; i < new_rank; i++) {
                auto srci = i < (new_rank - src_rank) ? 1 : src[i - (new_rank - src_rank)];
                if (dst[i] != srci && srci != Shape::UNDEFINED_DIM) {
                    if (dst[i] == 1 || dst[i] == Shape::UNDEFINED_DIM) {
                        dst[i] = srci;
                    } else {
                        if (srci != 1) {
                            IE_THROW() << "Got imcompatible input shapes in snippets shape infer";
                        }
                    }
                }
            }
        };

        const size_t out_size = m_body->get_output_size();
        if (out_size == 1) {
            VectorDims masterShape;
            for (size_t i = 0; i < input_shapes.size(); i++) {
                if (i == 0)
                    masterShape = input_shapes[i];
                else
                    broadcast_merge(masterShape, input_shapes[i]);
            }
            size_t output_rank = m_body->get_output_partial_shape(0).rank().get_length();
            if (output_rank > masterShape.size()) {
                masterShape.insert(masterShape.begin(), output_rank - masterShape.size(), 1);
            }
            return {{masterShape}, ShapeInferStatus::success};
        } else {
            std::vector<VectorDims> outputDims;
            std::vector<ov::Shape> new_shapes;
            for (const auto& s : input_shapes)
                new_shapes.emplace_back(s);
            auto& params = m_body->get_parameters();
            if (params.size() != input_shapes.size()) {
                IE_THROW() << "Got invalid number of input shapes to reshape subgraph body";
            }
            for (size_t i = 0; i < params.size(); ++i) {
                params[i]->set_partial_shape(new_shapes[i]);
            }
            m_body->validate_nodes_and_infer_types();
            for (const auto& res : m_body->get_results()) {
                auto& pshape = res->get_input_partial_shape(0);
                if (!pshape.is_static()) {
                    IE_THROW() << "Subgraph inferred dynamic output shape during reshape with static inputs";
                }
                outputDims.emplace_back(pshape.get_shape());
            }

            return {outputDims, ShapeInferStatus::success};
        }
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    std::shared_ptr<ov::Model> m_body;
};

class SnippetShapeInferFactory : public ShapeInferFactory {
public:
    SnippetShapeInferFactory(const std::shared_ptr<ov::Node>& op) {
        auto subgraph = ov::as_type_ptr<snippets::op::Subgraph>(op);
        snippet_body = subgraph->body_ptr()->clone();
    }
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<SnippetShapeInfer>(snippet_body);
    }

private:
    std::shared_ptr<ov::Model> snippet_body = nullptr;
};
} // namespace node
} // namespace intel_cpu
} // namespace ov

