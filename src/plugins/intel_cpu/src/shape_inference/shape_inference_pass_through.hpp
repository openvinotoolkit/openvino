// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shape_inference_cpu.hpp"
#include "transformations/cpu_opset/common/op/read_value_with_subgraph.hpp"

namespace ov {
namespace intel_cpu {

/**
 * Specific shape inference implementation designed to cover cases where there are no actual output shape calculation
 * and all the output shapes are equal to the input tensor shapes.
 * 
 */
class ShapeInferPassThrough final : public ShapeInferEmptyPads {
public:
    ShapeInferPassThrough() = default;
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        OPENVINO_ASSERT(!input_shapes.empty());
        return {{input_shapes.front()}, ShapeInferStatus::success};
    }
    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

class MemInpSingleShapeInfer final : public ShapeInferEmptyPads {
public:
    explicit MemInpSingleShapeInfer(std::shared_ptr<ov::Model> body) : m_body(body) {}

    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const ParameterVector& parameters = m_body->get_parameters();
        const ResultVector& results = m_body->get_results();
        OPENVINO_ASSERT(parameters.size() == input_shapes.size(),
                        "Got invalid number of input shapes to reshape subgraph body");
        for (size_t i = 0; i < parameters.size(); ++i)
            parameters[i]->set_partial_shape(ov::PartialShape(input_shapes[i].get()));
        m_body->validate_nodes_and_infer_types();
        std::vector<VectorDims> outputDims;
        for (const auto& res : results)
            outputDims.emplace_back(res->get_input_partial_shape(0).get_shape());

        return {outputDims, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    std::shared_ptr<ov::Model> m_body;
};
class PassThroughShapeInferFactory final : public ShapeInferFactory {
public:
    explicit PassThroughShapeInferFactory() {}
    explicit PassThroughShapeInferFactory(const std::shared_ptr<ov::Node>& op) : m_op(op) {}
    explicit PassThroughShapeInferFactory(const std::shared_ptr<ov::Model>& body) : m_body(body) {}

    ShapeInferPtr makeShapeInfer() const override {
        if (m_op) {
            auto rvWithSubgraph = ov::as_type_ptr<ov::intel_cpu::ReadValueWithSubgraph>(m_op);
            if (rvWithSubgraph) {
                return std::make_shared<MemInpSingleShapeInfer>(rvWithSubgraph->get_function());
            }
        }
        if (m_body) {
            return std::make_shared<MemInpSingleShapeInfer>(m_body);
        }

        return std::make_shared<ShapeInferPassThrough>();
    }

private:
    std::shared_ptr<ov::Node> m_op = nullptr;
    std::shared_ptr<ov::Model> m_body = nullptr;
};

} // namespace intel_cpu
} // namespace ov
