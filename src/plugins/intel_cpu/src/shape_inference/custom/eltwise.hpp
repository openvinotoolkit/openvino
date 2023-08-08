// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>
#include "shape_inference/shape_inference_cpu.hpp"
#include "transformations/cpu_opset/common/op/power_static.hpp"

#pragma once
namespace ov {
namespace intel_cpu {
namespace node {
using Result = IShapeInfer::Result;

/**
 * Implements Eltwise shape inference algorithm. The algorithm is based on broadcasting all the input shapes
 * according to the NUMPY broadcast rule. This implementation is more lightweight than the ngraph one.
 *
 */
class EltwiseShapeInfer : public ShapeInferEmptyPads {
public:
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

class NoBroadCastEltwiseShapeInfer : public ShapeInferEmptyPads {
public:
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

class EltwiseShapeInferFactory : public ShapeInferFactory {
public:
    EltwiseShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        const auto& autob = m_op->get_autob();
        if (autob.m_type == ov::op::AutoBroadcastType::NONE
                && (!ov::is_type<const ov::intel_cpu::PowerStaticNode>(m_op))) {
            return std::make_shared<NoBroadCastEltwiseShapeInfer>();
        } else {
            return std::make_shared<EltwiseShapeInfer>();
        }
    }

private:
    std::shared_ptr<ov::Node> m_op;
};
} // namespace node
} // namespace intel_cpu
} // namespace ov

