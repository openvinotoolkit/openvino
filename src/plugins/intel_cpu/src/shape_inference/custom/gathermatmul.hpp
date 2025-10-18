// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cpu_memory.h"
#include "openvino/core/node.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

using Result = IShapeInfer::Result;

class GatherMatmulShapeInfer : public ShapeInferEmptyPads {
public:
    GatherMatmulShapeInfer(size_t output_rank, bool transpose_a, bool transpose_b)
        : m_out_rank(output_rank),
          m_transpose_a(transpose_a),
          m_transpose_b(transpose_b) {}

    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    size_t m_out_rank;
    bool m_transpose_a;
    bool m_transpose_b;
};

class GatherMatmulShapeInferFactory : public ShapeInferFactory {
public:
    explicit GatherMatmulShapeInferFactory(const std::shared_ptr<ov::Node>& op) : m_op(op) {}

    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override {
        const auto output_rank = m_op->get_output_partial_shape(0).rank().get_length();
        // BatchGatherMatmul has fixed transpose settings: transpose_a=false, transpose_b=true
        return std::make_shared<GatherMatmulShapeInfer>(output_rank, false, true);
    }

private:
    std::shared_ptr<const ov::Node> m_op;
};

}  // namespace ov::intel_cpu::node
