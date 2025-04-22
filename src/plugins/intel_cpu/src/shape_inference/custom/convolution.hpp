// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <cstddef>
#include <utility>

#include "cpu_types.h"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

VectorDims convolution_shape_infer(const VectorDims& data_shape,
                                   const VectorDims& filters_shape,
                                   const ov::Strides& strides,
                                   const ov::Strides& dilations,
                                   const ov::CoordinateDiff& pads_begin,
                                   const ov::CoordinateDiff& pads_end,
                                   bool auto_padding,
                                   bool isGrouped);

using Result = IShapeInfer::Result;
class ConvolutionShapeInfer : public ShapeInferEmptyPads {
public:
    ConvolutionShapeInfer(ov::Strides strides,
                          ov::Strides dilations,
                          ov::CoordinateDiff pads_begin,
                          ov::CoordinateDiff pads_end,
                          bool auto_padding,
                          bool isGrouped = false)
        : m_strides(std::move(strides)),
          m_dilations(std::move(dilations)),
          m_pads_begin(std::move(pads_begin)),
          m_pads_end(std::move(pads_end)),
          m_auto_padding(auto_padding),
          m_isGrouped(isGrouped) {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

private:
    ov::Strides m_strides;
    ov::Strides m_dilations;
    ov::CoordinateDiff m_pads_begin;
    ov::CoordinateDiff m_pads_end;
    bool m_auto_padding;
    bool m_isGrouped;
};

class ConvolutionShapeInferFactory : public ShapeInferFactory {
public:
    ConvolutionShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};

}  // namespace ov::intel_cpu::node
