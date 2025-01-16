// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu_memory.h>
#include <openvino/core/node.hpp>
#include <openvino/core/coordinate_diff.hpp>
#include "shape_inference_status.hpp"

namespace ov {
namespace intel_cpu {
/**
 * This is CPU plugin specific shape inference interface.
 *
 */
class IShapeInfer {
public:
    using port_mask_t = uint32_t;

    struct Result {
        std::vector<VectorDims> dims;
        ShapeInferStatus status;
    };

public:
    virtual ~IShapeInfer() = default;

    /**
     * @brief This method actually performs all the necessary shape inference computations
     *
     * @param input_shapes are the input tensors shapes
     * @param data_dependency are the input tensors data, which are required by the shape inference algorithm. To define
     * which inputs data are actually required, get_port_mask() is used
     * @return ShapeInferResult which contains resulting array of calculated shapes (per each output port) plus status
     * of the shape infer call
     */
    virtual Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) = 0;

    /**
     * @brief Shape inference implementation may generate padding as by-product, these APIs is designed to retrieve them
     * back.
     *
     * @return const ov::CoordinateDiff&
     */
    virtual const ov::CoordinateDiff& get_pads_begin() = 0;
    virtual const ov::CoordinateDiff& get_pads_end() = 0;

    /**
     * @brief Some shape inference implementation may require input data stored inside the input tensors. To define
     * which inputs data are required, the port mask is used. Each set bit corresponds to the specific input port
     * number.
     *
     * @return port_mask_t a bit mask where each bit corresponds to an input port number.
     */
    virtual port_mask_t get_port_mask() const = 0;
};

/**
 * This is the base class for implementations that are not supposed to operate with padding. The corresponding methods
 * always return empty vectors.
 *
 */
class ShapeInferEmptyPads : public IShapeInfer {
public:
    const ov::CoordinateDiff& get_pads_begin() override final { // NOLINT
        return m_emptyVec;
    }
    const ov::CoordinateDiff& get_pads_end() override final { // NOLINT
        return m_emptyVec;
    }
private:
    static const ov::CoordinateDiff m_emptyVec;
};

using ShapeInferPtr = std::shared_ptr<IShapeInfer>;
using ShapeInferCPtr = std::shared_ptr<const IShapeInfer>;

constexpr IShapeInfer::port_mask_t EMPTY_PORT_MASK = 0x0;
constexpr IShapeInfer::port_mask_t FULL_PORT_MASK = 0xffffffff;

class ShapeInferFactory {
public:
    virtual ~ShapeInferFactory() = default;
    virtual ShapeInferPtr makeShapeInfer() const = 0;
};

/**
 * Shape inference factory creates shape inference objects that use ngraph shape inference implementations.
 *
 */
class NgraphShapeInferFactory final : public ShapeInferFactory {
public:
    /**
     * @brief Construct a new Ngraph Shape Infer Factory object
     *
     * @param op ngraph operation
     * @param port_mask port mask should be defined by the user. Will be stored in the shape infer object and returned
     * by the get_port_mask() call
     */
    NgraphShapeInferFactory(std::shared_ptr<ov::Node> op, IShapeInfer::port_mask_t port_mask) : m_op(op), m_port_mask(port_mask) {}
    ShapeInferPtr makeShapeInfer() const override;
private:
    std::shared_ptr<ov::Node> m_op;
    IShapeInfer::port_mask_t m_port_mask;
};
}   // namespace intel_cpu
}   // namespace ov
