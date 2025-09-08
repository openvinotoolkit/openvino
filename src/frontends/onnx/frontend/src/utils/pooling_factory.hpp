// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <type_traits>

#include "core/node.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace pooling {
///
/// \brief      Factory class which generates sub-graphs for ONNX 'regular' pooling
///             operators.
///
/// \note       This factory is intended for creating pooling operations like:
///             - AveragePool
///             - MaxPool
///
///             This class holds all common attributes like srides, dilations,
///             paddings, kernel shape and auto_pad type.
class PoolingFactory {
public:
    explicit PoolingFactory(const Node& node);
    virtual ~PoolingFactory() = default;

    ///
    /// \brief      Creates average pooling ONNX operation.
    /// \return     Vector of output nodes.
    ///
    ov::OutputVector make_avg_pool() const;

    ///
    /// \brief      Creates max pooling ONNX operation.
    /// \return     Vector of output nodes.
    ///
    ov::OutputVector make_max_pool() const;

    /// \brief Creates max pooling ONNX operation with 2 outputs (values and indices).
    ov::OutputVector make_max_pool_with_indices() const;

protected:
    Node m_onnx_node;

    const ov::OutputVector m_inputs;
    ov::Shape m_kernel_shape;
    ov::Strides m_strides;
    ov::Strides m_dilations;
    ov::Shape m_padding_below;
    ov::Shape m_padding_above;
    ov::op::PadType m_auto_pad;
    ov::op::RoundingType m_rounding_type;

    enum class StorageOrder : int64_t { ROW_MAJOR = 0, COLUMN_MAJOR = 1 };

    StorageOrder m_storage_order;
};
}  // namespace pooling
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
