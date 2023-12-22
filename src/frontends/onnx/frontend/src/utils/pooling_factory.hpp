// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <type_traits>

#include "ngraph/node.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "onnx_import/core/node.hpp"
#include "openvino/core/deprecated.hpp"

namespace ngraph {
namespace onnx_import {
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
    OPENVINO_SUPPRESS_DEPRECATED_START
    explicit PoolingFactory(const Node& node);
    virtual ~PoolingFactory() = default;
    OPENVINO_SUPPRESS_DEPRECATED_END

    ///
    /// \brief      Creates average pooling ONNX operation.
    /// \return     Vector of output nodes.
    ///
    OutputVector make_avg_pool() const;

    ///
    /// \brief      Creates max pooling ONNX operation.
    /// \return     Vector of output nodes.
    ///
    OutputVector make_max_pool() const;

    /// \brief Creates max pooling ONNX operation with 2 outputs (values and indices).
    OutputVector make_max_pool_with_indices() const;

protected:
    OPENVINO_SUPPRESS_DEPRECATED_START
    Node m_onnx_node;
    OPENVINO_SUPPRESS_DEPRECATED_END
    const OutputVector m_inputs;
    Shape m_kernel_shape;
    Strides m_strides;
    Strides m_dilations;
    Shape m_padding_below;
    Shape m_padding_above;
    ngraph::op::PadType m_auto_pad;
    ngraph::op::RoundingType m_rounding_type;

    enum class StorageOrder : int64_t { ROW_MAJOR = 0, COLUMN_MAJOR = 1 };

    StorageOrder m_storage_order;
};
}  // namespace pooling
}  // namespace onnx_import
}  // namespace ngraph
