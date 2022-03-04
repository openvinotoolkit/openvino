// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/reduction_base.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Abstract base class for logical reduction operations, i.e., operations where
///        chosen axes of the input tensors are eliminated (reduced out) by repeated
///        application of a particular binary logical operation.
class OPENVINO_API LogicalReduction : public ReductionBase {
protected:
    /// \brief Constructs a logical reduction operation.
    LogicalReduction();
    /// \brief Constructs a logical reduction operation.
    ///
    /// \param arg Output that produces the first input tensor.
    /// \param reduction_axes The axis positions (0-based) to be eliminated.
    LogicalReduction(const Output<Node>& arg, const AxisSet& reduction_axes);
    /// \brief Constructs a 'dynamic' logical reduction operation.
    ///
    /// \param arg Node that produces the first input tensor.
    /// \param reduction_axes The axis positions (0-based) to be eliminated.
    LogicalReduction(const Output<Node>& arg, const Output<Node>& reduction_axes);

public:
    OPENVINO_OP("LogicalReduction", "util");
    BWDCMP_RTTI_DECLARATION;
    void validate_and_infer_types() override;

    /// \return true if reduction axes are constant else false.
    bool reduction_axes_constant() const;

    /// \return The axis positions (0-based) to be eliminated through reduction.
    /// \throws CheckFailure if the reduction axes are not constant. (Use
    ///           reduction_axes_constant to check.)
    const AxisSet get_reduction_axes() const;
    void set_reduction_axes(const AxisSet& reduction_axes);
};
}  // namespace util
}  // namespace op
}  // namespace ov
