// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/op/util/reduction_base.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief Abstract base class for arithmetic reduction operations, i.e., operations
///        where chosen axes of the input tensors are eliminated (reduced out) by
///        repeated application of a particular binary arithmetic operation.
class OPENVINO_API ArithmeticReduction : public ReductionBase {
protected:
    /// \brief Constructs an arithmetic reduction operation.
    ArithmeticReduction();

    /// \brief Constructs an arithmetic reduction operation.
    ///
    /// \param arg Output that produces the first input tensor.
    /// \param reduction_axes The axis positions (0-based) to be eliminated.
    ArithmeticReduction(const Output<Node>& arg, const Output<Node>& reduction_axes);

public:
    OPENVINO_OP("ArithmeticReduction", "util");
    BWDCMP_RTTI_DECLARATION;
    void validate_and_infer_types() override;

    /// \return true if reduction axes are constant else false.
    bool reduction_axes_constant() const;

    /// \return The axis positions (0-based) to be eliminated through reduction.
    /// \throws CheckFailure if the reduction axes are not constant. (Use
    ///           reduction_axes_constant to check.)
    const AxisSet get_reduction_axes() const;

    /// \brief Change the reduction axes
    void set_reduction_axes(const AxisSet& reduction_axes);
};
}  // namespace util
}  // namespace op
}  // namespace ov
