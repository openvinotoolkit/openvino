// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
class OPENVINO_API ReductionBase : public Op {
protected:
    /// \brief Constructs a reduction operation.
    ReductionBase();

    /// \brief Constructs a reduction operation.
    ///
    /// \param arg Output that produces the first input tensor.
    /// \param reduction_axes The axis positions (0-based) to be eliminated.
    ReductionBase(const Output<Node>& arg, const Output<Node>& reduction_axes);

    /// \brief      Infers reduction operations output shape.
    ///
    /// \param[in] keep_dims    Reduction operation keeps dimensions.
    ///
    /// \return Partial shape of the output.
    PartialShape infer_reduction_output_shape(const bool keep_dims);

public:
    OPENVINO_OP("ReductionBase", "util");
    BWDCMP_RTTI_DECLARATION;
};
}  // namespace util
}  // namespace op
}  // namespace ov
