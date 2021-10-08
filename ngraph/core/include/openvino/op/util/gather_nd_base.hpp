// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
/// \brief GatherNDBase basic class for GatherND v5 and v8
class OPENVINO_API GatherNDBase : public Op {
public:
    OPENVINO_OP("GatherNDBase", "util");
    BWDCMP_RTTI_DECLARATION;
    GatherNDBase() = default;

    /// \brief Constructs a GatherND operation.
    ///
    /// \param data Node producing data that are gathered
    /// \param indices Node producing indices by which the operation gathers elements
    /// or slices from data
    /// \param batch_dims Specifies a number of batch dimensions
    GatherNDBase(const Output<Node>& data,
                 const Output<Node>& indices,
                 const size_t batch_dims = 0);

    void validate_and_infer_types() override;

    size_t get_batch_dims() const {
        return m_batch_dims;
    }

protected:
    size_t m_batch_dims = 0;
};
}  // namespace util
}  // namespace op
}  // namespace ov
