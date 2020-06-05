//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>

#include "ngraph/op/op.hpp"
#include <transformations_visibility.hpp>

namespace ngraph {
namespace op {
/// \brief Elementwise addition operation.
///
class TRANSFORMATIONS_API MultiplyAdd : public Op {
public:
    RTTI_DECLARATION;
    /// \brief Constructs an uninitialized addition operation
    MultiplyAdd() {
    }

    /// \brief Constructs an addition operation.
    ///
    /// \param data The main input tensor that will be scaled and shifted.<br            >
    /// `[d0, ...]`
    /// \param scale Scale factor; broadcastable to data
    /// `[d0, ...]`
    /// \param shift Shift term; broadcastable to data
    /// `[d0, ...]`
    /// \param auto_broadcast Auto broadcast specification
    ///
    /// Output `[d0, ...]`
    ///
    MultiplyAdd(const Output<Node>& data,
        const Output<Node>& scale,
        const Output<Node>& shift);
    // TODO: add autob support

    void validate_and_infer_types() override;

    std::shared_ptr<Node>
        clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;
};

} // namespace op
} // namespace ngraph
