//*****************************************************************************
// Copyright 2017-2022 Intel Corporation
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

#include <cstddef>
#include <vector>

#include "openvino/op/op.hpp"
#include "openvino/op/util/fft_base.hpp"

namespace ov {
namespace op {
namespace v9 {
/// \brief An operation IRDFT that computes the discrete inverse complex-to-real Fourier transformation.
class OPENVINO_API IRDFT : public util::FFTBase {
public:
    OPENVINO_OP("IRDFT", "opset9", util::FFTBase, 9);
    BWDCMP_RTTI_DECLARATION;
    IRDFT() = default;

    /// \brief Constructs a IRDFT operation. IRDFT is performed for full size axes.
    ///
    /// \param data  Input data
    /// \param axes Axes to perform IRDFT
    IRDFT(const Output<Node>& data, const Output<Node>& axes);

    /// \brief Constructs a IRDFT operation.
    ///
    /// \param data  Input data
    /// \param axes Axes to perform IRDFT
    /// \param signal_size Signal sizes for 'axes'
    IRDFT(const Output<Node>& data, const Output<Node>& axes, const Output<Node>& signal_size);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v7
}  // namespace op
}  // namespace ov
