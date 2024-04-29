// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/multiclass_nms.hpp"
#include "openvino/op/util/multiclass_nms_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

class TRANSFORMATIONS_API MulticlassNmsIEInternal : public ov::op::v9::MulticlassNms {
public:
    OPENVINO_OP("MulticlassNmsIEInternal", "ie_internal_opset", ov::op::v9::MulticlassNms);

    MulticlassNmsIEInternal() = default;

    MulticlassNmsIEInternal(const Output<Node>& boxes,
                            const Output<Node>& scores,
                            const op::util::MulticlassNmsBase::Attributes& attrs);

    MulticlassNmsIEInternal(const Output<Node>& boxes,
                            const Output<Node>& scores,
                            const Output<Node>& roisnum,
                            const op::util::MulticlassNmsBase::Attributes& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace internal
}  // namespace op
}  // namespace ov
