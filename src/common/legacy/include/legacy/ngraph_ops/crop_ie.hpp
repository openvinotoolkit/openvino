// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class CropIE : public Op {
public:
    OPENVINO_OP("CropIE", "legacy");
    BWDCMP_RTTI_DECLARATION;

    CropIE(const Output<Node>& data1,
           std::vector<int64_t> axes,
           std::vector<int64_t> dim,
           std::vector<int64_t> offset);

    bool visit_attributes(AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    std::vector<int64_t> axes, dim, offset;
};

}  // namespace op
}  // namespace ngraph
