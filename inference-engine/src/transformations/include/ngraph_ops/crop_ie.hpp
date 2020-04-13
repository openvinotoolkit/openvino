// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(CropIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"CropIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    CropIE(const Output<Node>& data1,
           std::vector<int64_t> axes,
           std::vector<int64_t> dim,
           std::vector<int64_t> offset);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    std::vector<int64_t> axes, dim, offset;
};

}  // namespace op
}  // namespace ngraph
