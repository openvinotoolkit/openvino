// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(TileIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"TileIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    TileIE(const Output<Node>& data1,
            const int64_t axis,
            const int64_t tiles);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t axis, tiles;
};

}  // namespace op
}  // namespace ngraph
