// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <ie_api.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class INFERENCE_ENGINE_API_CLASS(TopKIE) : public Op {
public:
    static constexpr NodeTypeInfo type_info{"TopKIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    TopKIE(const Output<Node> &data,
           const Output<Node> &k,
           const int64_t axis,
           const std::string& mode,
           const std::string& sort,
           const Shape& output_shape);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;
    int64_t get_axis();
    std::string get_mode();
    std::string get_sort_type();
    Shape get_output_shape();

    int64_t axis;
    std::string mode, sort_type;
    Shape output_shape;
};

}  // namespace op
}  // namespace ngraph
