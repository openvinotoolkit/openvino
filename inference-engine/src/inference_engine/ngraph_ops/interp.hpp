// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <memory>
#include <string>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {
typedef struct {
    int height = -1;
    int width = -1;
    float zoom_factor = 0;
    float shrink_factor = 0;
    float scale_factor = 1.0;
    bool align_corners = true;
    bool antialias = true;
    std::string mode = "";
    int pad_beg = 0;
    int pad_end = 0;
} InterpolateIEAttrs;

class Interp : public Op {
public:
    Interp(const std::shared_ptr<Node>& image, const InterpolateIEAttrs& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    InterpolateIEAttrs get_attrs() { return m_attrs; }

private:
    InterpolateIEAttrs m_attrs;
};

class ResampleV2 : public Op {
public:
    ResampleV2(const std::shared_ptr<Node>& image,
               const std::shared_ptr<Node>& output_shape,
               const InterpolateIEAttrs& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

private:
    InterpolateIEAttrs m_attrs;
};
}  // namespace op
}  // namespace ngraph