// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API AssignAndReadValueTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("AssignAndReadValueTransformation", "0");
    AssignAndReadValueTransformation(const std::shared_ptr<ov::Model> model, const Params& params = Params());
    bool transform(TransformationContext& context, ov::pass::pattern::Matcher& m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
private:
    std::shared_ptr<ov::Model> model;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
