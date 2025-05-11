// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief TransparentBaseTransformation is base type for precision preserved operation transformation.
 */
class LP_TRANSFORMATIONS_API TransparentBaseTransformation : public LayerTransformation {
public:
    TransparentBaseTransformation(const Params& params) : LayerTransformation(params) {}
    ~TransparentBaseTransformation() override {};
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
