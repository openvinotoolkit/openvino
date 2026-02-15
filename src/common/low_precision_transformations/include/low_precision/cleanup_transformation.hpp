// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief Base class for cleanup low precision transformation.
 */
class LP_TRANSFORMATIONS_API CleanupTransformation : public LayerTransformation {
public:
    CleanupTransformation(const Params& params);

    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
    static bool canBeTransformedStatic(
        const std::shared_ptr<Node>& layer,
        const std::vector<ov::element::Type>& defaultPrecisions = precision_set::get_int8_support());
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
