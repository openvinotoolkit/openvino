// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "low_precision/cleanup_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief Base class for fuse elementwise to FakeQuantize low precision transformation.
 */
class LP_TRANSFORMATIONS_API FuseElementwiseToFakeQuantizeTransformation : public CleanupTransformation {
public:
    FuseElementwiseToFakeQuantizeTransformation(const Params& params);

    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
