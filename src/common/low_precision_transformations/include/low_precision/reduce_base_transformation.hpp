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
 * @brief ReduceBaseTransformation: base class for Reduce*Transformation,
 * detects dequantization operations in front of the Reduce* operation and
 * propagates them through the Reduce* if possible.
 */

class LP_TRANSFORMATIONS_API ReduceBaseTransformation : public LayerTransformation {
public:
    ReduceBaseTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher& m) override;
    bool canBeTransformed(const std::shared_ptr<Node>& reduce) const override;

protected:
    virtual void changeDequantizationValues(
        const std::shared_ptr<Node>& reduce,
        FakeQuantizeDequantization& dequantization) const;
    virtual bool getUpdatePrecision(const std::shared_ptr<Node>& reduce) const;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
