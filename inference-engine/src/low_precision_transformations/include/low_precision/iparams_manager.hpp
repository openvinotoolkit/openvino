// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <ngraph/ngraph.hpp>
#include "low_precision/lpt_visibility.hpp"

namespace ngraph {
namespace pass {

/**
 * @brief low precision transformation component interface.
  */
class LP_TRANSFORMATIONS_API IParamsManager {
public:
    // TODO FIXME: it is not correct to have a string as a key here, try to use NodeTypeInfo
    virtual std::vector<element::Type> getPrecisionsOnActivations(const Node& op) const noexcept = 0;
};

}  // namespace pass
}  // namespace ngraph
