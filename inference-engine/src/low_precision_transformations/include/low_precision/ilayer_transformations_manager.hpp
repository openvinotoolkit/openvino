// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/node.hpp>
#include "transformations_visibility.hpp"

namespace ngraph {
namespace pass {

/**
 * @brief low precision transformation component interface.
  */
class TRANSFORMATIONS_API ILayerTransformationsManager {
public:
    virtual bool isQuantized(const std::shared_ptr<Node>& layer) const noexcept = 0;
    virtual bool isPrecisionPreserved(const std::shared_ptr<Node>& layer) const noexcept = 0;
};

}  // namespace pass
}  // namespace ngraph
