// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ngraph/node.hpp>

namespace ngraph {
namespace pass {

/**
 * @brief low precision transformation component interface.
  */
class TRANSFORMATIONS_API ILayerTransformationsManager {
public:
    virtual bool isQuantized(std::shared_ptr<Node> layer) const noexcept = 0;
    virtual bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept = 0;
};

}  // namespace pass
}  // namespace ngraph
