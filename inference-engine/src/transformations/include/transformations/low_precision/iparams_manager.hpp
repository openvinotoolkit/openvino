// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <transformations_visibility.hpp>


namespace ngraph {
namespace pass {

/**
 * @brief low precision transformation component interface.
  */
class TRANSFORMATIONS_API IParamsManager {
public:
    // TODO FIXME: it is not correct to have a string as a key here, try to use NodeTypeInfo
    virtual std::vector<element::Type> getPrecisionsOnActivations(const std::string& layerType) const noexcept = 0;
};

}  // namespace pass
}  // namespace ngraph
