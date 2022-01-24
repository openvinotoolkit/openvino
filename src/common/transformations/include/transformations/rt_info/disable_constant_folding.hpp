// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <set>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <transformations_visibility.hpp>
#include "openvino/pass/constant_folding.hpp"

namespace ov {

using pass::disable_constant_folding;
using pass::enable_constant_folding;
using pass::constant_folding_is_disabled;
using pass::DisableConstantFolding;

}  // namespace ov
