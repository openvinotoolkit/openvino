// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/shared_value_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <iterator>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;
