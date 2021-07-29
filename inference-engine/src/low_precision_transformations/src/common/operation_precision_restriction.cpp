// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/common/operation_precision_restriction.hpp"

#include <memory>
#include <unordered_set>
#include <set>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"

using namespace ngraph;

