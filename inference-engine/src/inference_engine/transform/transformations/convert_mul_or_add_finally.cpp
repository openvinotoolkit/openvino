// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include <ngraph_ops/scaleshift.hpp>
#include <ngraph_ops/power.hpp>

#include "convert_mul_or_add_finally.hpp"
#include "convert_mul_add_to_scaleshift_or_power.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/pattern/matcher.hpp"

#include "ngraph/graph_util.hpp"
