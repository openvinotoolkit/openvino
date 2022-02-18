// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/assign.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/read_value.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/sink.hpp"
#include "ngraph/op/util/variable.hpp"
#include "openvino/core/model.hpp"

namespace ngraph {
using Function = ov::Model;
}  // namespace ngraph
