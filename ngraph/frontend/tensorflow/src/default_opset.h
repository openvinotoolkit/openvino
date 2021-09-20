// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NGRAPH_TF_BRIDGE_DEFAULT_OPSET_H_
#define NGRAPH_TF_BRIDGE_DEFAULT_OPSET_H_
#pragma once

#include "ngraph/opsets/opset5.hpp"

namespace tensorflow {
namespace ngraph_bridge {

namespace opset = ngraph::opset5;
namespace default_opset = ngraph::opset5;

#define NGRAPH_TF_FE_NOT_IMPLEMENTED { std::cerr << "[ NOT IMPLEMENTED ] source: " << __FILE__ << ":" << __LINE__ << "\n"; throw "NOT IMPLEMENTED"; }

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif