/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

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