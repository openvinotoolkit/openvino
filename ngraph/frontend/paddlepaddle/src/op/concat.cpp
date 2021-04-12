//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <ngraph/opsets/opset6.hpp>
#include "concat.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

OutputVector concat (const NodeContext& node) {
    auto data = node.get_ng_inputs("X");
    auto axis = node.get_attribute<int>("axis");
    return {std::make_shared<ngraph::opset6::Concat>(data, axis)};
}

}}}}