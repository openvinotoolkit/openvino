//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#pragma once

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/reshape.hpp"
#include "quantization_utils.hpp"

namespace ngraph
{
    namespace builder
    {
        NGRAPH_API
        std::shared_ptr<Node> QuantizedConcatBuilder(const NodeVector& args,
                                                     size_t concatenation_axis,
                                                     const NodeVector& mins,
                                                     const NodeVector& maxs);
    }
}
