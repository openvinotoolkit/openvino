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

#include <limits>
#include <vector>
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace builder
    {
        namespace quantization_utils
        {
            std::shared_ptr<Node> max_abs(const Output<Node>& a, const Output<Node>& b);

            std::shared_ptr<Node> get_scale(const Output<Node>& input_min_range,
                                            const Output<Node>& input_max_range,
                                            const ngraph::element::Type& quant_type,
                                            bool bump_by_eps = false);

            std::shared_ptr<Node> get_bias_scale(Output<Node> min_input,
                                                 Output<Node> max_input,
                                                 Output<Node> min_filter,
                                                 Output<Node> max_filter);

            std::shared_ptr<Node> get_sum_scale(Output<Node> min_freezed_output_conv_1,
                                                Output<Node> max_freezed_output_conv_1,
                                                Output<Node> min_freezed_output_conv_2,
                                                Output<Node> max_freezed_output_conv_2);

            std::shared_ptr<Node> get_dot_scale(Output<Node> min_input,
                                                Output<Node> max_input,
                                                Output<Node> min_filter,
                                                Output<Node> max_filter,
                                                Output<Node> min_freezed_output,
                                                Output<Node> max_freezed_output,
                                                const ngraph::element::Type& input_type,
                                                const ngraph::element::Type& output_type,
                                                const bool requantize = true);

            void check_concat(const OutputVector& args,
                              const OutputVector& mins,
                              const OutputVector& maxs);
        }
    }
}
