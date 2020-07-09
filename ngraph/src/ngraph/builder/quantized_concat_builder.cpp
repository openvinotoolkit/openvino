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

#include <memory>

#include "ngraph/builder/quantized_concat_builder.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace builder
    {
        shared_ptr<Node> QuantizedConcatBuilder(const OutputVector& args,
                                                size_t concatenation_axis,
                                                const OutputVector& mins,
                                                const OutputVector& maxs)
        {
            quantization_utils::check_concat(args, mins, maxs);
            auto quant_type = args[0].get_element_type();

            // output scale
            auto min = make_shared<op::Min>(make_shared<op::Concat>(mins, 0), ngraph::AxisSet{0});
            auto max = make_shared<op::Max>(make_shared<op::Concat>(maxs, 0), ngraph::AxisSet{0});
            auto out_scale = quantization_utils::get_scale(min, max, quant_type);

            OutputVector rescaled_args(args.size());
            for (size_t i = 0; i < args.size(); ++i)
            {
                auto q_type = args[i].get_element_type();
                auto in_scale = make_shared<ngraph::op::Reshape>(
                    quantization_utils::get_scale(mins[i], maxs[i], q_type),
                    AxisVector{0},
                    Shape{});
                auto zero = make_constant(q_type, in_scale->get_output_shape(0), 0);

                rescaled_args[i] =
                    make_shared<op::Dequantize>(args[i], in_scale, zero, element::f32, AxisSet{});
                rescaled_args[i] =
                    make_shared<op::Quantize>(rescaled_args[i],
                                              out_scale,
                                              zero,
                                              q_type,
                                              AxisSet{},
                                              op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN);
            }
            OutputVector base = args;
            for (auto node : mins)
            {
                base.push_back(node);
            };
            for (auto node : maxs)
            {
                base.push_back(node);
            };
            return make_shared<op::Concat>(rescaled_args, concatenation_axis)
                ->add_provenance_group_members_above(base);
        }
    }
}
