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

#include "ngraph/builder/quantize_builder.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace builder
    {
        shared_ptr<Node> QuantizeBuilder(const Output<Node>& input,
                                         const Output<Node>& min,
                                         const Output<Node>& max,
                                         const ngraph::element::Type& quant_type,
                                         const ngraph::AxisSet& axes,
                                         op::Quantize::RoundMode round_mode)
        {
            auto real_type = input.get_element_type();

            if (min.get_element_type() != real_type)
            {
                throw ngraph_error("QuantizeBuilder: min must match input type");
            }

            if (max.get_element_type() != real_type)
            {
                throw ngraph_error("QuantizeBuilder: max must match input type");
            }

            auto shape = min.get_shape();
            if (shape != max.get_shape())
            {
                throw ngraph_error("QuantizeBuilder: min and max must have same shape");
            }

            auto zero = make_constant(quant_type, shape, 0);
            auto scale = quantization_utils::get_scale(min, max, quant_type, true);
            return make_shared<op::Quantize>(input, scale, zero, quant_type, axes, round_mode)
                ->add_provenance_group_members_above({input, min, max});
        }
    }
}
