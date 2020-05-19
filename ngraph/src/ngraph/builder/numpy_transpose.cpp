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

#include <sstream>

#include "ngraph/axis_vector.hpp"
#include "ngraph/builder/numpy_transpose.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    [[noreturn]] void numpy_transpose_error(const AxisVector& order, const Shape& in_shape)
    {
        std::ostringstream os;
        os << "The axes order ";
        os << "[ " << ngraph::join(order) << " ]";
        os << " is incompatible with the input shape ";
        os << "[ " << ngraph::join(in_shape) << " ]";
        os << " during numpy_transpose.";
        throw ngraph_error(os.str());
    }

    namespace builder
    {
        std::shared_ptr<Node> numpy_transpose(const Output<Node>& value, AxisVector order)
        {
            auto in_shape = value.get_shape();
            // default, reverse the order of the axes
            if (order.size() == 0)
            {
                auto n = in_shape.size();
                order = AxisVector(n);
                std::generate(order.begin(), order.end(), [&n]() { return --n; });
            }
            else if (order.size() == in_shape.size())
            {
                // validate that the axes order is valid, i.e., unique and the right size
                std::unordered_set<ngraph::AxisVector::value_type> axes;
                for (auto o : order)
                {
                    if (o < in_shape.size() && !axes.count(o))
                    {
                        axes.insert(o);
                    }
                    else
                    {
                        numpy_transpose_error(order, in_shape);
                    }
                }
            }
            else
            {
                numpy_transpose_error(order, in_shape);
            }

            // create output shape
            Shape out_shape;
            for (size_t i = 0; i < in_shape.size(); ++i)
                out_shape.push_back(in_shape[order[i]]);

            // do the reshaping with the order
            return std::make_shared<ngraph::op::Reshape>(value, order, out_shape)
                ->add_provenance_group_members_above({value});
        }

    } // namespace builder
} // namespace ngraph
