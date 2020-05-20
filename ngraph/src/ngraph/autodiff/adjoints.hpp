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

#include <map>
#include <memory>
#include <unordered_map>

#include "ngraph/coordinate.hpp"
#include "ngraph/output_vector.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    class Node;
    class Function;

    template <typename T>
    class Output;

    namespace autodiff
    {
        class NGRAPH_API Adjoints
        {
        public:
            /// \brief (dy/dx)(c) for all x used to compute y
            ///
            /// \param y The dependent value
            /// \param c An expression for where to evaluate the derivatives
            Adjoints(const OutputVector& y, const OutputVector& c);

            Adjoints(const Adjoints& adjoints) = default;
            Adjoints& operator=(const Adjoints& adjoints) = default;
            Adjoints() = default;

            /// \brief Add a backprop contribution to x's adjoint
            ///
            /// \param x The adjoint node
            /// \param delta A backprop contribution
            void add_delta(const Output<Node>& x, const Output<Node>& delta);

            /// \brief Add a backprop contribution to a slice of x's adjoint
            ///
            /// \param x The adjoint node
            /// \param delta A backprop contribution
            /// \param lower_bounds Lower bounds of slice to add to
            /// \param upper_bounds Upper bounds of slice to add to
            /// \param strides Strides of slice to add to
            void add_delta_to_slice(const Output<Node>& x,
                                    const Output<Node>& delta,
                                    const Coordinate& lower_bounds,
                                    const Coordinate& upper_bounds,
                                    const Strides& strides);

            /// \brief (dy/dx)(c)
            ///
            /// \param x The output whose adjoint is desired.
            Output<Node> backprop_output(const Output<Node>& x);

        protected:
            std::map<Node*, OutputVector> m_adjoint_map;
        };
    }
}
