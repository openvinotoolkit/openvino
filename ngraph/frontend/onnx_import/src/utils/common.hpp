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

#pragma once

#include <algorithm>   // std::generate
#include <cmath>       // std::floor, std::min
#include <cstddef>     // std::size_t
#include <cstdint>     // std::int64_t
#include <iterator>    // std::begin, std::end
#include <memory>      // std::shared_ptr, std::make_shared
#include <type_traits> // std::enable_if
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "onnx_import/core/node.hpp"
#include "onnx_import/default_opset.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace common
        {
            const ngraph::element::Type& get_ngraph_element_type(std::int64_t onnx_type);

            /// \brief      Return a monotonic sequence.
            ///
            /// \note       Limitations: this function may not work for very large integer values
            ///             (near numerical limits).
            ///
            /// \param[in]  start_value  The start value of the sequence.
            /// \param[in]  end_value    The end value of the sequence.
            /// \param[in]  step         The step value for the sequence.
            ///
            /// \tparam     T            The data value type.
            ///
            /// \return     The vector with monotonic sequence
            template <typename T>
            std::vector<T> get_monotonic_range(T end_value, T start_value = T{0}, T step = T{1})
            {
                auto value_count =
                    static_cast<std::size_t>(std::floor((end_value - start_value) / step));

                std::vector<T> range(value_count);

                // Calculate initial value (one step below starting value)
                size_t n = start_value - step;
                // Generate a vector of values by adding step to previous value
                std::generate(
                    std::begin(range), std::end(range), [&n, &step]() -> T { return n += step; });

                return range;
            }

            /// \brief      Return a monotonic sequence which end is determined by value rank.
            ///
            /// \param[in]  value        The value node which rank determines end of the sequence.
            /// \param[in]  start_value  The start value of the sequence.
            /// \param[in]  step         The step value for the sequence.
            ///
            /// \return     The node which represents monotonic sequence.
            std::shared_ptr<ngraph::Node> get_monotonic_range_along_node_rank(
                const Output<ngraph::Node>& value, int64_t start_value = 0, int64_t step = 1);

            /// \brief Creates a shifted square identity matrix.
            /// \note Shifting in the context of this operator means that
            ///       the matrix can be created with elements equal to 1 not only in the main
            ///       diagonal. Shifting adds an offset and moves the diagonal up or down
            ///
            /// \param[in] output_shape Shape of the resulting matrix.
            /// \param[in] output_type Element type of the resulting matrix.
            /// \param[in] shift Shifting of diagonal.
            ///
            /// \return A Constant node representing shifted identity matrix.
            template <typename T = double>
            std::shared_ptr<default_opset::Constant>
                shifted_square_identity(const Shape output_shape,
                                        const element::Type& output_type,
                                        const std::int64_t shift)
            {
                std::vector<T> identity_matrix(shape_size(output_shape), T{0});
                std::int64_t rows = output_shape[0];
                std::int64_t cols = output_shape[1];
                for (std::int64_t row = 0; row < rows; ++row)
                {
                    const std::int64_t diagonal_element_idx = (row * cols) + row + shift;
                    if (row + shift < 0)
                    {
                        continue;
                    }
                    else if (row + shift >= cols)
                    {
                        break;
                    }
                    identity_matrix.at(diagonal_element_idx) = T{1};
                }

                return std::make_shared<default_opset::Constant>(
                    output_type, output_shape, identity_matrix);
            }

            /// \brief Creates a square identity matrix.
            ///
            /// \param[in] n Order of the resulting matrix.
            ///
            /// \return A Constant node representing identity matrix with shape (n, n).
            template <typename T = double>
            std::shared_ptr<default_opset::Constant> square_identity(const size_t n,
                                                                     const element::Type& type)
            {
                return shifted_square_identity(Shape{n, n}, type, 0);
            }

            /// \brief Performs validation of an input that is expected to be a scalar.
            /// \note  This function throws an exception if any of the validation steps fails.
            ///
            /// \param[in] input_name A human-readable name of an input (used for logging)
            /// \param[in] input An input node to be validated
            /// \param[in] allowed_types An optional set of allowed element types for this input
            void validate_scalar_input(const char* input_name,
                                       const std::shared_ptr<ngraph::Node> input,
                                       const std::set<element::Type> allowed_types = {});

        } // namespace  common
    }     // namespace onnx_import
} // namespace ngraph
