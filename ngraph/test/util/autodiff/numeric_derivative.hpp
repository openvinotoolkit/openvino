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

#include <memory>
#include <vector>

#include "ngraph/type/element_type.hpp"
#include "runtime/backend.hpp"

namespace ngraph
{
    namespace autodiff
    {
        /// \brief numeric approximation of the derivative
        /// \param f A function
        /// \param args Values for the arguments (the independent variables)
        /// \param delta increment for the variables
        /// \param indep_params parameters with respect to which to compute derivatives
        /// \returns vector of dy/dvar, where each dy/dvar's shape is concat(y.shape(), var.shape())
        template <typename T>
        std::vector<std::shared_ptr<runtime::Tensor>>
            numeric_derivative(runtime::Backend* backend,
                               const std::shared_ptr<Function>& f,
                               const std::vector<std::shared_ptr<runtime::Tensor>>& args,
                               T delta,
                               const std::vector<std::shared_ptr<op::Parameter>>& indep_params)
        {
            Shape y_shape = f->get_output_shape(0);

            auto params = f->get_parameters();

            // Results for each derivative, shape Y|X_i
            std::vector<std::shared_ptr<runtime::Tensor>> results;

            for (auto param : indep_params)
            {
                Shape s = y_shape;
                auto param_shape = param->get_shape();
                s.insert(s.end(), param_shape.begin(), param_shape.end());
                results.push_back(backend->create_tensor<T>(s));
            }

            // ref_y is the function evaluated at the args
            auto ref_y = backend->create_tensor<T>(y_shape);

            auto f_handle = backend->compile(f);

            f_handle->call_with_validate(
                std::vector<std::shared_ptr<ngraph::runtime::Tensor>>{ref_y}, args);
            auto ref_vec = read_vector<T>(ref_y);

            // inc_y will hold f(x+dx) values
            auto inc_y = backend->create_tensor<T>(y_shape);

            // Assuming vars, y, and results are row-major

            T inv_delta = 1 / delta;

            size_t pos = 0;

            for (size_t i = 0; i < args.size(); ++i)
            {
                if (std::find(indep_params.begin(), indep_params.end(), params[i]) !=
                    indep_params.end())
                {
                    auto arg = args[i];
                    auto res = read_vector<T>(results[pos]);
                    auto vec = read_vector<T>(arg);
                    for (size_t j = 0; j < vec.size(); j++)
                    {
                        auto old_val = vec[j];
                        vec[j] += delta;
                        write_vector(arg, vec);
                        f_handle->call_with_validate({inc_y}, args);
                        auto inc_vec = read_vector<T>(inc_y);
                        vec[j] = old_val;
                        write_vector(arg, vec);
                        size_t res_k = j;
                        for (size_t k = 0; k < inc_vec.size(); k++)
                        {
                            auto y1 = inc_vec[k];
                            auto y0 = ref_vec[k];
                            res[res_k] = inv_delta * (y1 - y0);
                            res_k += vec.size();
                        }
                    }
                    write_vector(results[pos], res);
                    pos++;
                }
            }
            return results;
        }
    }
}
