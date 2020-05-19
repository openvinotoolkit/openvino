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
#include <unordered_map>

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/test_tools.hpp"

namespace ngraph
{
    class Node;
    class Function;

    static std::unordered_map<std::shared_ptr<Function>, std::shared_ptr<Function>> s_df_map;
    static std::unordered_map<std::shared_ptr<Function>, std::shared_ptr<Function>> s_clone_bwd_map;

    namespace runtime
    {
        class Backend;
        class Manager;
    }

    namespace autodiff
    {
        template <typename T>
        std::vector<std::shared_ptr<runtime::Tensor>>
            get_autodiff(runtime::Backend* backend,
                         std::shared_ptr<Function>& df,
                         const std::vector<std::shared_ptr<runtime::Tensor>>& df_input_args,
                         const std::vector<std::shared_ptr<op::Parameter>>& indep_params)
        {
            // df/dX* = f'(c, ...)
            // using X* to denote all x "of interest" (represented by indep_params)

            // return value for this function
            std::vector<std::shared_ptr<runtime::Tensor>> results;

            // adjoint
            auto c_arg = df_input_args[0];
            auto y_shape = c_arg->get_shape();

            // df/dX* arguments
            std::vector<std::shared_ptr<runtime::Tensor>> df_output_args;

            // for each x "of interest"
            for (auto x : indep_params)
            {
                // add df/dx to df/dX* arguments
                auto x_shape = x->get_shape();
                df_output_args.push_back(backend->create_tensor<T>(x_shape));

                // each element of y has a derivative with respect to each element of x
                // hence, create a y by x sized tensor for this result
                auto y_by_x_shape = y_shape;
                y_by_x_shape.insert(y_by_x_shape.end(), x_shape.begin(), x_shape.end());
                results.push_back(backend->create_tensor<T>(y_by_x_shape));
            }

            // create storage for results
            std::vector<std::vector<T>> result_vect;
            std::vector<typename std::vector<T>::iterator> result_pos;
            for (auto result : results)
            {
                result_vect.push_back(read_vector<T>(result));
                result_pos.push_back(result_vect.back().begin());
            }

            // get adjoint and force to all elements to zero
            auto c_vec = read_vector<T>(c_arg);
            fill(c_vec.begin(), c_vec.end(), static_cast<T>(0));

            auto df_handle = backend->compile(clone_function(*df));

            // for each element of the adjoint
            // same as saying for each element of y
            for (size_t i = 0; i < c_vec.size(); i++)
            {
                // set a single adjoint element
                c_vec[i] = 1;
                write_vector(c_arg, c_vec);

                // call modified df/dX* = f'(c, cached)
                df_handle->call_with_validate(df_output_args, df_input_args);

                // reset the adjoint element
                c_vec[i] = 0;

                // for each result
                // same as saying for each x "of interest"
                for (size_t j = 0; j < results.size(); j++)
                {
                    // copy df/dx to storage for this element of y
                    auto dfdx = read_vector<T>(df_output_args[j]);
                    result_pos[j] = std::copy(dfdx.begin(), dfdx.end(), result_pos[j]);
                }
            }

            // copy storage to results and return
            for (size_t j = 0; j < results.size(); j++)
            {
                write_vector(results[j], result_vect[j]);
            }
            return results;
        }

        template <typename T>
        std::vector<std::shared_ptr<runtime::Tensor>>
            backprop_derivative(runtime::Backend* backend,
                                const std::shared_ptr<Function>& f,
                                const std::vector<std::shared_ptr<runtime::Tensor>>& f_input_args,
                                const std::vector<std::shared_ptr<op::Parameter>>& indep_params)
        {
            // y = f(X)
            // using X (upper case) to denote all paramenters of f (represented by f_input_args)
            // using x (lower case) to denote an individual paramemter of f
            // using X* to denote all x "of interest" (represented by indep_params)
            Shape y_shape = f->get_output_shape(0);

            // adjoint
            auto c_param = std::make_shared<op::Parameter>(element::from<T>(), y_shape);
            auto c_arg = backend->create_tensor<T>(y_shape);

            // df/dX*
            std::vector<Output<Node>> df_output_params;

            Adjoints adjoints(OutputVector{f->output(0)}, OutputVector{c_param});

            // for each x "of interest"
            for (auto x : indep_params)
            {
                // add df/dx to df/dX*
                df_output_params.push_back(adjoints.backprop_output(x));
            }

            // (c, X)
            std::vector<std::shared_ptr<op::Parameter>> df_input_params = f->get_parameters();
            df_input_params.insert(df_input_params.begin(), c_param);

            // df/dX* = f'(c, X)
            if (!s_df_map[f])
            {
                s_df_map[f] = std::make_shared<Function>(df_output_params, df_input_params);
            }
            auto df = s_df_map[f];

            // (c, X) arguments
            std::vector<std::shared_ptr<runtime::Tensor>> df_input_args = f_input_args;
            df_input_args.insert(df_input_args.begin(), c_arg);

            // call f'(c,X) to get df/dX*
            auto dfdx = get_autodiff<T>(backend, df, df_input_args, indep_params);

            // create fprop cache
            // creates modified forward function -> (y, cached) = f(x)
            // creates modified backward function -> df/dX* = f'(c, cached)
            auto fprop_cache = cache_fprop(f, df);

            // (y, cached) arguments
            std::vector<std::shared_ptr<runtime::Tensor>> mod_f_output_args;
            mod_f_output_args.push_back(backend->create_tensor<T>(y_shape));

            // (c, cached) arguments
            std::vector<std::shared_ptr<runtime::Tensor>> mod_df_input_args = df_input_args;

            // add cached nodes to both modified f output and modified f' input arguments
            for (auto weak_value : fprop_cache.fprop_output_nodes)
            {
                Output<Node> value(weak_value);
                auto tv = backend->create_tensor(value.get_element_type(), value.get_shape());
                mod_f_output_args.push_back(tv);
                mod_df_input_args.push_back(tv);
            }

            // compile and run modified (y, cached) = f(x)
            static std::unordered_map<std::shared_ptr<Function>, std::shared_ptr<Function>>
                s_clone_fwd_map;
            if (!s_clone_fwd_map[f])
            {
                s_clone_fwd_map[f] = clone_function(*fprop_cache.fprop);
            }
            auto clone_fwd = s_clone_fwd_map[f];
            auto clone_fwd_handle = backend->compile(clone_fwd);

            clone_fwd_handle->call_with_validate(mod_f_output_args, f_input_args);

            // call modfied f'(c, cached) to get df/dX*
            if (!s_clone_bwd_map[f])
            {
                s_clone_bwd_map[f] = clone_function(*fprop_cache.bprop);
            }
            auto clone_bwd = s_clone_bwd_map[f];
            auto cache_dfdx = get_autodiff<T>(backend, clone_bwd, mod_df_input_args, indep_params);

            const T numpy_atol = static_cast<const T>(1e-5f);
            const T numpy_rtol = static_cast<const T>(1e-8f);
            auto close = ngraph::test::all_close<T>(dfdx, cache_dfdx, numpy_atol, numpy_rtol);
            if (!close)
            {
                throw ngraph_error(
                    "Derivatives mismatch between cache and non-cache bprop functions");
            }

            return dfdx;
        }
    }
}
