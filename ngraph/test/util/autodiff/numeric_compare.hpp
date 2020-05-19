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

#include "ngraph/log.hpp"
#include "ngraph/type/element_type.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_derivative.hpp"
#include "util/autodiff/numeric_derivative.hpp"
#include "util/test_tools.hpp"

// TODO: Consider removing template since only <float> is being used in tests and numerical
//       derivative does not work with int types
// TODO: Always compute the numerical derivatives in double
template <typename T>
::testing::AssertionResult
    autodiff_numeric_compare(ngraph::runtime::Backend* backend,
                             std::shared_ptr<ngraph::Function> f,
                             std::shared_ptr<ngraph::Function> g,
                             const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& args,
                             T rtol,
                             T atol)
{
    T delta = static_cast<T>(0.0009765625f); // Binary-representable number near 0.001

    // Use INTERPRETER to compute numerical derivatives
    auto interpreter_backend = ngraph::runtime::Backend::create("INTERPRETER");

    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> interpreter_args;
    for (auto arg : args)
    {
        auto interpreter_arg =
            interpreter_backend->create_tensor(arg->get_element_type(), arg->get_shape());

        // TODO: copy_data should not require T. Quick fix here for bool used in `Select`
        if (arg->get_element_type() == ngraph::element::boolean)
        {
            copy_data(interpreter_arg, read_vector<char>(arg));
        }
        else
        {
            copy_data(interpreter_arg, read_vector<T>(arg));
        }
        interpreter_args.push_back(interpreter_arg);
    }
    auto results_num = ngraph::autodiff::numeric_derivative<T>(
        interpreter_backend.get(), f, interpreter_args, delta, f->get_parameters());

    // Use the backend being tested to compute symbolic derivatives
    auto results_sym =
        ngraph::autodiff::backprop_derivative<T>(backend, g, args, g->get_parameters());

    // Cast to HostTensor for comparision
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> interpreter_results_sym;
    for (auto result : results_sym)
    {
        auto interpreter_result =
            interpreter_backend->create_tensor(ngraph::element::from<T>(), result->get_shape());
        copy_data(interpreter_result, read_vector<T>(result));
        interpreter_results_sym.push_back(interpreter_result);
    }

    return ngraph::test::all_close(results_num, interpreter_results_sym, rtol, atol);
}

template <typename T>
::testing::AssertionResult
    autodiff_numeric_compare(ngraph::runtime::Backend* backend,
                             std::function<std::shared_ptr<ngraph::Function>()> make_graph,
                             const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& args,
                             T rtol,
                             T atol)
{
    return autodiff_numeric_compare(backend, make_graph(), make_graph(), args, rtol, atol);
}

template <typename T>
::testing::AssertionResult autodiff_numeric_compare_selective(
    ngraph::runtime::Backend* backend,
    std::shared_ptr<ngraph::Function> f,
    std::shared_ptr<ngraph::Function> g,
    const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& args,
    T rtol,
    T atol,
    const std::vector<bool>& indep_param_mask)
{
    // Use INTERPRETER to compute numerical derivatives
    std::vector<std::shared_ptr<ngraph::op::Parameter>> f_indep_params;

    size_t i = 0;

    for (auto b : indep_param_mask)
    {
        if (b)
        {
            f_indep_params.push_back(f->get_parameters().at(i));
        }
        i++;
    }

    auto interpreter_backend = ngraph::runtime::Backend::create("INTERPRETER");

    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> interpreter_args;
    for (auto arg : args)
    {
        auto interpreter_arg =
            interpreter_backend->create_tensor(arg->get_element_type(), arg->get_shape());

        // TODO: copy_data should not require T. Quick fix here for bool used in `Select`
        if (arg->get_element_type() == ngraph::element::boolean)
        {
            copy_data(interpreter_arg, read_vector<char>(arg));
        }
        else
        {
            copy_data(interpreter_arg, read_vector<T>(arg));
        }
        interpreter_args.push_back(interpreter_arg);
    }
    auto results_num = ngraph::autodiff::numeric_derivative<T>(
        interpreter_backend.get(), f, interpreter_args, .001f, f_indep_params);

    // Use the backend being tested to compute symbolic derivatives
    std::vector<std::shared_ptr<ngraph::op::Parameter>> g_indep_params;

    i = 0;

    for (auto b : indep_param_mask)
    {
        if (b)
        {
            g_indep_params.push_back(g->get_parameters().at(i));
        }
        i++;
    }

    auto results_sym = ngraph::autodiff::backprop_derivative<T>(backend, g, args, g_indep_params);

    // Cast to HostTensor for comparision
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> interpreter_results_sym;
    for (auto result : results_sym)
    {
        auto interpreter_result =
            interpreter_backend->create_tensor(ngraph::element::from<T>(), result->get_shape());
        copy_data(interpreter_result, read_vector<T>(result));
        interpreter_results_sym.push_back(interpreter_result);
    }

    return ngraph::test::all_close(results_num, interpreter_results_sym, rtol, atol);
}

template <typename T>
::testing::AssertionResult autodiff_numeric_compare_selective(
    ngraph::runtime::Backend* backend,
    std::function<std::shared_ptr<ngraph::Function>()> make_graph,
    const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& args,
    T rtol,
    T atol,
    const std::vector<bool>& indep_param_mask)
{
    return autodiff_numeric_compare_selective(
        backend, make_graph(), make_graph(), args, rtol, atol, indep_param_mask);
}
