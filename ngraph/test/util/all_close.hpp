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

#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/pass/manager.hpp"
#include "ngraph/type/element_type.hpp"
#include "random.hpp"
#include "test_tools.hpp"

namespace ngraph
{
    namespace test
    {
        /// \brief Same as numpy.allclose
        /// \param a First tensor to compare
        /// \param b Second tensor to compare
        /// \param rtol Relative tolerance
        /// \param atol Absolute tolerance
        /// \returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename T>
        typename std::enable_if<std::is_floating_point<T>::value, ::testing::AssertionResult>::type
            all_close(const std::vector<T>& a,
                      const std::vector<T>& b,
                      T rtol = static_cast<T>(1e-5),
                      T atol = static_cast<T>(1e-8))
        {
            bool rc = true;
            ::testing::AssertionResult ar_fail = ::testing::AssertionFailure();
            if (a.size() != b.size())
            {
                throw std::invalid_argument("all_close: Argument vectors' sizes do not match");
            }
            size_t count = 0;
            for (size_t i = 0; i < a.size(); ++i)
            {
                if (std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i]) || !std::isfinite(a[i]) ||
                    !std::isfinite(b[i]))
                {
                    if (count < 5)
                    {
                        ar_fail << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                                << a[i] << " is not close to " << b[i] << " at index " << i
                                << std::endl;
                    }
                    count++;
                    rc = false;
                }
            }
            ar_fail << "diff count: " << count << " out of " << a.size() << std::endl;
            return rc ? ::testing::AssertionSuccess() : ar_fail;
        }

        /// \brief Same as numpy.allclose
        /// \param a First tensor to compare
        /// \param b Second tensor to compare
        /// \param rtol Relative tolerance
        /// \param atol Absolute tolerance
        /// \returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename T>
        typename std::enable_if<std::is_integral<T>::value, ::testing::AssertionResult>::type
            all_close(const std::vector<T>& a,
                      const std::vector<T>& b,
                      T rtol = static_cast<T>(1e-5),
                      T atol = static_cast<T>(1e-8))
        {
            bool rc = true;
            ::testing::AssertionResult ar_fail = ::testing::AssertionFailure();
            if (a.size() != b.size())
            {
                throw std::invalid_argument("all_close: Argument vectors' sizes do not match");
            }
            for (size_t i = 0; i < a.size(); ++i)
            {
                T abs_diff = (a[i] > b[i]) ? (a[i] - b[i]) : (b[i] - a[i]);
                if (abs_diff > atol + rtol * b[i])
                {
                    // use unary + operator to force integral values to be displayed as numbers
                    ar_fail << +a[i] << " is not close to " << +b[i] << " at index " << i
                            << std::endl;
                    rc = false;
                }
            }
            return rc ? ::testing::AssertionSuccess() : ar_fail;
        }

        /// \brief Same as numpy.allclose
        /// \param a First tensor to compare
        /// \param b Second tensor to compare
        /// \param rtol Relative tolerance
        /// \param atol Absolute tolerance
        /// Returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename T>
        ::testing::AssertionResult all_close(const std::shared_ptr<ngraph::runtime::Tensor>& a,
                                             const std::shared_ptr<ngraph::runtime::Tensor>& b,
                                             T rtol = 1e-5f,
                                             T atol = 1e-8f)
        {
            if (a->get_shape() != b->get_shape())
            {
                return ::testing::AssertionFailure()
                       << "Cannot compare tensors with different shapes";
            }

            return all_close(read_vector<T>(a), read_vector<T>(b), rtol, atol);
        }

        /// \brief Same as numpy.allclose
        /// \param as First tensors to compare
        /// \param bs Second tensors to compare
        /// \param rtol Relative tolerance
        /// \param atol Absolute tolerance
        /// Returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
        template <typename T>
        ::testing::AssertionResult
            all_close(const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& as,
                      const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& bs,
                      T rtol,
                      T atol)
        {
            if (as.size() != bs.size())
            {
                return ::testing::AssertionFailure()
                       << "Cannot compare tensors with different sizes";
            }
            for (size_t i = 0; i < as.size(); ++i)
            {
                auto ar = all_close(as[i], bs[i], rtol, atol);
                if (!ar)
                {
                    return ar;
                }
            }
            return ::testing::AssertionSuccess();
        }
    } // namespace test
} // namespace ngraph

// apply pass, execute and compare with INTERPRETER using random data
template <typename T, typename TIN, typename TOUT = TIN>
bool compare_pass_int(std::shared_ptr<ngraph::Function>& baseline_f,
                      std::shared_ptr<ngraph::Function>& optimized_f,
                      std::vector<std::vector<TIN>> args = std::vector<std::vector<TIN>>{})
{
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::Validate>();
    pass_manager.register_pass<T>();
    pass_manager.run_passes(optimized_f);

    if (args.size() == 0)
    {
        for (auto& p : baseline_f->get_parameters())
        {
            args.emplace_back(shape_size(p->get_shape()), 0);
            if (std::is_integral<TIN>())
            {
                std::generate(args.back().begin(), args.back().end(), rand);
            }
            else
            {
                static ngraph::test::Uniform<float> rng{0, 1, 0};
                rng.initialize(args.back());
            }
        }
    }
    auto baseline_results = execute<TIN, TOUT>(baseline_f, args, "INTERPRETER");
    auto optimized_results = execute<TIN, TOUT>(optimized_f, args, "INTERPRETER");
    return ngraph::test::all_close(baseline_results.at(0), optimized_results.at(0));
}
