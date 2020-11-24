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

#include <gtest/gtest.h>

namespace ngraph
{
    namespace test
    {
        /// An interface that each test case engine needs to implement. This interface wraps
        /// a couple of generic methods which are required by the TestCase class to execute
        /// a unit test for a given ngraph::Function.
        /// The interface operates on C++ types while internally it can use implementation-specific
        /// types, containers and structures.
        class TestCaseEngine
        {
        public:
            virtual ~TestCaseEngine() noexcept = default;

            /// Performs the inference using data stored as internal state
            virtual void infer() = 0;

            /// Resets the internal state so that the test can be executed again
            virtual void reset() = 0;

            /// Compares computed and expected results, returns AssertionSuccess or AssertionFailure
            virtual testing::AssertionResult compare_results(const size_t tolerance_bits) = 0;

            /// Compares computed and expected results, returns AssertionSuccess or AssertionFailure
            virtual testing::AssertionResult
                compare_results_with_tolerance_as_fp(const float tolerance) = 0;

            /// Additionally the interface implementing class needs to define
            /// the following 2 methods. They are called from the TestCase class
            /// but they can't be a part of interface since they need to be declared as templates

            /// Passes data (along with its shape) to the next available input.
            /// The data should be stored as internal state, not necessarily as vectors
            // template <typename T>
            // void add_input(const Shape& shape, const std::vector<T>& values)

            /// Sets the expected data (along with its shape) for the next available output
            /// The data should be stored as internal state, not necessarily as vectors
            // template <typename T>
            // void add_expected_output(const ngraph::Shape& expected_shape,
            //                          const std::vector<T>& values)
        };
    }
}
