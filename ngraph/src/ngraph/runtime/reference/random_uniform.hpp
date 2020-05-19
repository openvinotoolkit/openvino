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

#include <random>

#include "ngraph/state/uniform_rng_state.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void random_uniform(
                T* out, T min_val, T max_val, size_t count, ngraph::UniformRNGState* rng_state)
            {
                auto& gen = rng_state->get_generator();
                auto& bd = rng_state->get_distribution();

                for (size_t i = 0; i < count; i++)
                {
                    out[i] = static_cast<T>(bd(gen)) * (max_val - min_val) + min_val;
                }
            }

            template <typename T>
            void random_uniform_with_fixed_seed(
                T* out, T min_val, T max_val, size_t count, size_t fixed_seed)
            {
                ngraph::UniformRNGState rng_state(fixed_seed);
                random_uniform(out, min_val, max_val, count, &rng_state);
            }
        }
    }
}
