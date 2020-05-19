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

#include "ngraph/state/bernoulli_rng_state.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void generate_mask(T* out,
                               size_t count,
                               ngraph::BernoulliRNGState* rng_state,
                               bool training)
            {
                auto& gen = rng_state->get_generator();
                auto& bd = rng_state->get_distribution();

                for (size_t i = 0; i < count; i++)
                {
                    out[i] = training ? static_cast<T>(bd(gen)) : static_cast<T>(1);
                }
            }

            template <typename T>
            void generate_mask_no_state(
                T* out, size_t count, bool training, uint32_t seed, double prob)
            {
                std::mt19937 gen(seed);
                std::bernoulli_distribution bd(prob);

                for (size_t i = 0; i < count; i++)
                {
                    out[i] = training ? static_cast<T>(bd(gen)) : static_cast<T>(1);
                }
            }
        }
    }
}
