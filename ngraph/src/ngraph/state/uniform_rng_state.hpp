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

#include <functional>
#include <memory>
#include <random>

#include "state.hpp"

namespace ngraph
{
    class UniformRNGState : public State
    {
    public:
        UniformRNGState(std::mt19937::result_type seed)
            : State()
            , m_generator(std::mt19937::result_type(seed))
            , m_distribution()
        {
        }
        UniformRNGState()
            : State()
            , m_generator(std::random_device()())
            , m_distribution()
        {
        }
        virtual void activate() override {}
        virtual void deactivate() override {}
        virtual ~UniformRNGState() override {}
        std::mt19937& get_generator() { return m_generator; }
        std::uniform_real_distribution<double>& get_distribution() { return m_distribution; }
    private:
        std::mt19937 m_generator;
        std::uniform_real_distribution<double> m_distribution;
    };
}
