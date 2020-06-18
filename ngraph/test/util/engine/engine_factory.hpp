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

#include "ngraph/function.hpp"
#include "util/engine/engine_traits.hpp"

namespace ngraph
{
    namespace test
    {
        enum class TestCaseType
        {
            STATIC,
            DYNAMIC
        };

        namespace
        {
            /// A factory that can create engines supporting devices but not dynamic backends.
            /// Currently: IE_CPU_Backend and IE_GPU_Backend
            template <typename Engine>
            typename std::enable_if<EngineTraits<Engine>::supports_devices, Engine>::type
                create_engine_impl(const std::shared_ptr<ngraph::Function> function,
                                   const TestCaseType)
            {
                return Engine{function, EngineTraits<Engine>::device};
            }

            /// A factory that can create engines which support dynamic backends
            /// but do not support devices. Currently: INTERPRETER_Engine
            template <typename Engine>
            typename std::enable_if<EngineTraits<Engine>::supports_dynamic, Engine>::type
                create_engine_impl(const std::shared_ptr<ngraph::Function> function,
                                   const TestCaseType tct)
            {
                if (tct == TestCaseType::DYNAMIC)
                {
                    return Engine::dynamic(function);
                }
                else
                {
                    return Engine{function};
                }
            }
        }

        /// A factory that is able to create all types of test Engines
        /// in both static and dynamic mode
        template <typename Engine>
        Engine create_engine(const std::shared_ptr<ngraph::Function> function,
                             const TestCaseType tct)
        {
            return create_engine_impl<Engine>(function, tct);
        };
    }
}
