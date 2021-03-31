// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
            typename std::enable_if<supports_devices<Engine>::value, Engine>::type
                create_engine_impl(const std::shared_ptr<ngraph::Function> function,
                                   const TestCaseType)
            {
                return Engine{function};
            }

            /// A factory that can create engines which support dynamic backends
            /// but do not support devices. Currently: INTERPRETER_Engine
            template <typename Engine>
            typename std::enable_if<supports_dynamic<Engine>::value, Engine>::type
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
