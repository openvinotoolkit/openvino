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
            /// A factory that can create engines supporting devices
            template <typename Engine>
            typename std::enable_if<EngineTraits<Engine>::supports_devices, Engine>::type
                create_engine_impl(const std::shared_ptr<ngraph::Function> function,
                                   const TestCaseType)
            {
                return Engine{function, EngineTraits<Engine>::device};
            }

            /// A factory that can create engines supporting dynamic backends
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
