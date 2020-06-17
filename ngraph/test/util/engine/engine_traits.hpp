#pragma once

namespace ngraph
{
    namespace test
    {
        /// A specialization of this struct should be created for each engine type
        template <typename Engine>
        struct EngineTraits
        {
        };
    }
}
