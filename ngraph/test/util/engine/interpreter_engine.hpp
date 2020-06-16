#pragma once
#include "../../util/all_close.hpp"
#include "../../util/all_close_f.hpp"
#include "ngraph/function.hpp"

namespace ngraph
{
    namespace test
    {
        // TODO - implement when IE_CPU engine is done
        class INTERPRETER_Engine
        {
        public:
            INTERPRETER_Engine(const std::shared_ptr<Function> function) {}
            void infer() {}
            testing::AssertionResult
                compare_results(const size_t tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS)
            {
                return testing::AssertionSuccess();
            }
            template <typename T>
            void add_input(const Shape& shape, const std::vector<T>& values)
            {
            }
            template <typename T>
            void add_expected_output(const ngraph::Shape& expected_shape,
                                     const std::vector<T>& values)
            {
            }
        };
    }
}