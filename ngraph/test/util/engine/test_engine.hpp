#pragma once

#include "ngraph/function.hpp"

namespace ngraph
{
    namespace test
    {
        class IE_CPU_Engine
        {
        public:
            IE_CPU_Engine(const std::shared_ptr<Function>& function)
                : m_function{function}
            {
            }

            virtual ~IE_CPU_Engine() noexcept = default;

            void infer()
            {
                std::cout << "Running function inference on IE Engine\n";
            };

            template <typename T>
            void add_input(const Shape& shape, const std::vector<T>& values)
            {
                std::cout << "Adding input data to IE Engine\n";
            }

        private:
            std::shared_ptr<Function> m_function;
        };

        // TODO: implement afterwards
        using INTERPRETER_Engine = IE_CPU_Engine;
        using IE_GPU_Engine = IE_CPU_Engine;
    }
}