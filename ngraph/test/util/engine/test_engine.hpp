#pragma once

#include <ie_core.hpp>
#include "ngraph/function.hpp"

namespace ngraph
{
    namespace test
    {
        class IE_CPU_Engine
        {
        public:
            IE_CPU_Engine() = delete;
            IE_CPU_Engine(const std::shared_ptr<Function>& function);

            virtual ~IE_CPU_Engine() noexcept = default;

            void infer() { std::cout << "Running function inference on IE Engine\n"; };
            template <typename T>
            void add_input(const Shape& shape, const std::vector<T>& values)
            {
                std::cout << "Adding input data to IE Engine\n";
            }

        private:
            std::shared_ptr<Function> m_function;
            InferenceEngine::CNNNetwork m_network;

            std::shared_ptr<Function>
                upgrade_and_validate_function(std::shared_ptr<Function> function) const;
        };

        // TODO: implement afterwards
        using INTERPRETER_Engine = IE_CPU_Engine;
        using IE_GPU_Engine = IE_CPU_Engine;
    }
}