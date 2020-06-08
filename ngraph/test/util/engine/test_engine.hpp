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
            IE_CPU_Engine(std::shared_ptr<Function> function);

            virtual ~IE_CPU_Engine() noexcept = default;

            void infer()
            {
                // TODO moved from runtime::ie::IE_Executable::call - check if the number of created
                //      inputs matches input info size
                // if (input_info.size() != inputs.size())
                // {
                //     THROW_IE_EXCEPTION << "Function inputs number differ from number of given
                //     inputs";
                // }
                std::cout << "Running function inference on IE Engine\n";
            };

            template <typename T>
            void add_input(const Shape& shape, const std::vector<T>& values)
            {
                std::cout << "Adding input data to IE Engine\n";
            }

        private:
            // std::shared_ptr<Function> m_function;
            InferenceEngine::InputsDataMap m_network_inputs;
            InferenceEngine::InferRequest m_inference_req;

            std::shared_ptr<Function>
                upgrade_and_validate_function(std::shared_ptr<Function> function) const;

            std::set<NodeTypeInfo> get_ie_ops() const;
        };

        // TODO: implement afterwards
        using INTERPRETER_Engine = IE_CPU_Engine;
        using IE_GPU_Engine = IE_CPU_Engine;
    }
}