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
                m_inference_req.Infer();
            };

            template <typename T>
            void add_input(const Shape& shape, const std::vector<T>& values)
            {
                // retrieve the next function parameter which has not been set yet
                // the params are stored in a vector in the order of creation
                const auto& function_params = m_function->get_parameters();
                const auto& input_to_allocate = function_params[m_allocated_inputs];
                // TODO: check if input exists
                // retrieve the corresponding CNNNetwork input using param's friendly name
                // here the inputs are stored in the map and are accessible by a string key
                const auto& input_info = m_network_inputs[input_to_allocate->get_friendly_name()];

                auto blob =
                    std::make_shared<InferenceEngine::TBlob<T>>(input_info->getTensorDesc());
                blob->allocate();
                auto* blob_buffer = blob->wmap().template as<T*>();
                // TODO: assert blob->size() == values.size() ?
                std::copy(values.begin(), values.end(), blob_buffer);

                m_inference_req.SetBlob(input_to_allocate->get_friendly_name(), blob);

                ++m_allocated_inputs;
            }

            template <typename T>
            void add_expected_output(ngraph::Shape expected_shape, const std::vector<T>& values)
            {
                const auto& function_output = m_function->get_results()[m_expected_outputs];
                std::cout << "Adding blob to output name: " << function_output->get_friendly_name() << std::endl;
                // TODO: assert that function_output->get_friendly_name() is in network outputs
                const auto output_info = m_network_outputs[function_output->get_friendly_name()];
                auto blob =
                    std::make_shared<InferenceEngine::TBlob<T>>(output_info->getTensorDesc());
                blob->allocate();
                auto* blob_buffer = blob->wmap().template as<T*>();
                // TODO: assert blob->size() == values.size() ?
                std::copy(values.begin(), values.end(), blob_buffer);

                m_expected_outputs_map.emplace(function_output->get_friendly_name(), blob);

                ++m_expected_outputs;
            }

            template <typename T>
            std::vector<T> output_data()
            {
                InferenceEngine::MemoryBlob::CPtr output_blob =
                    InferenceEngine::as<InferenceEngine::MemoryBlob>(
                        m_inference_req.GetBlob(m_output_name));

                if (!output_blob)
                {
                    THROW_IE_EXCEPTION << "Cannot retrieve output MemoryBlob for output: "
                                       << m_output_name;
                }

                const T* output_buffer = output_blob->rmap().template as<const T*>();

                return std::vector<T>(output_buffer, output_buffer + output_blob->size());
            }

        private:
            std::shared_ptr<Function> m_function;
            InferenceEngine::InputsDataMap m_network_inputs;
            InferenceEngine::OutputsDataMap m_network_outputs;
            InferenceEngine::InferRequest m_inference_req;
            std::map<std::string, InferenceEngine::Blob::Ptr> m_expected_outputs_map;
            std::string m_output_name;
            unsigned int m_allocated_inputs = 0;
            unsigned int m_expected_outputs = 0;

            std::shared_ptr<Function>
                upgrade_and_validate_function(std::shared_ptr<Function> function) const;

            std::set<NodeTypeInfo> get_ie_ops() const;
        };

        // TODO: implement afterwards
        using INTERPRETER_Engine = IE_CPU_Engine;
        using IE_GPU_Engine = IE_CPU_Engine;
    }
}