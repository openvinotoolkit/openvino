#pragma once

#include <ie_core.hpp>
#include "../../util/all_close_f.hpp"
#include "ngraph/function.hpp"

// Builds a class name for a given backend prefix
// The prefix should come from cmake
// Example: INTERPRETER -> INTERPRETER_Engine
// Example: IE_CPU -> IE_CPU_Engine
#define ENGINE_CLASS_NAME(backend) backend##_Engine
namespace ngraph
{
    namespace test
    {
        // TODO - implement when IE_CPU engine is done
        class INTERPRETER_Engine
        {
        public:
            INTERPRETER_Engine(std::shared_ptr<Function> function) {}
            void infer() {}
            testing::AssertionResult compare_results(const size_t tolerance_bits)
            {
                return testing::AssertionSuccess();
            }
            template <typename T>
            void add_input(const Shape& shape, const std::vector<T>& values)
            {
            }
            template <typename T>
            void add_expected_output(ngraph::Shape expected_shape, const std::vector<T>& values)
            {
            }
        };

        // TODO -inherit from IE_CPU_Engine?
        using IE_GPU_Engine = INTERPRETER_Engine;

        class IE_CPU_Engine
        {
        public:
            IE_CPU_Engine() = delete;
            IE_CPU_Engine(std::shared_ptr<Function> function);

            void infer()
            {
                if (m_network_inputs.size() != m_allocated_inputs)
                {
                    THROW_IE_EXCEPTION << "The tested graph has " << m_network_inputs.size()
                                       << " inputs, but " << m_allocated_inputs << " were passed.";
                }
                else
                {
                    m_inference_req.Infer();
                }
            };

            testing::AssertionResult compare_results(const size_t tolerance_bits)
            {
                auto comparison_result = testing::AssertionSuccess();

                for (const auto output : m_network_outputs)
                {
                    InferenceEngine::MemoryBlob::CPtr computed_output_blob =
                        InferenceEngine::as<InferenceEngine::MemoryBlob>(
                            m_inference_req.GetBlob(output.first));

                    const auto& expected_output_blob = m_expected_outputs[output.first];

                    // TODO: assert that both blobs have the same precision?
                    const auto& precision = computed_output_blob->getTensorDesc().getPrecision();

                    // TODO: assert that both blobs have the same number of elements?
                    comparison_result =
                        compare(computed_output_blob, expected_output_blob, precision);

                    if (comparison_result == testing::AssertionFailure())
                    {
                        break;
                    }
                }

                return comparison_result;
            }

            template <typename T>
            void add_input(const Shape& shape, const std::vector<T>& values)
            {
                // Retrieve the next function parameter which has not been set yet.
                // The params are stored in a vector in the order of their creation.
                const auto& function_params = m_function->get_parameters();
                const auto& input_to_allocate = function_params[m_allocated_inputs];
                // TODO: check if input exists
                // Retrieve the corresponding CNNNetwork input using param's friendly name.
                // Here the inputs are stored in the map and are accessible by a string key.
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
                const auto& function_output =
                    m_function->get_results()[m_allocated_expected_outputs];
                // TODO: assert that function_output->get_friendly_name() is in network outputs
                const auto output_info = m_network_outputs[function_output->get_friendly_name()];
                auto blob =
                    std::make_shared<InferenceEngine::TBlob<T>>(output_info->getTensorDesc());
                blob->allocate();
                auto* blob_buffer = blob->wmap().template as<T*>();
                // TODO: assert blob->size() == values.size() ?
                std::copy(values.begin(), values.end(), blob_buffer);

                m_expected_outputs.emplace(function_output->get_friendly_name(), blob);

                ++m_allocated_expected_outputs;
            }

        private:
            std::shared_ptr<Function> m_function;
            InferenceEngine::InputsDataMap m_network_inputs;
            InferenceEngine::OutputsDataMap m_network_outputs;
            InferenceEngine::InferRequest m_inference_req;
            std::map<std::string, InferenceEngine::MemoryBlob::Ptr> m_expected_outputs;
            std::string m_output_name;
            unsigned int m_allocated_inputs = 0;
            unsigned int m_allocated_expected_outputs = 0;

            std::shared_ptr<Function>
                upgrade_and_validate_function(std::shared_ptr<Function> function) const;

            std::set<NodeTypeInfo> get_ie_ops() const;

            template <typename T>
            typename std::enable_if<std::is_floating_point<T>::value,
                                    testing::AssertionResult>::type
                values_match(InferenceEngine::MemoryBlob::CPtr computed,
                             InferenceEngine::MemoryBlob::CPtr expected) const
            {
                const auto computed_data = computed->rmap();
                const auto expected_data = expected->rmap();

                const auto* computed_data_buffer = computed_data.template as<T*>();
                const auto* expected_data_buffer = computed_data.template as<T*>();

                const std::vector<T> computed_values(computed_data_buffer,
                                                     computed_data_buffer + computed->size());
                const std::vector<T> expected_values(expected_data_buffer,
                                                     expected_data_buffer + computed->size());

                return ngraph::test::all_close_f(
                    expected_values, computed_values, DEFAULT_FLOAT_TOLERANCE_BITS);
            }

            testing::AssertionResult compare(InferenceEngine::MemoryBlob::CPtr computed,
                                             InferenceEngine::MemoryBlob::CPtr expected,
                                             const InferenceEngine::Precision& precision) const
            {
                switch (static_cast<InferenceEngine::Precision::ePrecision>(precision))
                {
                case InferenceEngine::Precision::FP32:
                    return values_match<float>(computed, expected);
                    break;
                default: THROW_IE_EXCEPTION << "Not implemented yet";
                }
            }
        };
    }
}