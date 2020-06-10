#pragma once

#include <ie_core.hpp>
#include "../../util/all_close_f.hpp"
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
                std::cout << "Running inference with IE_CPU Engine\n";
                m_inference_req.Infer();
            };

            ::testing::AssertionResult compare_results()
            {
                std::cout << "Comparing results with IE_CPU Engine\n";
                auto comparison_result = testing::AssertionSuccess();

                for (const auto output : m_network_outputs)
                {
                    InferenceEngine::MemoryBlob::CPtr computed_output_blob =
                        InferenceEngine::as<InferenceEngine::MemoryBlob>(
                            m_inference_req.GetBlob(output.first));

                    const auto& expected_output_blob = m_expected_outputs_map[output.first];

                    // TODO: assert that both blobs have the same precision?
                    const auto& precision = computed_output_blob->getTensorDesc().getPrecision();

                    // TODO: assert that both blobs have the same number of elements?
                    comparison_result =
                        compare(computed_output_blob, expected_output_blob, precision);

                    if (comparison_result == ::testing::AssertionFailure())
                    {
                        break;
                    }
                }

                std::cout << "Comparisone done\n";

                return comparison_result;
            }

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
            std::map<std::string, InferenceEngine::MemoryBlob::Ptr> m_expected_outputs_map;
            std::string m_output_name;
            unsigned int m_allocated_inputs = 0;
            unsigned int m_expected_outputs = 0;

            std::shared_ptr<Function>
                upgrade_and_validate_function(std::shared_ptr<Function> function) const;

            std::set<NodeTypeInfo> get_ie_ops() const;

            // using blob_comparator_t = std::function<::testing::AssertionResult(
            //     const InferenceEngine::MemoryBlob::CPtr, const
            //     InferenceEngine::MemoryBlob::CPtr)>;

            template <typename T>
            ::testing::AssertionResult
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

            ::testing::AssertionResult compare(InferenceEngine::MemoryBlob::CPtr computed,
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

        // TODO: implement afterwards
        using INTERPRETER_Engine = IE_CPU_Engine;
        using IE_GPU_Engine = IE_CPU_Engine;
    }
}