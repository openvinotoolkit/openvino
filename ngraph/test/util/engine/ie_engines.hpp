//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <ie_core.hpp>
#include "ngraph/function.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/engine_traits.hpp"
#include "util/engine/test_case_engine.hpp"

namespace ngraph
{
    namespace test
    {
        /// A generic engine that uses OV objects natively
        class IE_Engine : public TestCaseEngine
        {
        public:
            IE_Engine() = delete;

            /// Constructs an IE test engine for a given device (plugin)
            IE_Engine(const std::shared_ptr<Function> function, const char* device);

            void infer() override;

            testing::AssertionResult compare_results(
                const size_t tolerance_bits = DEFAULT_FLOAT_TOLERANCE_BITS) override;

            testing::AssertionResult
                compare_results_with_tolerance_as_fp(const float tolerance = 1.0e-5f) override;

            void reset() override;

            template <typename T>
            void add_input(const Shape& shape, const std::vector<T>& values)
            {
                // Retrieve the next function parameter which has not been set yet.
                // The params are stored in a vector in the order of their creation.
                const auto& function_params = m_function->get_parameters();
                const auto& input_to_allocate = function_params[m_allocated_inputs];

                NGRAPH_CHECK(
                    m_network_inputs.count(input_to_allocate->get_friendly_name()) == 1,
                    "nGraph function's input number ",
                    m_allocated_inputs,
                    " was not found in the CNNNetwork built from it. Function's input name: ",
                    input_to_allocate->get_friendly_name());

                // Retrieve the corresponding CNNNetwork input using param's friendly name.
                // Here the inputs are stored in the map and are accessible by a string key.
                const auto& input_info = m_network_inputs[input_to_allocate->get_friendly_name()];

                auto blob =
                    std::make_shared<InferenceEngine::TBlob<T>>(input_info->getTensorDesc());
                blob->allocate();
                auto* blob_buffer = blob->wmap().template as<T*>();

                NGRAPH_CHECK(blob->size() == values.size(),
                             "The allocated blob for input '",
                             input_to_allocate->get_friendly_name(),
                             " ' expects ",
                             blob->size(),
                             " elements while ",
                             values.size(),
                             " were provided.");

                std::copy(values.begin(), values.end(), blob_buffer);

                m_inference_req.SetBlob(input_to_allocate->get_friendly_name(), blob);

                ++m_allocated_inputs;
            }

            template <typename T>
            void add_expected_output(const ngraph::Shape& expected_shape,
                                     const std::vector<T>& values)
            {
                std::string network_out_name;
                InferenceEngine::DataPtr network_output;
                if (m_function->get_results().size() == 1)
                {
                    network_out_name = m_network_outputs.begin()->first;
                    network_output = m_network_outputs.begin()->second;
                }
                else
                {
                    const auto& function_output =
                        m_function->get_results()[m_allocated_expected_outputs];

                    network_out_name = function_output->get_friendly_name();

                    NGRAPH_CHECK(
                        m_network_outputs.count(network_out_name) == 1,
                        "nGraph function's output number ",
                        m_allocated_expected_outputs,
                        " was not found in the CNNNetwork built from it. Function's output name: ",
                        function_output->get_friendly_name());

                    network_output = m_network_outputs[function_output->get_friendly_name()];
                }

                auto blob =
                    std::make_shared<InferenceEngine::TBlob<T>>(network_output->getTensorDesc());
                blob->allocate();

                NGRAPH_CHECK(blob->size() == values.size(),
                             "The allocated blob for output '",
                             network_out_name,
                             " ' expects ",
                             blob->size(),
                             " elements while ",
                             values.size(),
                             " were provided.");

                auto* blob_buffer = blob->wmap().template as<T*>();
                std::copy(values.begin(), values.end(), blob_buffer);

                m_expected_outputs.emplace(network_out_name, blob);

                ++m_allocated_expected_outputs;
            }

        private:
            const std::shared_ptr<Function> m_function;
            InferenceEngine::InputsDataMap m_network_inputs;
            InferenceEngine::OutputsDataMap m_network_outputs;
            InferenceEngine::InferRequest m_inference_req;
            std::map<std::string, InferenceEngine::MemoryBlob::Ptr> m_expected_outputs;
            unsigned int m_allocated_inputs = 0;
            unsigned int m_allocated_expected_outputs = 0;

            /// Upgrades functions containing legacy opset0 to opset1
            /// and checks if the graph can be executed
            std::shared_ptr<Function>
                upgrade_and_validate_function(const std::shared_ptr<Function> function) const;

            /// Retrieves a set of all ops IE can execute
            std::set<NodeTypeInfo> get_ie_ops() const;
        };

        class IE_CPU_Engine final : public IE_Engine
        {
        public:
            IE_CPU_Engine(const std::shared_ptr<Function> function)
                : IE_Engine{function, m_device}
            {
            }

        private:
            static constexpr const char* m_device = "CPU";
        };

        class IE_GPU_Engine final : public IE_Engine
        {
        public:
            IE_GPU_Engine(const std::shared_ptr<Function> function)
                : IE_Engine{function, m_device}
            {
            }

        private:
            static constexpr const char* m_device = "GPU";
        };

        template <>
        struct supports_devices<IE_CPU_Engine>
        {
            static constexpr bool value = true;
        };

        template <>
        struct supports_devices<IE_GPU_Engine>
        {
            static constexpr bool value = true;
        };
    }
}
