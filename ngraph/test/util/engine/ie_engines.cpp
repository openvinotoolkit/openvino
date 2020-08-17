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

#include "ie_engines.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/pass/manager.hpp"
#include "pass/opset1_upgrade.hpp"

using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

namespace
{
    /// Extracts the data from two blobs and returns them as a pair of vectors.
    template <typename T>
    std::pair<std::vector<T>, std::vector<T>>
        extract_test_results(InferenceEngine::MemoryBlob::CPtr computed,
                             InferenceEngine::MemoryBlob::CPtr expected)
    {
        const auto computed_data = computed->rmap();
        const auto expected_data = expected->rmap();

        const auto* computed_data_buffer = computed_data.template as<const T*>();
        const auto* expected_data_buffer = expected_data.template as<const T*>();

        std::vector<T> computed_values(computed_data_buffer,
                                       computed_data_buffer + computed->size());
        std::vector<T> expected_values(expected_data_buffer,
                                       expected_data_buffer + computed->size());

        return std::make_pair(std::move(computed_values), std::move(expected_values));
    }

    /// Compares two blobs containing floating point elements.
    template <typename T>
    typename std::enable_if<std::is_floating_point<T>::value, testing::AssertionResult>::type
        compare_blobs(InferenceEngine::MemoryBlob::CPtr computed,
                      InferenceEngine::MemoryBlob::CPtr expected,
                      const size_t tolerance_bits)
    {
        const auto test_results = extract_test_results<T>(computed, expected);

        return ngraph::test::all_close_f(test_results.first, test_results.second, tolerance_bits);
    }

    /// Compares two blobs containing integer elements.
    template <typename T>
    typename std::enable_if<std::is_integral<T>::value, testing::AssertionResult>::type
        compare_blobs(InferenceEngine::MemoryBlob::CPtr computed,
                      InferenceEngine::MemoryBlob::CPtr expected,
                      const size_t)
    {
        const auto test_results = extract_test_results<T>(computed, expected);

        return ngraph::test::all_close<T>(test_results.first, test_results.second);
    }

    /// Compares two blobs elementwise
    inline testing::AssertionResult compare_blobs(InferenceEngine::MemoryBlob::CPtr computed,
                                                  InferenceEngine::MemoryBlob::CPtr expected,
                                                  const size_t tolerance_bits)
    {
        const auto& computed_precision = computed->getTensorDesc().getPrecision();
        const auto& expected_precision = expected->getTensorDesc().getPrecision();

        if (computed_precision != expected_precision)
        {
            return testing::AssertionFailure();
        }

        switch (static_cast<InferenceEngine::Precision::ePrecision>(computed_precision))
        {
        case InferenceEngine::Precision::FP32:
            return compare_blobs<float>(computed, expected, tolerance_bits);
            break;
        case InferenceEngine::Precision::I8:
            return compare_blobs<int8_t>(computed, expected, tolerance_bits);
            break;
        case InferenceEngine::Precision::I16:
            return compare_blobs<int16_t>(computed, expected, tolerance_bits);
            break;
        case InferenceEngine::Precision::I32:
            return compare_blobs<int32_t>(computed, expected, tolerance_bits);
            break;
        case InferenceEngine::Precision::I64:
            return compare_blobs<int64_t>(computed, expected, tolerance_bits);
            break;
        case InferenceEngine::Precision::U8:
            return compare_blobs<uint8_t>(computed, expected, tolerance_bits);
            break;
        case InferenceEngine::Precision::U16:
            return compare_blobs<uint16_t>(computed, expected, tolerance_bits);
            break;
        case InferenceEngine::Precision::U32:
            return compare_blobs<uint32_t>(computed, expected, tolerance_bits);
            break;
        case InferenceEngine::Precision::U64:
            return compare_blobs<uint64_t>(computed, expected, tolerance_bits);
            break;
        case InferenceEngine::Precision::BOOL:
            return compare_blobs<uint8_t>(computed, expected, tolerance_bits);
            break;
        default: THROW_IE_EXCEPTION << "Not implemented yet";
        }
    }
};

test::IE_Engine::IE_Engine(const std::shared_ptr<Function> function, const char* device)
    : m_function{function}
{
    upgrade_and_validate_function(m_function);
    const auto cnn_network = InferenceEngine::CNNNetwork(m_function);
    m_network_inputs = cnn_network.getInputsInfo();
    m_network_outputs = cnn_network.getOutputsInfo();

    InferenceEngine::Core ie;
    auto exe_network = ie.LoadNetwork(cnn_network, device);
    m_inference_req = exe_network.CreateInferRequest();
}

void test::IE_Engine::infer()
{
    if (m_network_inputs.size() != m_allocated_inputs)
    {
        THROW_IE_EXCEPTION << "The tested graph has " << m_network_inputs.size() << " inputs, but "
                           << m_allocated_inputs << " were passed.";
    }
    else
    {
        m_inference_req.Infer();
    }
}

testing::AssertionResult test::IE_Engine::compare_results(const size_t tolerance_bits)
{
    auto comparison_result = testing::AssertionSuccess();

    for (const auto& output : m_network_outputs)
    {
        InferenceEngine::MemoryBlob::CPtr computed_output_blob =
            InferenceEngine::as<InferenceEngine::MemoryBlob>(m_inference_req.GetBlob(output.first));

        const auto& expected_output_blob = m_expected_outputs[output.first];

        comparison_result =
            compare_blobs(computed_output_blob, expected_output_blob, tolerance_bits);

        if (comparison_result == testing::AssertionFailure())
        {
            break;
        }
    }

    return comparison_result;
}

std::shared_ptr<Function>
    test::IE_Engine::upgrade_and_validate_function(const std::shared_ptr<Function> function) const
{
    pass::Manager passes;
    passes.register_pass<pass::Opset1Upgrade>();
    passes.run_passes(function);

    static std::set<NodeTypeInfo> ie_ops = get_ie_ops();
    for (const auto& node : function->get_ops())
    {
        if (ie_ops.find(node->get_type_info()) == ie_ops.end())
        {
            THROW_IE_EXCEPTION << "Unsupported operator detected in the graph: "
                               << node->get_type_info().name;
        }
    }

    return function;
}

std::set<NodeTypeInfo> test::IE_Engine::get_ie_ops() const
{
    std::set<NodeTypeInfo> ie_ops = get_opset1().get_type_info_set();
    const auto& opset2 = get_opset2().get_type_info_set();
    ie_ops.insert(opset2.begin(), opset2.end());
    const auto& opset3 = get_opset3().get_type_info_set();
    ie_ops.insert(opset3.begin(), opset3.end());
    const auto& opset4 = get_opset4().get_type_info_set();
    ie_ops.insert(opset4.begin(), opset4.end());
    return ie_ops;
}

void test::IE_Engine::reset()
{
    m_allocated_inputs = 0;
    m_allocated_expected_outputs = 0;
    m_expected_outputs.clear();
}

namespace InferenceEngine
{
// those definitions and template specializations are required for clang (both Linux and Mac)
// Without this section the linker is not able to find destructors for missing TBlob specializations
// which are instantiated in the unit tests that use TestCase and this engine
#ifdef __clang__
    template <typename T, typename U>
    TBlob<T, U>::~TBlob()
    {
        free();
    }

    template class TBlob<unsigned int>;
    template class TBlob<bool>;
    template class TBlob<ngraph::bfloat16>;
    template class TBlob<ngraph::float16>;
    template class TBlob<char>;
#endif
}
