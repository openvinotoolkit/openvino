// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_engines.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/pass/manager.hpp"
#include "pass/opset1_upgrade.hpp"
#include "shared_utils.hpp"

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

    template <typename T>
    typename std::enable_if<std::is_class<T>::value, testing::AssertionResult>::type
        compare_blobs(InferenceEngine::MemoryBlob::CPtr computed,
                      InferenceEngine::MemoryBlob::CPtr expected,
                      const size_t tolerance_bits)
    {
        const auto test_results = extract_test_results<T>(computed, expected);

        NGRAPH_CHECK(test_results.first.size() == test_results.second.size(),
                     "Number of expected and computed results don't match");

        std::vector<double> expected_double(test_results.first.size());
        std::vector<double> result_double(test_results.second.size());

        for (size_t i = 0; i < test_results.first.size(); ++i)
        {
            expected_double[i] = static_cast<double>(test_results.first[i]);
            result_double[i] = static_cast<double>(test_results.second[i]);
        }

        return ngraph::test::all_close_f(expected_double, result_double, tolerance_bits);
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
        case InferenceEngine::Precision::FP64:
            return compare_blobs<double>(computed, expected, tolerance_bits);
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
        case InferenceEngine::Precision::BF16:
            return compare_blobs<bfloat16>(computed, expected, tolerance_bits);
            break;
        default: THROW_IE_EXCEPTION << "Not implemented yet";
        }
    }
}; // namespace

namespace
{
    InferenceEngine::Precision ng_type_to_precission(const element::Type& target_type)
    {
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
        switch (target_type)
        {
        case element::Type_t::boolean: return InferenceEngine::Precision::BOOL; break;
        case element::Type_t::bf16: return InferenceEngine::Precision::BF16; break;
        case element::Type_t::f16: return InferenceEngine::Precision::FP16; break;
        case element::Type_t::f32: return InferenceEngine::Precision::FP32; break;
        case element::Type_t::f64: return InferenceEngine::Precision::FP64; break;
        case element::Type_t::i8: return InferenceEngine::Precision::I8; break;
        case element::Type_t::i16: return InferenceEngine::Precision::I16; break;
        case element::Type_t::i32: return InferenceEngine::Precision::I32; break;
        case element::Type_t::i64: return InferenceEngine::Precision::I64; break;
        case element::Type_t::u8: return InferenceEngine::Precision::U8; break;
        case element::Type_t::u16: return InferenceEngine::Precision::U16; break;
        case element::Type_t::u32: return InferenceEngine::Precision::U32; break;
        case element::Type_t::u64: return InferenceEngine::Precision::U64; break;
        case element::Type_t::u1: return InferenceEngine::Precision::BIN; break;
        case element::Type_t::i4:
        case element::Type_t::u4:
        case element::Type_t::undefined:
        case element::Type_t::dynamic: throw std::runtime_error("unsupported type");
        }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
        throw std::runtime_error("unsupported type");
    }
} // namespace

test::IE_Engine::IE_Engine(const std::shared_ptr<Function> function, const char* device)
    : m_function{function}
{
    upgrade_and_validate_function(m_function);
    const auto cnn_network = InferenceEngine::CNNNetwork(m_function);
    m_network_inputs = cnn_network.getInputsInfo();
    m_network_outputs = cnn_network.getOutputsInfo();

    for (const auto& result : m_function->get_results())
    {
        const auto& out_name = get_output_name(result);
        m_network_outputs[out_name]->setPrecision(
            ng_type_to_precission(result->get_element_type()));
    }

    InferenceEngine::Core ie;
    auto exe_network = ie.LoadNetwork(cnn_network, device);
    m_inference_req = exe_network.CreateInferRequest();
}

void test::IE_Engine::infer()
{
    if (m_network_inputs.size() != m_allocated_inputs)
    {
        IE_THROW() << "The tested graph has " << m_network_inputs.size() << " inputs, but "
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

std::string test::IE_Engine::get_output_name(const std::shared_ptr<op::v0::Result>& ng_result)
{
    if (m_function->get_results().size() == 1)
    {
        // ng_result argument is ignored
        return m_network_outputs.begin()->first;
    }
    else
    {
        const auto& prev_layer = ng_result->input_value(0);
        auto network_out_name = prev_layer.get_node_shared_ptr()->get_friendly_name();
        if (prev_layer.get_node_shared_ptr()->get_output_size() != 1)
        {
            network_out_name += "." + std::to_string(prev_layer.get_index());
        }

        NGRAPH_CHECK(m_network_outputs.count(network_out_name) == 1,
                     "nGraph function's output number ",
                     m_allocated_expected_outputs,
                     " was not found in the CNNNetwork built from it. Function's output name: ",
                     network_out_name);

        return network_out_name;
    }
}

testing::AssertionResult
    test::IE_Engine::compare_results_with_tolerance_as_fp(const float tolerance)
{
    auto comparison_result = testing::AssertionSuccess();

    for (const auto& output : m_network_outputs)
    {
        if (comparison_result == testing::AssertionFailure())
        {
            break;
        }

        InferenceEngine::MemoryBlob::CPtr computed_output_blob =
            InferenceEngine::as<InferenceEngine::MemoryBlob>(m_inference_req.GetBlob(output.first));

        const auto& expected_output_blob = m_expected_outputs[output.first];

        switch (expected_output_blob->getTensorDesc().getPrecision())
        {
        case InferenceEngine::Precision::FP32:
        {
            const auto test_results =
                extract_test_results<float>(computed_output_blob, expected_output_blob);
            comparison_result =
                test::compare_with_tolerance(test_results.first, test_results.second, tolerance);
            break;
        }
        default:
            comparison_result = testing::AssertionFailure()
                                << "Unsupported data type encountered in "
                                   "'compare_results_with_tolerance_as_fp' method";
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
            IE_THROW() << "Unsupported operator detected in the graph: "
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
    const auto& opset5 = get_opset5().get_type_info_set();
    ie_ops.insert(opset5.begin(), opset5.end());
    const auto& opset6 = get_opset6().get_type_info_set();
    ie_ops.insert(opset6.begin(), opset6.end());
    const auto& opset7 = get_opset7().get_type_info_set();
    ie_ops.insert(opset7.begin(), opset7.end());
    const auto& opset8 = get_opset8().get_type_info_set();
    ie_ops.insert(opset8.begin(), opset8.end());
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
// Without this section the linker is not able to find destructors for missing TBlob specializations
// which are instantiated in the unit tests that use TestCase and this engine
    template <typename T, typename U>
    TBlob<T, U>::~TBlob()
    {
        free();
    }

    template class TBlob<ngraph::bfloat16>;
    template class TBlob<ngraph::float16>;
} // namespace InferenceEngine
