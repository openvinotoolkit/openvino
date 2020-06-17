#include "ie_cpu_engine.hpp"

#include "ngraph/op/get_output_element.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"

using namespace ngraph;

test::IE_CPU_Engine::IE_CPU_Engine(const std::shared_ptr<Function> function, const char* device)
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

void test::IE_CPU_Engine::infer()
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

testing::AssertionResult test::IE_CPU_Engine::compare_results(const size_t tolerance_bits)
{
    auto comparison_result = testing::AssertionSuccess();

    for (const auto output : m_network_outputs)
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

std::shared_ptr<Function> test::IE_CPU_Engine::upgrade_and_validate_function(
    const std::shared_ptr<Function> function) const
{
    pass::Manager passes;
    passes.register_pass<pass::Opset1Upgrade>();
    passes.run_passes(function);

    static std::set<NodeTypeInfo> ie_ops = get_ie_ops();
    for (const auto& node : function->get_ops())
    {
        if (ie_ops.find(node->get_type_info()) == ie_ops.end())
        {
            if (node->get_type_info() == op::GetOutputElement::type_info)
            {
                // IE currently can handle GetOutuputElement op;
                continue;
            }
            else
            {
                THROW_IE_EXCEPTION << "Unsupported operator detected in the graph: "
                                   << node->get_type_info().name;
            }
        }
    }

    return function;
}

std::set<NodeTypeInfo> test::IE_CPU_Engine::get_ie_ops() const
{
    std::set<NodeTypeInfo> ie_ops = get_opset1().get_type_info_set();
    const auto& opset2 = get_opset2().get_type_info_set();
    ie_ops.insert(opset2.begin(), opset2.end());
    const auto& opset3 = get_opset3().get_type_info_set();
    ie_ops.insert(opset3.begin(), opset3.end());
    return ie_ops;
}

void test::IE_CPU_Engine::reset()
{
    m_allocated_inputs = 0;
    m_allocated_expected_outputs = 0;
    m_expected_outputs.clear();
}
