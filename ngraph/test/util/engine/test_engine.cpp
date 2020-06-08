#include "test_engine.hpp"

#include "ngraph/op/get_output_element.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"

using namespace ngraph;

namespace
{
    std::set<NodeTypeInfo> get_ie_ops()
    {
        std::set<NodeTypeInfo> ie_ops = get_opset1().get_type_info_set();
        auto& opset2 = get_opset2().get_type_info_set();
        ie_ops.insert(opset2.begin(), opset2.end());
        auto& opset3 = get_opset3().get_type_info_set();
        ie_ops.insert(opset3.begin(), opset3.end());
        return ie_ops;
    }
}

test::IE_CPU_Engine::IE_CPU_Engine(const std::shared_ptr<Function>& function)
{
    m_function = upgrade_and_validate_function(function);
    m_network = InferenceEngine::CNNNetwork(m_function);
    // set_parameters_and_results(*m_function);
}

std::shared_ptr<Function>
    test::IE_CPU_Engine::upgrade_and_validate_function(std::shared_ptr<Function> function) const
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
                std::cout << "UNSUPPORTED OP DETECTED: " << node->get_type_info().name << std::endl;
                THROW_IE_EXCEPTION << "Detected op not belonging to opset1!";
            }
        }
    }

    return function;
}
