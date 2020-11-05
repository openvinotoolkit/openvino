// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fusing_test_utils.hpp"

using namespace LayerTestsDefinitions;

namespace CPUTestUtils {


std::string CpuTestWithFusing::getTestCaseName(fusingSpecificParams params) {
    std::ostringstream result;
    std::shared_ptr<ngraph::Function> postFunction;
    std::vector<postNode> postNodes;
    std::vector<std::string> fusedOps;
    std::tie(postFunction, postNodes, fusedOps) = params;

    if (postFunction) {
        result << "_Fused=" << postFunction->get_friendly_name();
    } else if (!postNodes.empty()) {
        result << "_Fused=";
        const char* separator = "";
        for (const auto& item : postNodes) {
            result << separator << item.name;
            separator = ",";
        }
    }

    return result.str();
}

std::shared_ptr<ngraph::Node>
CpuTestWithFusing::modifyGraph(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params, const std::shared_ptr<ngraph::Node> &lastNode) const {
    CPUTestsBase::modifyGraph(ngPrc, params, lastNode);
    std::shared_ptr<ngraph::Node> retNode = lastNode;
    if (postFunction) {
        retNode = addPostFunction(lastNode);
    } else if (!postNodes.empty()) {
        retNode = addPostNodes(ngPrc, params, lastNode);
    }
    return retNode;
}

std::shared_ptr<ngraph::Node> CpuTestWithFusing::addPostFunction(const std::shared_ptr<ngraph::Node> &lastNode) const {
    auto clonedPostFunction = clone_function(*postFunction);
    clonedPostFunction->set_friendly_name(postFunction->get_friendly_name());
    clonedPostFunction->replace_node(clonedPostFunction->get_parameters()[0], lastNode);
    return clonedPostFunction->get_result()->get_input_node_shared_ptr(0);
}

std::shared_ptr<ngraph::Node>
CpuTestWithFusing::addPostNodes(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params, const std::shared_ptr<ngraph::Node> &lastNode) const {
    std::shared_ptr<ngraph::Node> tmpNode = lastNode;

    for (auto postNode : postNodes) {
        tmpNode = postNode.makeNode(tmpNode, ngPrc, params);
    }
    return tmpNode;
}

void CpuTestWithFusing::CheckPluginRelatedResults(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType) const {
    CPUTestsBase::CheckPluginRelatedResults(execNet, nodeType);
    InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();
    auto function = execGraphInfo.getFunction();
    ASSERT_NE(nullptr, function);
    for (const auto & op : function->get_ops()) {
        const auto &rtInfo = op->get_rt_info();

        auto getExecValue = [](const std::string &paramName, const ngraph::Node::RTMap& rtInfo) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);

            return value->get();
        };

        auto layerType = getExecValue("layerType", rtInfo);
        if (layerType == nodeType) {
            auto originalLayersNames = getExecValue("originalLayersNames", rtInfo);
            auto pos = originalLayersNames.find(nodeType);
            ASSERT_TRUE(pos != std::string::npos) << "Node type " << nodeType << " has not been found!";
            for (auto fusedOp : fusedOps) {
                pos = originalLayersNames.find(fusedOp, pos);
                ASSERT_TRUE(pos != std::string::npos) << "Fused op " << fusedOp << " has not been found!";
            }
        }
    }
}
} // namespace CPUTestUtils
