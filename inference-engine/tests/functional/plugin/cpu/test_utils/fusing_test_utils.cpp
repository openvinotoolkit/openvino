// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fusing_test_utils.hpp"

using namespace LayerTestsDefinitions;

namespace CPUTestUtils {


std::string CpuTestWithFusing::getTestCaseName(fusingSpecificParams params) {
    std::ostringstream result;
    std::vector<std::string> fusedOps;
    std::shared_ptr<postOpMgr> postOpMgrPtr;
    std::tie(postOpMgrPtr, fusedOps) = params;

    if (postOpMgrPtr) {
        auto postOpsNames = postOpMgrPtr->getFusedOpsNames();
        if (!postOpsNames.empty()) {
            result << "_Fused=" << postOpsNames;
        }
    }

    return result.str();
}

std::shared_ptr<ngraph::Node>
CpuTestWithFusing::modifyGraph(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params, const std::shared_ptr<ngraph::Node> &lastNode) const {
    CPUTestsBase::modifyGraph(ngPrc, params, lastNode);
    std::shared_ptr<ngraph::Node> retNode = lastNode;
    if (postOpMgrPtr) {
        retNode = postOpMgrPtr->addPostOps(ngPrc, params, lastNode);
    }

    return retNode;
}

void CpuTestWithFusing::CheckFusingResults(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType) const {
    InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();
    auto function = execGraphInfo.getFunction();
    ASSERT_NE(nullptr, function);
    bool isNodeFound = false;
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
            isNodeFound = true;
            auto originalLayersNames = getExecValue("originalLayersNames", rtInfo);
            std::string opFriendlyName = op->get_friendly_name();
            auto pos = originalLayersNames.find(opFriendlyName);
            ASSERT_TRUE(pos != std::string::npos) << "Operation name " << op->get_friendly_name() << " has not been found in originalLayersNames!";
            for (auto fusedOp : fusedOps) {
                pos = originalLayersNames.find(fusedOp, checkFusingPosition ? pos : 0);
                ASSERT_TRUE(pos != std::string::npos) << "Fused op " << fusedOp << " has not been found!";
            }
        }
    }
    ASSERT_TRUE(isNodeFound) << "Node type name: \"" << nodeType << "\" has not been found.";
}

void CpuTestWithFusing::CheckPluginRelatedResults(InferenceEngine::ExecutableNetwork &execNet, std::string nodeType) const {
    CPUTestsBase::CheckPluginRelatedResults(execNet, nodeType);
    CheckFusingResults(execNet, nodeType);
}

std::shared_ptr<ngraph::Node>
postFunctionMgr::addPostOps(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params, const std::shared_ptr<ngraph::Node> &lastNode) const {
    auto clonedPostFunction = clone_function(*_pFunction);
    clonedPostFunction->set_friendly_name(_pFunction->get_friendly_name());
    clonedPostFunction->replace_node(clonedPostFunction->get_parameters()[0], lastNode);
    return clonedPostFunction->get_result()->get_input_node_shared_ptr(0);
}

std::string postFunctionMgr::getFusedOpsNames() const {
    return _pFunction->get_friendly_name();
}

postNodesMgr::postNodesMgr(std::vector<postNodeBuilder> postNodes) : _postNodes(std::move(postNodes)) {}

std::shared_ptr<ngraph::Node>
postNodesMgr::addPostOps(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params, const std::shared_ptr<ngraph::Node> &lastNode) const {
    std::shared_ptr<ngraph::Node> tmpNode = lastNode;

    for (auto postNode : _postNodes) {
        tmpNode = postNode.makeNode(tmpNode, ngPrc, params);
    }
    return tmpNode;
}

std::string postNodesMgr::getFusedOpsNames() const {
    std::ostringstream result;
    const char* separator = "";
    for (const auto& item : _postNodes) {
        result << separator << item.name;
        separator = ",";
    }
    return result.str();
}
} // namespace CPUTestUtils
