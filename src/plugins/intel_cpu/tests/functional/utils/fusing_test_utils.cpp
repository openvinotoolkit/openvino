// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "fusing_test_utils.hpp"

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

std::shared_ptr<ov::Node>
CpuTestWithFusing::modifyGraph(const ov::element::Type &ngPrc, ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) {
    CPUTestsBase::modifyGraph(ngPrc, params, lastNode);
    std::shared_ptr<ov::Node> retNode = lastNode;
    if (postOpMgrPtr) {
        retNode = postOpMgrPtr->addPostOps(ngPrc, params, lastNode);
    }

    return retNode;
}

void CpuTestWithFusing::CheckFusingResults(const std::shared_ptr<const ov::Model>& function, const std::set<std::string>& nodeType) const {
    ASSERT_NE(nullptr, function);
    bool isNodeFound = false;
    for (const auto & op : function->get_ops()) {
        const auto &rtInfo = op->get_rt_info();

        auto getExecValue = [](const std::string &paramName, const ov::Node::RTMap& rtInfo) -> std::string {
            auto it = rtInfo.find(paramName);
            OPENVINO_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        auto layerType = getExecValue("layerType", rtInfo);
        if (nodeType.count(layerType)) {
            isNodeFound = true;
            auto originalLayersNames = getExecValue("originalLayersNames", rtInfo);
            std::string opFriendlyName = op->get_friendly_name();
            ASSERT_TRUE(originalLayersNames.find(opFriendlyName) != std::string::npos)
                << "Operation name " << opFriendlyName << " has not been found in originalLayersNames!";

            size_t pos = 0;
            for (const auto& fusedOp : fusedOps) {
                pos = originalLayersNames.find(fusedOp, checkFusingPosition ? pos : 0);
                if (expectPostOpsToBeFused) {
                    ASSERT_TRUE(pos != std::string::npos) << "Fused op " << fusedOp << " has not been found!";
                } else {
                    ASSERT_TRUE(pos == std::string::npos) << "op" << fusedOp << " should not be fused!";
                }
            }
        }
    }
    std::stringstream error_message;
    error_message << "Node with types \"";
    for (const auto& elem : nodeType)
        error_message << elem << ", ";
    error_message << "\" wasn't found";
    ASSERT_TRUE(isNodeFound) << error_message.str();
}

void CpuTestWithFusing::CheckPluginRelatedResultsImpl(const std::shared_ptr<const ov::Model>& function, const std::set<std::string>& nodeType) const {
    CPUTestsBase::CheckPluginRelatedResultsImpl(function, nodeType);
    CheckFusingResults(function, nodeType);
}

std::shared_ptr<ov::Node>
postFunctionMgr::addPostOps(const ov::element::Type &ngPrc, ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) const {
    auto clonedPostFunction = _pFunction->clone();
    clonedPostFunction->set_friendly_name(_pFunction->get_friendly_name());
    clonedPostFunction->replace_node(clonedPostFunction->get_parameters()[0], lastNode);
    return clonedPostFunction->get_result()->get_input_node_shared_ptr(0);
}

std::string postFunctionMgr::getFusedOpsNames() const {
    return _pFunction->get_friendly_name();
}

postNodesMgr::postNodesMgr(std::vector<postNodeBuilder> postNodes) : _postNodes(std::move(postNodes)) {}

std::shared_ptr<ov::Node>
postNodesMgr::addPostOps(const ov::element::Type &ngPrc, ov::ParameterVector &params, const std::shared_ptr<ov::Node> &lastNode) const {
    std::shared_ptr<ov::Node> tmpNode = lastNode;

    postNodeConfig cfg{lastNode, tmpNode, ngPrc, params};

    for (const auto& postNode : _postNodes) {
        cfg.input = tmpNode;
        tmpNode = postNode.makeNode(cfg);
    }
    return tmpNode;
}

std::string postNodesMgr::getFusedOpsNames() const {
    std::ostringstream result;
    const char* separator = "";
    for (const auto& item : _postNodes) {
        result << separator << item.name;
        separator = ".";
    }
    return result.str();
}
} // namespace CPUTestUtils
