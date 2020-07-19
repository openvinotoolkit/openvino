// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fusing_test_utils.hpp"

using namespace LayerTestsDefinitions;

namespace FusingTestUtils {

std::shared_ptr<ngraph::Function> makeSwishPattern() {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP32);
    auto params = ngraph::builder::makeParams(ngPrc, {fakeShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto sigmoid = ngraph::builder::makeActivation(paramOuts[0], ngPrc, ngraph::helpers::Sigmoid);
    auto multiply = std::make_shared<ngraph::opset1::Multiply>(paramOuts[0], sigmoid);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(multiply)};
    auto func = std::make_shared<ngraph::Function>(results, params, "SwishOptimization");
    return func;
}

std::string postNodes2str(const std::vector<postNode> &postNodes) {
    std::string str;
    for (auto &node : postNodes) {
        str += node.nodePtr->get_type_name();
        std::string paramsStr;
        for (auto &param : node.addInfo) {
            (paramsStr += param.second) += ",";
        }
        if (!paramsStr.empty()) {
            paramsStr.erase(paramsStr.end() - 1);
            str += ("(" + paramsStr + ")");
        }
        str += ",";
    }
    str.erase(str.end() - 1);
    return str;
}

std::shared_ptr<ngraph::Function> makeNgraphFunction(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params,
        const std::shared_ptr<ngraph::Node> &lastNode, const std::shared_ptr<ngraph::Function> &postFunction) {
    auto clonedPostFunction = clone_function(*postFunction);
    clonedPostFunction->set_friendly_name(postFunction->get_friendly_name());

    clonedPostFunction->replace_node(clonedPostFunction->get_parameters()[0], lastNode);
    ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(clonedPostFunction->get_result()->get_input_node_shared_ptr(0))};

    return std::make_shared<ngraph::Function>(results, params, "groupConvolutionFusing");
}

std::shared_ptr<ngraph::Function> makeNgraphFunction(const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params,
        const std::shared_ptr<ngraph::Node> &lastNode, const std::vector<postNode> &postNodes) {
    std::shared_ptr<ngraph::Node> tmpNode = lastNode;

    auto getNewShape = [](ngraph::Shape shape, std::string gran) -> ngraph::Shape {
        ngraph::Shape newShape;
        if (gran == "PerTensor") {
            newShape = ngraph::Shape(shape.size(), 1);
        } else if (gran == "PerChannel") {
            if (shape.size() == 1)
                THROW_IE_EXCEPTION << "If shape.size() == 1 then Granularity can be PerTensor only";
            newShape = ngraph::Shape(shape.size(), 1);
            newShape[1] = shape[1];
        } else {
            newShape = shape;
        }
        return newShape;
    };

    for (auto postNode : postNodes) {
        if (postNode.nodePtr->get_type_name() == std::string("FakeQuantize")) {
            if (postNode.addInfo["Inputs"] == "Parameters") {
                // TODO:
                THROW_IE_EXCEPTION << "FakeQuantize with dynamic inputs is not supported now";
            } else {
                auto newShape = getNewShape(tmpNode->get_shape(), postNode.addInfo["Granularity"]);
                tmpNode = ngraph::builder::makeFakeQuantize(tmpNode, ngPrc, 256, newShape);
            }
        } else {
            ngraph::OutputVector newInputs;
            newInputs.push_back(tmpNode);
            auto newShape = getNewShape(tmpNode->get_shape(), postNode.addInfo["Granularity"]);
            for (int i = 1; i < postNode.nodePtr->get_inputs().size(); i++) {
                if (postNode.addInfo["Inputs"] == "Parameters") {
                    ngraph::ParameterVector newParams = ngraph::builder::makeParams(ngPrc, {newShape});
                    params.insert(params.end(), newParams.begin(), newParams.end());
                    auto newParamOuts = ngraph::helpers::convert2OutputVector(
                            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(newParams));
                    newInputs.push_back(newParamOuts[0]);
                } else {
                    auto constNode = ngraph::builder::makeConstant(ngPrc, newShape, {}, true);
                    newInputs.push_back(constNode);
                }
            }
            tmpNode = postNode.nodePtr->copy_with_new_inputs(newInputs);
        }
    }
    ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(tmpNode)};

    return std::make_shared<ngraph::Function>(results, params, "groupConvolutionFusing");
}

} // namespace FusingTestUtils
