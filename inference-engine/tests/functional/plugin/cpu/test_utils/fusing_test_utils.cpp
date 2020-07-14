// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fusing_test_utils.hpp"

using namespace LayerTestsDefinitions;

namespace FusingTestUtils {

std::shared_ptr<ngraph::Function> makeSwishPattern(std::vector<size_t> shape) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP32);
    auto params = ngraph::builder::makeParams(ngPrc, {shape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto sigmoid = ngraph::builder::makeActivation(paramOuts[0], ngPrc, ngraph::helpers::Sigmoid);
    auto multiply = std::make_shared<ngraph::opset1::Multiply>(paramOuts[0], sigmoid);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(multiply)};
    auto func = std::make_shared<ngraph::Function>(results, params, "SwishOptimization");
    return func;
}

std::shared_ptr<ngraph::Function> makeFakeQuantizeActivationPattern(size_t levels, ngraph::helpers::ActivationTypes type, std::vector<size_t> shape) {
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP32);
    auto params = ngraph::builder::makeParams(ngPrc, {shape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto fq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngPrc, levels, {1, 1, 1, 1});
    auto activation = ngraph::builder::makeActivation(fq, ngPrc, type);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(activation)};
    auto func = std::make_shared<ngraph::Function>(results, params, "FQ_Activation");
    return func;
}

std::string postNodes2str(const std::vector<std::shared_ptr<ngraph::Node>> &postNodes) {
    std::string str;
    for (auto &node : postNodes) {
        std::string typeName = node->get_type_name();
        if (typeName != "Constant" && typeName != "Parameter") {
            (str += typeName) += ",";
        }
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
        const std::shared_ptr<ngraph::Node> &lastNode, const std::vector<std::shared_ptr<ngraph::Node>> &postNodes) {
    std::shared_ptr<ngraph::Node> tmpNode = lastNode;
    ngraph::OutputVector newInputs;
    for (auto &node : postNodes) {
        if (newInputs.empty()) {
            newInputs.push_back(tmpNode);
        }
        if (std::string(node->get_type_name()) == std::string("Constant")) {
            auto shape = tmpNode->get_shape();
            ngraph::Shape newShape(shape.size(), 1);
            newShape[1] = shape[1];
            auto constNode = ngraph::builder::makeConstant(ngPrc, newShape, {}, true);
            newInputs.push_back(constNode);
        } else if (std::string(node->get_type_name()) == std::string("Parameter")) {
            auto shape = tmpNode->get_shape();
            ngraph::ParameterVector newParams = ngraph::builder::makeParams(ngPrc, {shape});
            params.insert(params.end(), newParams.begin(), newParams.end());
            auto newParamOuts = ngraph::helpers::convert2OutputVector(
                    ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(newParams));
            newInputs.push_back(newParamOuts[0]);
        } else {
            tmpNode = node->copy_with_new_inputs(newInputs);
            newInputs.clear();
        }
    }
    ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(tmpNode)};

    return std::make_shared<ngraph::Function>(results, params, "groupConvolutionFusing");
}

} // namespace FusingTestUtils
