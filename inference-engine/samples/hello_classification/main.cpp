// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <samples/classification_results.h>

#include <inference_engine.hpp>
#include <memory>
#include <samples/common.hpp>
#include <string>
#include <vector>
#include <ngraph/graph_util.hpp>
#include <ngraph/variant.hpp>
#include "../../src/plugin_api/ie_algorithm.hpp"

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset1.hpp>


using namespace InferenceEngine;

int main(int argc, char* argv[]) {
    std::string dev = "HETERO:GPU,CPU";
    InferenceEngine::Core core;

    auto ngPrc = ngraph::element::f32;
    auto makeParams = [](const ngraph::element::Type &type, const std::vector<std::vector<size_t>> &shapes){
        ngraph::ParameterVector outs;
        for (const auto &shape : shapes) {
            auto paramNode = std::make_shared<ngraph::opset1::Parameter>(type, ngraph::Shape(shape));
            outs.push_back(paramNode);
        }
        return outs;
    };
    auto params = makeParams(ngPrc, {std::vector<size_t>({10, 10, 10})});
    auto convert2OutputVector = [](const std::vector<std::shared_ptr<ngraph::Node>> &nodes) {
        ngraph::OutputVector outs;
        std::for_each(nodes.begin(), nodes.end(), [&outs](const std::shared_ptr<ngraph::Node> &n) {
            for (const auto &out_p : n->outputs()) {
                outs.push_back(out_p);
            }
        });
        return outs;
    };

    auto castOps2Nodes = [] (const std::vector<std::shared_ptr<ngraph::op::Parameter>> &ops) {
        ngraph::NodeVector nodes;
        for (const auto &op : ops) {
            nodes.push_back(std::dynamic_pointer_cast<ngraph::Node>(op));
        }
        return nodes;
    };
    auto paramIn = convert2OutputVector(castOps2Nodes(params));

    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    auto inType = ngraph::element::f32;
    ngraph::ParameterVector paramVector;
    auto paramData = std::make_shared<ngraph::opset1::Parameter>(inType, ngraph::Shape({10, 10}));
    paramVector.push_back(paramData);

    auto paramOuts = convert2OutputVector(castOps2Nodes(paramVector));

    auto myNode = std::dynamic_pointer_cast<ngraph::op::MyNode>(std::make_shared<ngraph::op::MyNode>(paramOuts[0]));

    auto negat = std::make_shared<ngraph::opset1::Negative>(myNode);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(negat)};
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(results, paramVector, "extFunction");

    auto cnnNetwork = InferenceEngine::CNNNetwork{function};
    QueryNetworkResult qr = core.QueryNetwork(cnnNetwork, dev, {});
    std::cout << "Negative affinity = " << qr.supportedLayersMap["MyNode_2"] << "\n";
    std::cout <<"MyNode affinity = " <<  qr.supportedLayersMap["Negative_3"] << "\n";
    auto clonedFunction = ngraph::clone_function(*function);
    auto orderedOps = clonedFunction->get_ordered_ops();
    for (auto&& node : function->get_ops()) {
        auto& affinity = qr.supportedLayersMap[node->get_friendly_name()];
        // Store affinity mapping using node runtime information
        node->get_rt_info()["affinity"] = std::make_shared<ngraph::VariantWrapper<std::string>>(affinity);
    }

    auto exeNetwork = core.LoadNetwork(cnnNetwork, dev);

    auto req = exeNetwork.CreateInferRequest();
    req.Infer();
    return EXIT_SUCCESS;
}
