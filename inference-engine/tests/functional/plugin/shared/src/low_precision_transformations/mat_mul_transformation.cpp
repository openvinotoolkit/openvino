// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/mat_mul_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <queue>
#include <ie_core.hpp>

#include "ngraph/op/op.hpp"
#include <transformations/init_node_info.hpp>
#include "low_precision_transformations/mat_mul_transformation.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/mat_mul_function.hpp"

namespace LayerTestsDefinitions {

std::string MatMulTransformation::getTestCaseName(testing::TestParamInfo<MatMulTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::vector<std::shared_ptr<ngraph::Node>> nodes;
    std::tie(netPrecision, inputShapes, targetDevice, params, version, nodes) = obj.param;

    std::ostringstream result;
    result << version << "_" << nodes[0]->get_shape() << "_" << nodes[1]->get_shape() << "_" << netPrecision.name() << "_" << targetDevice << "_" << toString(params);
    return result.str();
}

InferenceEngine::Blob::Ptr MatMulTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    if ((info.name() != "input1") && (info.name() != "input2")) {
        THROW_IE_EXCEPTION << "unexpected layer name " << info.name();
    }

    size_t low;
    size_t high;
    if (info.name() == "input1") {
        low = 1ul;
        high = 10ul;
    } else {
        low = 10ul;
        high = 20ul;
    }

    return FuncTestUtils::createAndFillBlobConsistently(info.getTensorDesc(), high - low, low, 1ul);
}

void MatMulTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::vector<std::shared_ptr<ngraph::Node>> nodes;
    std::tie(netPrecision, inputShape, targetDevice, params, version, nodes) = this->GetParam();

    ConfigurePlugin(version);

    const auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    function = ngraph::builder::subgraph::MatMulFunction::getOriginal(ngPrecision, inputShape, nodes);

    ngraph::pass::InitNodeInfo().run_on_function(function);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void MatMulTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::vector<std::shared_ptr<ngraph::Node>> nodes;
    std::tie(netPrecision, inputShape, targetDevice, params, version, nodes) = this->GetParam();

    {
        InferenceEngine::CNNNetwork net(function);
        std::shared_ptr<InferenceEngine::ICNNNetwork> clonedNetwork = InferenceEngine::cloneNetwork(net);
        if (!fakeQuantizeExists(*clonedNetwork)) {
            return;
        }
    }

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = it->second->getCreatorLayer().lock();
    EXPECT_TRUE(outputLayer != nullptr);

    const std::vector<std::shared_ptr<ngraph::op::Parameter>> inputs = ngraph::builder::subgraph::MatMulFunction::getInputs(nodes);
    if ((inputs.size() == 2ul) && (params.precisionsOnActivations[0] == InferenceEngine::Precision::U8)) {
        // TODO: not completed
        return;
    }

    EXPECT_EQ("ScaleShift", outputLayer->type);

    EXPECT_EQ(1ul, outputLayer->insData.size());
    const InferenceEngine::DataPtr insData = outputLayer->insData[0].lock();
    EXPECT_TRUE(insData != nullptr);
    const InferenceEngine::CNNLayerPtr layer = insData->getCreatorLayer().lock();
    EXPECT_TRUE(layer != nullptr);

    if (inputs.size() == 2ul) {
        EXPECT_EQ("Gemm", layer->type);

        if (params.updatePrecisions) {
            const std::vector<InferenceEngine::CNNLayerPtr> parents = InferenceEngine::details::CNNNetworkHelper::getParents(*layer);
            EXPECT_EQ(2, parents.size());

            EXPECT_EQ(params.precisionsOnActivations[0], parents[0]->outData[0]->getTensorDesc().getPrecision());
            EXPECT_EQ(params.precisionsOnWeights[0], parents[1]->outData[0]->getTensorDesc().getPrecision());
        }
    } else {
        EXPECT_EQ("FullyConnected", layer->type);

        if (params.updatePrecisions) {
            const std::vector<InferenceEngine::CNNLayerPtr> parents = InferenceEngine::details::CNNNetworkHelper::getParents(*layer);
            EXPECT_EQ(3, parents.size());

            InferenceEngine::CNNLayerPtr fakeQuantizeOnActivations;
            if (parents[0]->type == "FakeQuantize") {
                fakeQuantizeOnActivations = parents[0];
            } else {
                EXPECT_EQ("Eltwise", parents[0]->type) << "unexpected layer type " << parents[0]->type << " " << parents[0]->name;
                const InferenceEngine::CNNLayerPtr parent = InferenceEngine::details::CNNNetworkHelper::getParent(*parents[0]);
                EXPECT_EQ("FakeQuantize", parent->type) << "unexpected layer type " << parents[0]->type << " " << parents[0]->name;
                fakeQuantizeOnActivations = parent;
            }
            EXPECT_EQ(params.precisionsOnActivations[0], fakeQuantizeOnActivations->outData[0]->getTensorDesc().getPrecision());
            EXPECT_EQ(params.precisionsOnWeights[0], parents[1]->outData[0]->getTensorDesc().getPrecision());
            EXPECT_EQ(InferenceEngine::Precision::FP32, parents[2]->outData[0]->getTensorDesc().getPrecision());
        }
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(MatMulTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
