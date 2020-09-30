// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/mat_mul_with_constant_transformation.hpp"

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

std::string MatMulWithConstantTransformation::getTestCaseName(testing::TestParamInfo<MatMulWithConstantTransformationParams> obj) {
    ngraph::element::Type precision;
    std::string targetDevice;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    MatMulWithConstantTransformationTestValues testValues;
    std::tie(precision, targetDevice, version, testValues) = obj.param;

    std::ostringstream result;
    result << version << "_" <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.fqOnData << "_" <<
        testValues.fqOnWeights;

    return result.str();
}

InferenceEngine::Blob::Ptr MatMulWithConstantTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    if ((info.name() != "input1") && (info.name() != "input2")) {
        THROW_IE_EXCEPTION << "unexpected layer name " << info.name();
    }

    size_t low;
    size_t high;
    if (info.name() == "input1") {
        low = 1ul;
        high = 5ul;
    } else if (info.name() == "input2") {
        low = 5ul;
        high = 10ul;
    } else {
        THROW_IE_EXCEPTION << "unexpected input name " << info.name();
    }

    return FuncTestUtils::createAndFillBlobConsistently(info.getTensorDesc(), high - low, low, 1ul);
}

void MatMulWithConstantTransformation::SetUp() {
    ngraph::element::Type precision;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    MatMulWithConstantTransformationTestValues testValues;
    std::tie(precision, targetDevice, version, testValues) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::MatMulFunction::getOriginal(
        precision,
        testValues.inputShape,
        testValues.fqOnData,
        testValues.weightsConstShape,
        testValues.weightsConstValues,
        testValues.fqOnWeights);

    ngraph::pass::InitNodeInfo().run_on_function(function);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void MatMulWithConstantTransformation::validate() {
    ngraph::element::Type_t precision;
    std::string targetDevice;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    MatMulWithConstantTransformationTestValues testValues;
    std::tie(precision, targetDevice, version, testValues) = this->GetParam();

    {
        InferenceEngine::CNNNetwork net(function);
        std::shared_ptr<InferenceEngine::ICNNNetwork> clonedNetwork = InferenceEngine::cloneNetwork(net);
        if (!fakeQuantizeExists(*clonedNetwork)) {
            return;
        }
    }

    const auto params = LayerTestsUtils::LayerTransformationParamsFactory::createParams();
    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);

    EXPECT_EQ("ScaleShift", outputLayer->type);

    EXPECT_EQ(1ul, outputLayer->insData.size());
    const InferenceEngine::DataPtr insData = outputLayer->insData[0].lock();
    EXPECT_TRUE(insData != nullptr);
    const InferenceEngine::CNNLayerPtr layer = getCreatorLayer(insData).lock();
    EXPECT_TRUE(layer != nullptr);

    const std::vector<std::shared_ptr<ngraph::op::Parameter>> parameters = function->get_parameters();
    EXPECT_EQ(1ul, parameters.size());
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
        if (parents.size() > 2ul) {
            EXPECT_EQ(InferenceEngine::Precision::FP32, parents[2]->outData[0]->getTensorDesc().getPrecision());
        }
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(MatMulWithConstantTransformation, CompareWithRefImpl) {
    Run();
    if (targetDevice == "CPU") {
        std::string kernel = std::get<3>(this->GetParam()).kernel;
        сheckKernel("FullyConnected", kernel);
    }
};

}  // namespace LayerTestsDefinitions
