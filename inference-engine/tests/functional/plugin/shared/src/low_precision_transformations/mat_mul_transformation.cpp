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
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    MatMulTransformationTestValues testValues;
    std::tie(precision, inputShape, targetDevice, version, testValues) = obj.param;

    std::ostringstream result;
    result << version << "_" <<
        precision << "_" <<
        targetDevice << "_" <<
        testValues.fqOnData1 << "_" <<
        testValues.fqOnData2;

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
        high = 5ul;
    } else if (info.name() == "input2") {
        low = 5ul;
        high = 10ul;
    } else {
        THROW_IE_EXCEPTION << "unexpected input name " << info.name();
    }

    return FuncTestUtils::createAndFillBlobConsistently(info.getTensorDesc(), high - low, low, 1ul);
}

void MatMulTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    MatMulTransformationTestValues testValues;
    std::tie(precision, inputShape, targetDevice, version, testValues) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::MatMulFunction::getOriginal(
        precision,
        testValues.inputShape1,
        testValues.fqOnData1,
        testValues.inputShape2,
        testValues.fqOnData2);

    ngraph::pass::InitNodeInfo().run_on_function(function);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void MatMulTransformation::validate() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    MatMulTransformationTestValues testValues;
    std::tie(precision, inputShape, targetDevice, version, testValues) = this->GetParam();

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
    ASSERT_EQ(2ul, parameters.size());
    EXPECT_EQ("Gemm", layer->type);

    if (params.updatePrecisions) {
        const std::vector<InferenceEngine::CNNLayerPtr> parents = InferenceEngine::details::CNNNetworkHelper::getParents(*layer);
        EXPECT_EQ(2, parents.size());

        EXPECT_TRUE(
            (params.precisionsOnActivations[0] == parents[0]->outData[0]->getTensorDesc().getPrecision()) ||
            (params.precisionsOnActivations[1] == parents[0]->outData[0]->getTensorDesc().getPrecision()));
        EXPECT_EQ(params.precisionsOnWeights[0], parents[1]->outData[0]->getTensorDesc().getPrecision());
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(MatMulTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
