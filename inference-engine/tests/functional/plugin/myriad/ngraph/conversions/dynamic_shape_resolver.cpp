// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

#include "ngraph/opsets/opset3.hpp"
#include "ngraph/function.hpp"

#include "cpp/ie_cnn_network.h"
#include <legacy/cnn_network_impl.hpp>
#include "ie_common.h"

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/common_utils.hpp"

#include <gtest/gtest.h>

namespace {

class DynamicShapeResolverTests : public CommonTestUtils::TestsCommon {
public:
    void SetUp() override {
        const auto tensorType  = ngraph::element::f16;
        const auto shapeType   = ngraph::element::i64;
        const auto tensorShape = std::initializer_list<std::size_t>{1, 800};

        const auto tensor = std::make_shared<ngraph::opset3::Parameter>(tensorType, ngraph::Shape{tensorShape});
        const auto shape  = std::make_shared<ngraph::opset3::Parameter>(shapeType, ngraph::Shape{tensorShape.size()});
        auto dynamicShapeResolver = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(tensor, shape);
        dynamicShapeResolver->set_friendly_name(s_FriendlyName);
        const auto function = std::make_shared<ngraph::Function>(ngraph::NodeVector{dynamicShapeResolver}, ngraph::ParameterVector{tensor, shape});

        cnnNetwork = InferenceEngine::CNNNetwork{function};
        triggerConversionToCNNNetwork();
    }

protected:
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::CNNLayerPtr getDynamicShapeResolverLayer() const {
        return CommonTestUtils::getLayerByName(cnnNetwork, s_FriendlyName);
    }
    IE_SUPPRESS_DEPRECATED_END
    InferenceEngine::CNNNetwork cnnNetwork;

private:
    void triggerConversionToCNNNetwork() {
        IE_SUPPRESS_DEPRECATED_START
        cnnNetwork = InferenceEngine::CNNNetwork(
            std::make_shared<InferenceEngine::details::CNNNetworkImpl>(cnnNetwork));
        IE_SUPPRESS_DEPRECATED_END
    }

    static const char s_FriendlyName[];
};

const char DynamicShapeResolverTests::s_FriendlyName[] = "DSR";

TEST_F(DynamicShapeResolverTests, smoke_NGraphFunctionCanBeConvertedToCNNNetwork) {
    ASSERT_EQ(cnnNetwork.getInputsInfo().size(), 2);
    ASSERT_EQ(cnnNetwork.layerCount(), cnnNetwork.getInputsInfo().size() + 1);
    ASSERT_EQ(cnnNetwork.getOutputsInfo().size(), 1);

    IE_SUPPRESS_DEPRECATED_START
    const auto dynamicShapeResolver = getDynamicShapeResolverLayer();
    ASSERT_EQ(dynamicShapeResolver->type, "DynamicShapeResolver");
    ASSERT_EQ(dynamicShapeResolver->insData.size(), 2);
    ASSERT_EQ(dynamicShapeResolver->outData.size(), 1);
    IE_SUPPRESS_DEPRECATED_END
}

}  // namespace
