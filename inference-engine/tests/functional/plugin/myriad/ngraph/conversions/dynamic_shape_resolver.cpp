// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

#include "ngraph/op/parameter.hpp"
#include "ngraph/function.hpp"

#include "cpp/ie_cnn_network.h"
#include "ie_common.h"

#include "common_test_utils/test_common.hpp"

#include <gtest/gtest.h>

namespace {

class DynamicShapeResolverTests : public CommonTestUtils::TestsCommon {
public:
    void SetUp() override {
        const auto tensorType  = ngraph::element::f16;
        const auto shapeType   = ngraph::element::u64;
        const auto tensorShape = std::initializer_list<std::size_t>{1, 800};

        const auto tensor = std::make_shared<ngraph::op::Parameter>(tensorType, ngraph::Shape{tensorShape});
        const auto shape  = std::make_shared<ngraph::op::Parameter>(shapeType, ngraph::Shape{tensorShape.size()});
        auto dynamicShapeResolver = std::make_shared<ngraph::op::DynamicShapeResolver>(tensor, shape);
        dynamicShapeResolver->set_friendly_name(s_FriendlyName);
        const auto function = std::make_shared<ngraph::Function>(ngraph::NodeVector{dynamicShapeResolver}, ngraph::ParameterVector{tensor, shape});

        cnnNetwork = InferenceEngine::CNNNetwork{function};
        triggerConversionToCNNNetwork();
    }

protected:
    InferenceEngine::CNNLayerPtr getDynamicShapeResolverLayer() const {
        return cnnNetwork.getLayerByName(s_FriendlyName.c_str());
    }
    InferenceEngine::CNNNetwork cnnNetwork;

private:
    void triggerConversionToCNNNetwork() {
        cnnNetwork.begin();
    }

    static const std::string s_FriendlyName;
};

const std::string DynamicShapeResolverTests::s_FriendlyName = "DSR";

TEST_F(DynamicShapeResolverTests, NGraphFunctionCanBeConvertedToCNNNetwork) {
    ASSERT_EQ(cnnNetwork.getInputsInfo().size(), 2);
    ASSERT_EQ(cnnNetwork.layerCount(), cnnNetwork.getInputsInfo().size() + 1);
    ASSERT_EQ(cnnNetwork.getOutputsInfo().size(), 1);

    const auto dynamicShapeResolver = getDynamicShapeResolverLayer();
    ASSERT_EQ(dynamicShapeResolver->type, "DynamicShapeResolver");
    ASSERT_EQ(dynamicShapeResolver->insData.size(), 2);
    ASSERT_EQ(dynamicShapeResolver->outData.size(), 1);
}

}  // namespace
