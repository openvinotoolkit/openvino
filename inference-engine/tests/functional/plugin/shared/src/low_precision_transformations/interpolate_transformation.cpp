// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/interpolate_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ngraph_functions/low_precision_transformations/interpolate_function.hpp"

namespace LayerTestsDefinitions {

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& values) {
    os << "{";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ",";
        }
    }
    os << "}";
    return os;
}

std::string InterpolateTransformation::getTestCaseName(testing::TestParamInfo<InterpolateTransformationParams> obj) {
    ngraph::element::Type precision;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    std::string targetDevice;
    interpAttributes attributes;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(precision, shapes, targetDevice, attributes, version) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, shapes.first, targetDevice, params, version) << "_" <<
        shapes.second << "_" <<
        attributes.align_corners << "_" <<
        attributes.antialias << "_" <<
        attributes.axes << "_" <<
        attributes.mode << "_" <<
        attributes.pads_begin << "_" <<
        attributes.pads_end;
    return result.str();
}

void InterpolateTransformation::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);
    ngraph::element::Type precision;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    interpAttributes attributes;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(precision, shapes, targetDevice, attributes, version) = this->GetParam();

    ngraph::op::InterpolateAttrs interpAttrs;
    interpAttrs.axes = attributes.axes;
    interpAttrs.mode = attributes.mode;
    interpAttrs.align_corners = attributes.align_corners;
    interpAttrs.antialias = attributes.antialias;
    interpAttrs.pads_begin = attributes.pads_begin;
    interpAttrs.pads_end = attributes.pads_end;

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::InterpolateFunction::getOriginal(precision, shapes.first, shapes.second, interpAttrs);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void InterpolateTransformation::validate() {
    ngraph::element::Type precision;
    std::pair<ngraph::Shape, ngraph::Shape> shapes;
    std::string targetDevice;
    interpAttributes interpAttrs;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(precision, shapes, targetDevice, interpAttrs, version) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(toCNNNetwork(params));

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ(interpAttrs.mode == "linear" ? "Interp" : "ScaleShift", outputLayer->type);

    EXPECT_EQ(1ul, outputLayer->insData.size());
    const InferenceEngine::DataPtr insData = outputLayer->insData[0].lock();
    EXPECT_TRUE(insData != nullptr);
    const InferenceEngine::CNNLayerPtr interpolate = getCreatorLayer(insData).lock();
    EXPECT_TRUE(interpolate != nullptr);
    EXPECT_EQ(interpAttrs.mode == "linear" ? "ScaleShift" : "Resample", interpolate->type);

    if (params.updatePrecisions && interpAttrs.mode == "nearest") {
        const InferenceEngine::Precision precision = interpolate->outData[0]->getTensorDesc().getPrecision();
        EXPECT_TRUE((precision == InferenceEngine::Precision::U8) || (precision == InferenceEngine::Precision::I8));
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(InterpolateTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
