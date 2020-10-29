// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"
#include "ie_precision.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/prior_box_clustered.hpp"
#include "legacy/ngraph_ops/prior_box_clustered_ie.hpp"

namespace LayerTestsDefinitions {
std::string PriorBoxClusteredLayerTest::getTestCaseName(const testing::TestParamInfo<priorBoxClusteredLayerParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes, imageShapes;
    std::string targetDevice;
    priorBoxClusteredSpecificParams specParams;
    std::tie(specParams,
        netPrecision,
        inPrc, outPrc, inLayout, outLayout,
        inputShapes,
        imageShapes,
        targetDevice) = obj.param;

    std::vector<float> widths, heights, variances;
    float step_width, step_height, offset;
    bool clip;
    std::tie(widths,
        heights,
        clip,
        step_width,
        step_height,
        offset,
        variances) = specParams;

    std::ostringstream result;
    const char separator = '_';

    result << "IS="      << CommonTestUtils::vec2str(inputShapes) << separator;
    result << "imageS="  << CommonTestUtils::vec2str(imageShapes) << separator;
    result << "netPRC="  << netPrecision.name()   << separator;
    result << "inPRC="   << inPrc.name() << separator;
    result << "outPRC="  << outPrc.name() << separator;
    result << "inL="     << inLayout << separator;
    result << "outL="    << outLayout << separator;
    result << "widths="  << CommonTestUtils::vec2str(widths)  << separator;
    result << "heights=" << CommonTestUtils::vec2str(heights) << separator;
    result << "variances=";
    if (variances.empty())
        result << "()" << separator;
    else
        result << CommonTestUtils::vec2str(variances) << separator;
    result << "stepWidth="  << step_width  << separator;
    result << "stepHeight=" << step_height << separator;
    result << "offset="     << offset      << separator;
    result << "clip=" << std::boolalpha << clip << separator;
    result << "trgDev=" << targetDevice;
    return result.str();
}

std::vector<std::vector<std::uint8_t>> PriorBoxClusteredLayerTest::CalculateRefs() {
    size_t numPriors = widths.size();
    const size_t layerWidth = inputShapes[3];
    const size_t layerHeight = inputShapes[2];
    size_t imgWidth = imageShapes[3];
    size_t imgHeight = imageShapes[2];

    if (variances.empty())
        variances.push_back(0.1f);
    size_t varSize = variances.size();

    size_t topDataOffset = 4 * layerWidth * layerHeight * numPriors;
    size_t outSize = 2 * topDataOffset;
    auto outBuf = std::vector<float>(outSize);
    float* topData_0 = outBuf.data();
    float* topData_1 = outBuf.data() + topDataOffset;

    if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
        //GPU inits buffers with 0.0f
        for (auto i = 0; i < outSize; i++)
            topData_0[i] = 0.0f;
    }

    float stepW = step_width;
    float stepH = step_height;
    if (stepW == 0 && stepH == 0) {
        stepW = static_cast<float>(imgWidth) / layerWidth;
        stepH = static_cast<float>(imgHeight) / layerHeight;
    }

    for (size_t h = 0; h < layerHeight; ++h) {
        for (size_t w = 0; w < layerWidth; ++w) {
            float center_x = (w + offset) * stepW;
            float center_y = (h + offset) * stepH;

            for (size_t s = 0; s < numPriors; ++s) {
                float box_width = widths[s];
                float box_height = heights[s];

                float xmin = (center_x - box_width / 2.0f) / imgWidth;
                float ymin = (center_y - box_height / 2.0f) / imgHeight;
                float xmax = (center_x + box_width / 2.0f) / imgWidth;
                float ymax = (center_y + box_height / 2.0f) / imgHeight;

                if (clip) {
                    xmin = (std::min)((std::max)(xmin, 0.0f), 1.0f);
                    ymin = (std::min)((std::max)(ymin, 0.0f), 1.0f);
                    xmax = (std::min)((std::max)(xmax, 0.0f), 1.0f);
                    ymax = (std::min)((std::max)(ymax, 0.0f), 1.0f);
                }

                topData_0[h * layerWidth * numPriors * 4 + w * numPriors * 4 + s * 4 + 0] = xmin;
                topData_0[h * layerWidth * numPriors * 4 + w * numPriors * 4 + s * 4 + 1] = ymin;
                topData_0[h * layerWidth * numPriors * 4 + w * numPriors * 4 + s * 4 + 2] = xmax;
                topData_0[h * layerWidth * numPriors * 4 + w * numPriors * 4 + s * 4 + 3] = ymax;

                for (int j = 0; j < varSize; j++)
                    topData_1[h * layerWidth * numPriors * varSize + w * numPriors * varSize +
                    s * varSize +
                    j] = variances[j];
            }
        }
    }

    // Be aligned with test utils ref calulcation method, which returns std::vector<std::vector<uint8_t>>...
    std::vector<std::vector<uint8_t>> ret(1);
    for (auto& val : outBuf) {
        uint8_t* u8_val = reinterpret_cast<uint8_t*>(&val);
        ret[0].push_back(u8_val[0]);
        ret[0].push_back(u8_val[1]);
        ret[0].push_back(u8_val[2]);
        ret[0].push_back(u8_val[3]);
    }

    return ret;
}

void PriorBoxClusteredLayerTest::SetUp() {
    priorBoxClusteredSpecificParams specParams;
    std::tie(specParams, netPrecision,
        inPrc, outPrc, inLayout, outLayout,
        inputShapes, imageShapes, targetDevice) = GetParam();

    std::tie(widths,
        heights,
        clip,
        step_width,
        step_height,
        offset,
        variances) = specParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, { inputShapes, imageShapes });
    auto paramsOut = ngraph::helpers::convert2OutputVector(
        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));

    ngraph::op::PriorBoxClusteredAttrs attributes;
    attributes.widths = widths;
    attributes.heights = heights;
    attributes.clip = clip;
    attributes.step_widths = step_width;
    attributes.step_heights = step_height;
    attributes.offset = offset;
    attributes.variances = variances;

    auto priorBoxClustered = std::make_shared<ngraph::op::PriorBoxClusteredIE>(
        paramsOut[0],
        paramsOut[1],
        attributes);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(priorBoxClustered) };
    function = std::make_shared<ngraph::Function>(results, paramsIn, "PB_Clustered");
}

TEST_P(PriorBoxClusteredLayerTest, CompareWithRefs) {
    Run();
};
}  // namespace LayerTestsDefinitions
