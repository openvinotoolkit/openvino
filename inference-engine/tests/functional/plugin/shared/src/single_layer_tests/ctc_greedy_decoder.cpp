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

#include "single_layer_tests/ctc_greedy_decoder.hpp"

namespace LayerTestsDefinitions {
std::string CTCGreedyDecoderLayerTest::getTestCaseName(
    const testing::TestParamInfo<ctcGreedyDecoderParams>& obj) {
    InferenceEngine::Precision inputPrecision, netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    bool mergeRepeated;
    std::tie(netPrecision,
        inputShapes,
        mergeRepeated,
        targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';

    result << "IS="     << CommonTestUtils::vec2str(inputShapes) << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "merge_repeated=" << std::boolalpha << mergeRepeated << separator;
    result << "targetDevice=" << targetDevice;

    return result.str();
}

std::vector<std::vector<std::uint8_t>> CTCGreedyDecoderLayerTest::CalculateRefs() {
    std::vector<const float *> inRawData;
    std::vector<InferenceEngine::Blob::Ptr> castedBlobs;
    for (size_t i = 0; i < inputs.size(); i++) {
        const auto precision = inputs[i]->getTensorDesc().getPrecision();
        const auto layout = inputs[i]->getTensorDesc().getLayout();
        const auto defLayout = InferenceEngine::TensorDesc::getLayoutByDims(inputs[i]->getTensorDesc().getDims());

        if (precision == InferenceEngine::Precision::FP32 && layout == defLayout) {
            inRawData.push_back(inputs[i]->cbuffer().template as<const float*>());
        } else {
            auto castedBlob = FuncTestUtils::copyBlobWithCast<InferenceEngine::Precision::FP32>(inputs[i]);
            castedBlob = FuncTestUtils::convertBlobLayout(castedBlob, defLayout);
            inRawData.push_back(castedBlob->cbuffer().template as<const float*>());
            castedBlobs.push_back(castedBlob);
        }
    }

    size_t T_ = inputShapes.at(0);
    size_t N_ = inputShapes.at(1);
    size_t C_ = inputShapes.at(2);
    auto outSize = T_ * N_;
    float a = 43;
    const float* probabilities = inRawData[0];
    const float* sequence_indicators = inRawData[1];
    auto outBuf = std::vector<float>(outSize);
    float* output_sequences = outBuf.data();

    for (auto i = 0; i < outSize; i++)
        output_sequences[i] = -1.0f;

    for (size_t n = 0; n < N_; ++n) {
        int prev_class_idx = -1;
        size_t output_index = n * T_;

        for (size_t t = 0; /* check at end */; ++t) {
            // get maximum probability and its index
            int max_class_idx = 0;

            const float* probs = probabilities + t * C_ * N_ + n * C_;
            float max_prob = probs[0];
            ++probs;

            for (size_t c = 1; c < C_; ++c, ++probs) {
                if (*probs > max_prob) {
                    max_class_idx = static_cast<int>(c);
                    max_prob = *probs;
                }
            }

            if (max_class_idx < static_cast<int>(C_) - 1 &&
                max_class_idx != prev_class_idx) {
                output_sequences[output_index] = static_cast<float>(max_class_idx);
                output_index++;
            }

            prev_class_idx = max_class_idx;

            if (t + 1 == T_ || sequence_indicators[(t + 1) * N_ + n] == 0) {
                break;
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

void CTCGreedyDecoderLayerTest::SetUp() {
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(netPrecision, inputShapes, mergeRepeated, targetDevice) = GetParam();
    sequenceLengths = { inputShapes.at(0), inputShapes.at(1) };
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, { inputShapes, sequenceLengths });
    auto paramsOut = ngraph::helpers::convert2OutputVector(
        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    auto ctcGreedyDecoder = std::make_shared<ngraph::opset1::CTCGreedyDecoder>(
        paramsOut[0],
        paramsOut[1],
        mergeRepeated);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(ctcGreedyDecoder) };
    function = std::make_shared<ngraph::Function>(results, paramsIn, "Grn");
}

TEST_P(CTCGreedyDecoderLayerTest, CompareWithRefs) {
    //std::vector<std::shared_ptr<float*>> CalculateRefs();
    Run();
};
}  // namespace LayerTestsDefinitions
