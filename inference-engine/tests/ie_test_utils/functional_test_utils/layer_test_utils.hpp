// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <typeindex>
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <gtest/gtest.h>
#include <ngraph/node.hpp>
#include <ngraph/function.hpp>

#include <ngraph/function.hpp>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"

#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"


namespace LayerTestsUtils {
typedef std::tuple<
        InferenceEngine::Precision,  // Input Precision
        InferenceEngine::Precision,  // Network Precision
        InferenceEngine::SizeVector, // Input Shape
        std::string                  // Target Device
> basicParams;

template<typename paramType>
class LayerTestsCommonClass : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<paramType> {
public:
    InferenceEngine::Precision netPrecision, inputPrecision;
    std::string targetDevice;
    std::shared_ptr<ngraph::Function> fnPtr;

    void inline inferAndValidate() {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(fnPtr);
        // Set target input Precisions for the network
        setNetInOutPrecision(cnnNet, inputPrecision);

        // Get Core from cache
        auto ie = PluginCache::get().ie();
        // Load CNNNetwork to target plugins
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
        // Create InferRequest
        auto req = execNet.CreateInferRequest();

        // Create and set input blobs
        std::vector<InferenceEngine::Blob::Ptr> inBlobs;
        for (const auto &inputItem : cnnNet.getInputsInfo()) {
            auto currentBlob = FuncTestUtils::createAndFillBlob(inputItem.second->getTensorDesc());
            req.SetBlob(inputItem.first, currentBlob);
            inBlobs.push_back(currentBlob);
        }

        // Create input vector with raw data for reference calculation
        std::vector<const float *> inRawData;
        // References are calculated in float precision, so blobs have to be copied and casted if required
        std::vector<InferenceEngine::Blob::Ptr> castedBlobs = inBlobs;
        for (size_t i = 0; i < castedBlobs.size(); i++) {
            if (inputPrecision != InferenceEngine::Precision::FP32) {
                castedBlobs[i] = FuncTestUtils::copyBlobWithCast<InferenceEngine::Precision::FP32>(inBlobs[i]);
            }
            inRawData.push_back(castedBlobs[i]->cbuffer().as<float *>());
        }
        // Run inference in IE
        req.Infer();

        // Get output raw data from resulting output blobs
        std::vector<float *> outBlobsRawData;
        std::vector<size_t> outElementsCount;  // output elements count required for compareRawBuffers()
        for (const auto &output : cnnNet.getOutputsInfo()) {
            auto currentBlob = req.GetBlob(output.first);
            outBlobsRawData.push_back(currentBlob->cbuffer().template as<float *>());
            outElementsCount.push_back(
                    std::accumulate(begin(output.second->getDims()), end(output.second->getDims()), 1,
                                    std::multiplies<float>()));
        }

        // Convert initial ngraph::Function to fp32 for references calculation
        convertFuncToF32(fnPtr, netPrecision);;
        // Run ngraph Interpreter backend to calculate references
        auto refOutData = ngraph::helpers::inferFnWithInterp<ngraph::element::Type_t::f32>(fnPtr, inRawData);
        // Compare IE infer results vs ngraph Interpreter reference results
        auto thr = FuncTestUtils::GetComparisonThreshold(netPrecision);
        FuncTestUtils::compareRawBuffers(outBlobsRawData, refOutData, outElementsCount, outElementsCount, thr);

        // Deallocate ngraph::Function pointer
        fnPtr.reset();
    }

protected:
    void setNetInOutPrecision(InferenceEngine::CNNNetwork &cnnNet, InferenceEngine::Precision inPrc,
                              InferenceEngine::Precision outPrc = InferenceEngine::Precision::UNSPECIFIED) {
        for (const auto &inputItem : cnnNet.getInputsInfo()) {
            inputItem.second->setPrecision(inPrc);
        }
        if (outPrc != InferenceEngine::Precision::UNSPECIFIED) {
            for (const auto &output : cnnNet.getOutputsInfo()) {
                output.second->setPrecision(outPrc);
            }
        }
    }

    void convertFuncToF32(std::shared_ptr<ngraph::Function> fn, InferenceEngine::Precision prc) {
        switch (prc) {
            case InferenceEngine::Precision::FP32:
                break;
            case InferenceEngine::Precision::FP16:
                ngraph::pass::ConvertPrecision<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32>().run_on_function(
                        fn);
                break;
            case InferenceEngine::Precision::I8:
                ngraph::pass::ConvertPrecision<ngraph::element::Type_t::i8, ngraph::element::Type_t::f32>().run_on_function(
                        fn);
                break;
            case InferenceEngine::Precision::I16:
                ngraph::pass::ConvertPrecision<ngraph::element::Type_t::i16, ngraph::element::Type_t::f32>().run_on_function(
                        fn);
                break;
            case InferenceEngine::Precision::U8:
                ngraph::pass::ConvertPrecision<ngraph::element::Type_t::u8, ngraph::element::Type_t::f32>().run_on_function(
                        fn);
                break;
            case InferenceEngine::Precision::U16:
                ngraph::pass::ConvertPrecision<ngraph::element::Type_t::u16, ngraph::element::Type_t::f32>().run_on_function(
                        fn);
                break;
            default:
                throw std::runtime_error("Precision not handled");
        }
    }
};

template<class opType>
inline std::vector<std::shared_ptr<ngraph::Node>> findTargetNodes(std::shared_ptr<ngraph::Function> fnPtr) {
    std::vector<std::shared_ptr<ngraph::Node>> nodes;
    for (const auto &op : fnPtr->get_ops()) {
        auto convOp = std::dynamic_pointer_cast<opType>(op);
        if (convOp) nodes.push_back(op);
    }
    return nodes;
}

}  // namespace LayerTestsUtils
