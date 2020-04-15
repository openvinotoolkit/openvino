// Copyright (C) 2019 Intel Corporation
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
#include <ie_plugin_config.hpp>
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
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inputPrecision;
    InferenceEngine::Precision outputPrecision;
    InferenceEngine::Layout inputLayout;
    InferenceEngine::Layout outputLayout;
    std::string targetDevice;
    std::shared_ptr<ngraph::Function> fnPtr;
    std::map<std::string, std::string> config;

    LayerTestsCommonClass() {
        netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        inputPrecision = InferenceEngine::Precision::UNSPECIFIED;
        outputPrecision = InferenceEngine::Precision::UNSPECIFIED;
        inputLayout = InferenceEngine::Layout::ANY;
        outputLayout = InferenceEngine::Layout::ANY;
    }

    void inline inferAndValidate() {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        // Create CNNNetwork from ngrpah::Function
        InferenceEngine::CNNNetwork cnnNet(fnPtr);
        // Set target input/output Precisions for the network
        setNetInOutPrecision(cnnNet, inputPrecision, outputPrecision);
        // Set target input Layouts for the network
        setNetInOutLayout(cnnNet, inputLayout, outputLayout);

        // Get Core from cache
        auto ie = PluginCache::get().ie();
        // Load config
        if (!config.empty()) {
            ie->SetConfig(config, targetDevice);
        }
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
        std::vector<InferenceEngine::Blob::Ptr> castedBlobs;
        for (size_t i = 0; i < inBlobs.size(); i++) {
            const auto precision = inBlobs[i]->getTensorDesc().getPrecision();
            const auto layout = inBlobs[i]->getTensorDesc().getLayout();
            const auto defLayout = InferenceEngine::TensorDesc::getLayoutByDims(inBlobs[i]->getTensorDesc().getDims());

            if (precision == InferenceEngine::Precision::FP32 && layout == defLayout) {
                inRawData.push_back(inBlobs[i]->cbuffer().template as<const float*>());
            } else {
                auto castedBlob = FuncTestUtils::copyBlobWithCast<InferenceEngine::Precision::FP32>(inBlobs[i]);
                castedBlob = FuncTestUtils::convertBlobLayout(castedBlob, defLayout);
                inRawData.push_back(castedBlob->cbuffer().template as<const float*>());
                castedBlobs.push_back(castedBlob);
            }
        }
        // Run inference in IE
        req.Infer();
        // Reset PluginCash
        if (!config.empty()) {
            PluginCache::get().reset();
        }
        // Get output raw data from resulting output blobs
        std::vector<float *> outBlobsRawData;
        std::vector<size_t> outElementsCount;  // output elements count required for compareRawBuffers()
        for (const auto &output : cnnNet.getOutputsInfo()) {
            auto currentBlob = req.GetBlob(output.first);

            outElementsCount.push_back(
                std::accumulate(
                    std::begin(output.second->getDims()), std::end(output.second->getDims()),
                    size_t {1}, std::multiplies<size_t>()));

            const auto precision = currentBlob->getTensorDesc().getPrecision();
            const auto layout = currentBlob->getTensorDesc().getLayout();
            const auto defLayout = InferenceEngine::TensorDesc::getLayoutByDims(currentBlob->getTensorDesc().getDims());

            if (precision == InferenceEngine::Precision::FP32 && layout == defLayout) {
                outBlobsRawData.push_back(currentBlob->cbuffer().template as<float*>());
            } else {
                auto castedBlob = FuncTestUtils::copyBlobWithCast<InferenceEngine::Precision::FP32>(currentBlob);
                castedBlob = FuncTestUtils::convertBlobLayout(castedBlob, defLayout);
                outBlobsRawData.push_back(castedBlob->cbuffer().template as<float*>());
                castedBlobs.push_back(castedBlob);
            }
        }

        // Convert initial ngraph::Function to fp32 for references calculation
        convertFuncToF32(fnPtr, netPrecision);
        // Run ngraph Interpreter backend to calculate references
        auto refOutData = ngraph::helpers::inferFnWithInterp<ngraph::element::Type_t::f32>(fnPtr, inRawData);
        // Compare IE infer results vs ngraph Interpreter reference results
        auto thr = FuncTestUtils::GetComparisonThreshold(netPrecision);
        FuncTestUtils::compareRawBuffers(outBlobsRawData, refOutData, outElementsCount, outElementsCount, thr);

        // Deallocate ngraph::Function pointer
        fnPtr.reset();
        if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
            PluginCache::get().reset();
        }
    }

protected:
    static void setNetInOutPrecision(InferenceEngine::CNNNetwork &cnnNet, InferenceEngine::Precision inPrc,
                              InferenceEngine::Precision outPrc = InferenceEngine::Precision::UNSPECIFIED) {
        if (inPrc != InferenceEngine::Precision::UNSPECIFIED) {
            for (const auto &inputItem : cnnNet.getInputsInfo()) {
                inputItem.second->setPrecision(inPrc);
            }
        }
        if (outPrc != InferenceEngine::Precision::UNSPECIFIED) {
            for (const auto &output : cnnNet.getOutputsInfo()) {
                output.second->setPrecision(outPrc);
            }
        }
    }

    static void setNetInOutLayout(InferenceEngine::CNNNetwork& cnnNet, InferenceEngine::Layout inputLayout,
                                  InferenceEngine::Layout outputLayout = InferenceEngine::Layout::ANY) {
        if (inputLayout != InferenceEngine::Layout::ANY) {
            for (const auto& inputItem : cnnNet.getInputsInfo()) {
                inputItem.second->setLayout(inputLayout);
            }
        }
        if (outputLayout != InferenceEngine::Layout::ANY) {
            for (const auto& output : cnnNet.getOutputsInfo()) {
                output.second->setLayout(outputLayout);
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

using TargetDevice = std::string;

class FuncTestsCommon : public CommonTestUtils::TestsCommon {
public:
    virtual InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const;
    virtual void Run();
    virtual void Compare(const std::vector<std::uint8_t>& expected, const InferenceEngine::Blob::Ptr& actual);

protected:
     FuncTestsCommon();
    ~FuncTestsCommon() override;

    template<class T>
    void Compare(const T* expected, const T* actual, std::size_t size, T threshold) {
        for (std::size_t i = 0; i < size; ++i) {
            const auto& ref = expected[i];
            const auto& res = actual[i];

            const auto absoluteDifference = std::abs(res - ref);
            if (absoluteDifference <= threshold) {
                continue;
            }

            const auto max = std::max(std::abs(res), std::abs(ref));
            ASSERT_TRUE(max != 0 && ((absoluteDifference / max) <= threshold))
                << "Relative comparison of values expected: " << ref << " and actual: " << res << " at index " << i << " with threshold " << threshold
                << " failed";
        }
    }

    TargetDevice targetDevice;
    std::shared_ptr<ngraph::Function> function;
    std::map<std::string, std::string> configuration;

private:
    void Configure() const;
    void LoadNetwork();
    void Infer();
    std::vector<InferenceEngine::Blob::Ptr> GetOutputs();
    void Validate();

    InferenceEngine::Core* core = nullptr;
    InferenceEngine::CNNNetwork cnnNetwork;
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest inferRequest;
    std::vector<InferenceEngine::Blob::Ptr> inputs;
};

}  // namespace LayerTestsUtils
