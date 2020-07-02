// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <unordered_set>

#include <gtest/gtest.h>
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "ie_precision.hpp"
#include <tests_common.hpp>
#include <tests_common_func.hpp>
#include <multi-device/multi_device_config.hpp>
#include "low_precision_transformations/transformer.hpp"
#include <regression_tests.hpp>
#include "common/validation.hpp"
#include "low_precision_transformations/concat_multi_channels.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/fully_connected.hpp"
#include "low_precision_transformations/eltwise.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"
#include "ie_util_internal.hpp"

#include "cnn_network_ngraph_impl.hpp"
#include <ie_system_conf.h>

using namespace ::testing;
using namespace InferenceEngine;

inline CNNLayerPtr getLayer(const ICNNNetwork& network, const std::string& layerName) {
    std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
    for (CNNLayerPtr layer : layers) {
        if (layer->name == layerName) {
            return layer;
        }
    }

    return nullptr;
}

inline void checkLayerOuputPrecision(const ICNNNetwork& network, const std::string& layerName, Precision expectedPrecision) {
    CNNLayerPtr layer = getLayer(network, layerName);
    for (DataPtr data : layer->outData) {
        ASSERT_EQ(expectedPrecision, data->getPrecision()) << " unexpected precision " << data->getPrecision() << " for layer " << layerName;
    }
}

struct network_params {
    std::string deviceName;
    std::string modelFile;
    std::string imageName;
    std::vector<std::pair<int, float>> refValue;
    // optional config (used for multi-device)
    std::map<std::string, std::string> config;

    std::string model() {
        ModelsPath result;
        result += kPathSeparator;
        result += modelFile;
        return result;
    }

    std::string weights() {
        ModelsPath result;
        result += kPathSeparator;
        result += testing::FileUtils::fileNameNoExt(modelFile);
        result += ".bin";
        return result;
    }

    std::string image() {
        std::string result = TestDataHelpers::get_data_path();
        result += kPathSeparator;
        result += imageName;
        return result;
    }
};

static LayerTransformation::Params createParam() {
    return LayerTransformation::Params(
        false,
        true,
        true,
        LayerTransformation::QuantizedTensorAlignment::None,
        LayerTransformation::QuantizedTensorAlignment::None,
        false);
}

static LayerTransformation::Params createParamU8I8() {
    return LayerTransformation::Params(
        false,
        true,
        true,
        LayerTransformation::QuantizedTensorAlignment::None,
        LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { Precision::U8 },
        { Precision::I8 });
}

static LayerTransformation::Params createParamU8U8() {
    return LayerTransformation::Params(
        false,
        true,
        true,
        LayerTransformation::QuantizedTensorAlignment::None,
        LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { Precision::U8 },
        { Precision::U8 });
}

static LayerTransformation::Params createParamI8I8() {
    return LayerTransformation::Params(
        false,
        true,
        true,
        LayerTransformation::QuantizedTensorAlignment::None,
        LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { Precision::I8 },
        { Precision::I8 });
}

static LayerTransformation::Params createParamCpu() {
    return LayerTransformation::Params(
        true,
        true,
        true,
        LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        LayerTransformation::QuantizedTensorAlignment::None,
        true,
        true,
        true);
}

static std::vector<float> generateInput(const size_t size, const bool reverse = false) {
    std::vector<float> in(size);
    for (size_t i = 0; i < in.size(); ++i) {
        in[i] = reverse ? in.size() - i : i;
    }
    return in;
}


class TransformationsParams;

class ModelParams {
public:
    ModelParams(
            const std::string name,
            const std::string irFilePath,
            const std::string dataFilePath,
            const std::vector<std::pair<int, float>> referenceOutputDataWithoutTransformations,
            const std::vector<std::pair<int, float>> referenceOutputDataWithTransformations = {}) :
            name(name),
            irFilePath(irFilePath),
            dataFilePath(dataFilePath),
            referenceOutputDataWithoutTransformations({ referenceOutputDataWithoutTransformations }),
            referenceOutputDataWithTransformations((referenceOutputDataWithTransformations.size() != 0ul) ?
                                                   std::vector<std::vector<std::pair<int, float>>>({ referenceOutputDataWithTransformations }) :
                                                   std::vector<std::vector<std::pair<int, float>>>({ referenceOutputDataWithoutTransformations })),
            validation(nullptr),
            inputs({}),
            transformations({}) {}


    ModelParams(
            const std::string name,
            const std::string irFilePath,
            const std::string dataFilePath,
            const std::vector<std::pair<int, float>> referenceOutputDataWithoutTransformations,
            const std::vector<std::pair<int, float>> referenceOutputDataWithTransformations,
            std::function<void(const TransformationsParams& params, CNNNetworkImplPtr usedNetwork)> validation,
            const std::vector<std::pair<std::string, std::vector<float>>> inputs = {},
            const std::vector<std::pair<std::string, std::shared_ptr<LayerTransformation>>> transformations = {}) :
            name(name),
            irFilePath(irFilePath),
            dataFilePath(dataFilePath),
            referenceOutputDataWithoutTransformations({ referenceOutputDataWithoutTransformations }),
            referenceOutputDataWithTransformations(referenceOutputDataWithTransformations.size() != 0ul ?
                                                   std::vector<std::vector<std::pair<int, float>>>({ referenceOutputDataWithTransformations }) :
                                                   std::vector<std::vector<std::pair<int, float>>>({ referenceOutputDataWithoutTransformations })),
            validation(validation),
            inputs(inputs),
            transformations(transformations) {}

    ModelParams(
            const std::string name,
            const std::string irFilePath,
            const std::string dataFilePath,
            const std::vector<std::vector<std::pair<int, float>>> referenceOutputDataWithoutTransformations,
            const std::vector<std::vector<std::pair<int, float>>> referenceOutputDataWithTransformations,
            std::function<void(const TransformationsParams& params, CNNNetworkImplPtr usedNetwork)> validation) :
            name(name),
            irFilePath(irFilePath),
            dataFilePath(dataFilePath),
            referenceOutputDataWithoutTransformations(referenceOutputDataWithoutTransformations),
            referenceOutputDataWithTransformations(referenceOutputDataWithTransformations.size() != 0ul ? referenceOutputDataWithTransformations : referenceOutputDataWithoutTransformations),
            validation(validation),
            inputs({}),
            transformations({}) {}

    const std::string name;
    const std::string irFilePath;
    const std::string dataFilePath;
    const std::vector<std::vector<std::pair<int, float>>> referenceOutputDataWithoutTransformations;
    const std::vector<std::vector<std::pair<int, float>>> referenceOutputDataWithTransformations;
    const std::function<void(const TransformationsParams& params, CNNNetworkImplPtr usedNetwork)> validation;
    const std::vector<std::pair<std::string, std::vector<float>>> inputs;
    const std::vector<std::pair<std::string, std::shared_ptr<LayerTransformation>>> transformations;
};

class TransformationsParams {
public:
    TransformationsParams(
            const bool transformationsInPluginEnabled = true,
            const bool transformationsInTestEnabled = false,
            const LayerTransformation::Params& params = LayerTransformation::Params(),
            const std::unordered_set<std::string>& notTransformedLayers = {},
            const size_t classesCanBeChangedIndex = 9999,
            const bool compareRawValues = true,
            const std::unordered_set<std::string>& removedLayers = {}) :
            deviceName(""),
            modelParams(ModelParams("", "", "", {})),
            batchSize(1ul),
            transformationsInPluginEnabled(transformationsInPluginEnabled),
            transformationsInTestEnabled(transformationsInTestEnabled),
            params(params),
            notTransformedLayers(notTransformedLayers),
            classesCanBeChangedIndex(classesCanBeChangedIndex),
            compareRawValues(compareRawValues),
            removedLayers(removedLayers) {}

    TransformationsParams(
            const std::string deviceName,
            const ModelParams modelParams,
            const size_t batchSize,
            const bool transformationsInPluginEnabled = true,
            const bool transformationsInTestEnabled = false,
            const LayerTransformation::Params& params = LayerTransformation::Params(),
            const std::unordered_set<std::string>& notTransformedLayers = {},
            const size_t classesCanBeChangedIndex = 9999,
            const bool compareRawValues = true,
            const std::unordered_set<std::string>& removedLayers = {},
            const std::vector<std::pair<std::string, std::vector<float>>> inputs = {},
            const std::vector<std::pair<std::string, std::shared_ptr<LayerTransformation>>> transformations = {}) :
            deviceName(deviceName),
            modelParams(modelParams),
            batchSize(batchSize),
            transformationsInPluginEnabled(transformationsInPluginEnabled),
            transformationsInTestEnabled(transformationsInTestEnabled),
            params(params),
            notTransformedLayers(notTransformedLayers),
            classesCanBeChangedIndex(classesCanBeChangedIndex),
            compareRawValues(compareRawValues),
            removedLayers(removedLayers) {}

    const std::string deviceName;
    const ModelParams modelParams;
    const size_t batchSize;

    static std::string getLowPrecisionTransformerSingleLayerTestName(testing::TestParamInfo<TransformationsParams> params) {
        const TransformationsParams& p = params.param;
        std::stringstream ss;
        ss << p.modelParams.name <<
           "_batch" << p.batchSize <<
           "_" << (p.transformationsInPluginEnabled ? "inPluginEnabled" : "inPluginDisabled") <<
           "_" << (p.transformationsInTestEnabled ? "inTestEnabled" : "inTestDisabled") <<
           "_" << (p.params.supportAsymmetricQuantization ? "asymmetric" : "symmetric") <<
           "_" << p.params.precisionsOnActivations <<
           "_" << p.params.precisionsOnWeights <<
           "_" << p.params.quantizedTensorAlignmentOnActivations;
        return ss.str();
    }

    const bool transformationsInPluginEnabled;
    const bool transformationsInTestEnabled;
    const LayerTransformation::Params params;
    const std::unordered_set<std::string> notTransformedLayers;
    const size_t classesCanBeChangedIndex;
    const bool compareRawValues;
    const std::unordered_set<std::string> removedLayers;
};

class smoke_NetworkClassifyTest : public TestsCommon, public TestsCommonFunc, public WithParamInterface<TransformationsParams> {
protected:
    void classify(
            network_params p,
            size_t batch_size = 1,
            float threshold = 0.005f,
            const TransformationsParams& transformationsParams = TransformationsParams(),
            const std::vector<std::pair<std::string, std::vector<float>>>& inputs = {},
            const std::vector<std::pair<std::string, std::shared_ptr<LayerTransformation>>>& transformations = {}) {
        CNNNetworkImplPtr usedNetwork;
        classify(p, batch_size, threshold, transformationsParams, usedNetwork, inputs, transformations);
    }

    void classify(
            network_params p,
            size_t batch_size,
            float threshold,
            const TransformationsParams& transformationsParams,
            CNNNetworkImplPtr& usedNetwork,
            const std::vector<std::pair<std::string, std::vector<float>>>& inputs = {},
            const std::vector<std::pair<std::string, std::shared_ptr<LayerTransformation>>>& transformations = {}) {

#ifdef DISPLAY_RESULTS
        std::cout << std::endl << p.modelFile << ": was started" << std::endl;
        if (transformationsParams.transformationsInTestEnabled) {
            std::cout <<
                "\tenabled: " << (transformationsParams.transformationsInTestEnabled ? "true" : "false") << std::endl <<
                "\tbatch_size: " << batch_size << std::endl <<
                "\tupdatePrecision: " << (transformationsParams.params.updatePrecisions ? "true" : "false") << std::endl <<
                "\tquantizeOutputs: " << (transformationsParams.params.quantizeOutputs ? "true" : "false") << std::endl <<
                "\tweightsToConst: " << (transformationsParams.params.weightsToConst ? "true" : "false") << std::endl <<
                "\tquantizedTensorAlignmentOnActivations: " << transformationsParams.params.quantizedTensorAlignmentOnActivations << std::endl <<
                "\tquantizedTensorAlignmentOnWeights: " << transformationsParams.params.quantizedTensorAlignmentOnWeights << std::endl <<
                "\troundQuantizedValues: " << (transformationsParams.params.roundQuantizedValues ? "true" : "false") << std::endl <<
                "\tupdateBiases: " << (transformationsParams.params.updateBiases ? "true" : "false") << std::endl <<
                "\tsupportAsymmetricQuantization: " << (transformationsParams.params.supportAsymmetricQuantization ? "true" : "false") << std::endl <<
                "\tprecisionsOnActivations: " << transformationsParams.params.precisionsOnActivations << std::endl <<
                "\tprecisionsOnWeights: " << transformationsParams.params.precisionsOnWeights << std::endl;
        } else {
            std::cout << "\tenabled: " << (transformationsParams.transformationsInTestEnabled ? "true" : "false") << std::endl;
        }
#endif

        Core ie;
        CNNNetwork network;
        if (*p.modelFile.begin() == '/') {
            network = ie.ReadNetwork(p.modelFile);
        } else {
            network = ie.ReadNetwork(p.model(), p.weights());
        }

        if (batch_size != 1)
            network.setBatchSize(batch_size);

        ie.SetConfig(p.config);

        if (transformationsParams.transformationsInTestEnabled) {
            ICNNNetwork& icnnnetwork = network;
            auto networkNGraph = dynamic_cast<CNNNetworkNGraphImpl*>(&icnnnetwork);
            if (networkNGraph) {
                std::shared_ptr<ICNNNetwork> networkPtr = networkNGraph->getCNNNetwork();
                network = CNNNetwork(networkPtr);
            }

            auto originalLayersInfo = LowPrecisionTransformationValidation::getLayers(network);
            for (const std::string removedLayer : transformationsParams.removedLayers) {
                for (auto originalLayerIt = originalLayersInfo.begin(); originalLayerIt != originalLayersInfo.end(); ++originalLayerIt) {
                    if (removedLayer == originalLayerIt->first) {
                        originalLayersInfo.erase(originalLayerIt);
                        break;
                    }
                }
            }

            LowPrecisionTransformations lowPrecisionTransformations = LowPrecisionTransformer::getAllTransformations(transformationsParams.params).
                    addBranchSpecific<EltwiseTransformation>(LayerTransformation::Params(transformationsParams.params), "Eltwise").
                    add<ConvolutionTransformation>(
                    LayerTransformation::Params(transformationsParams.params).setPrecisionsOnActivations({ Precision::U8 }),
                    "Convolution").
                    addCleanup<ScaleShiftToConvolutionTransformation>(
                    LayerTransformation::Params(transformationsParams.params).setPrecisionsOnActivations({ Precision::U8 }),
                    "ScaleShift");

            for (const auto transformation : transformations) {
                auto it = lowPrecisionTransformations.transformations.find(transformation.first);
                if (it != lowPrecisionTransformations.transformations.end()) {
                    lowPrecisionTransformations.transformations.erase(it);
                }

                lowPrecisionTransformations.transformations.emplace(transformation.first, transformation.second);
            }

            LowPrecisionTransformer transformer(lowPrecisionTransformations);
            transformer.transform(network);

            LowPrecisionTransformationValidation::validate(
                    network,
                    transformationsParams.params,
                    transformationsParams.notTransformedLayers,
                    originalLayersInfo);
        }

        std::map<std::string, std::string> config;
        if (!transformationsParams.transformationsInPluginEnabled) {
            config.emplace(PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE, PluginConfigParams::NO);
        }

        // use to enable LPT ON devices with explicit KEY_LP_TRANSFORMS_MODE definition (GPU)
        //config.emplace(
        //    PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE,
        //    transformationsParams.transformationsInPluginEnabled ? PluginConfigParams::YES : PluginConfigParams::NO);

        usedNetwork = cloneNet(network);
        ExecutableNetwork exeNetwork = ie.LoadNetwork(network, p.deviceName, config);
        InferRequest inferRequest = exeNetwork.CreateInferRequest();
        if (inputs.empty()) {
            Blob::Ptr src = readInput(p.image(), batch_size);
            ASSERT_NE(nullptr, src.get()) << "Cannot read Input " << p.image();
            auto inputsInfo = network.getInputsInfo();
            if (inputsInfo.size() == 3ul) {
                std::vector<float> data = { 1.f, 2.f, 3.f };
                Blob::Ptr blob = make_shared_blob<float>(TensorDesc(Precision::FP32, { 1ul, 3ul }, Layout::NC));
                blob->allocate();
                CNNNetworkHelper::fillBlobByFP32(blob, data.data());

                auto it = inputsInfo.begin();
                inferRequest.SetBlob(it->first, blob);

                ++it;
                inferRequest.SetBlob(it->first, src);

                ++it;
                inferRequest.SetBlob(it->first, src);
            } else {
                inferRequest.SetBlob(network.getInputsInfo().begin()->first, src);
            }
        } else {
            for (const auto input : inputs) {
                Blob::Ptr blob = make_shared_blob<float>(TensorDesc(Precision::FP32, { input.second.size() }, Layout::C));
                blob->allocate();
                CNNNetworkHelper::fillBlobByFP32(blob, input.second.data());
                inferRequest.SetBlob(input.first, blob);
            }
        }

        OutputsDataMap outInfo;
        outInfo = network.getOutputsInfo();
        ASSERT_EQ(outInfo.size(), 1);
        ASSERT_NE(outInfo.begin()->second, nullptr);
        Blob::Ptr dst = make_shared_blob<float>(outInfo.begin()->second->getTensorDesc());
        dst->allocate();
        inferRequest.SetBlob(outInfo.begin()->first, dst);

        inferRequest.Infer();

        for (size_t i = 0; i < batch_size; i++)
            ASSERT_TRUE(compareTop(*dst.get(), p.refValue, i, threshold, transformationsParams.classesCanBeChangedIndex, transformationsParams.compareRawValues)) << "Doesn't match with ref values";
    }

    Regression::Builder please() {
        std::shared_ptr<Core> ie = PluginCache::get().ie();
        Regression::Builder b(ie);
        b.usingDevice("CPU");

        return b;
    }

private:
    static bool onWeights(const CNNLayer& layer) {
        const std::vector<CNNLayerPtr> children = getChildren(layer);
        return (children.size() == 1) &&
               (children[0]->type == "Convolution") &&
               (children[0]->insData.size() >= 2) &&
               (children[0]->insData[1].lock()->getCreatorLayer().lock()->name == layer.name);
    }

    static std::vector<CNNLayerPtr> getChildren(const CNNLayer& layer, const std::string& exceptionLayerName = "") {
        std::vector<CNNLayerPtr> children;
        for (const DataPtr outData : layer.outData) {
            const std::map<std::string, CNNLayerPtr>& inputTo = outData->getInputTo();
            for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {
                CNNLayerPtr child = it->second;
                if (exceptionLayerName.empty() || child->name != exceptionLayerName) {
                    children.push_back(child);
                }
            }
        }
        return children;
    }
};

class ModelTransformationsTest : public smoke_NetworkClassifyTest {
protected:
    void SetUp() override {
        const TransformationsParams transformationsParam = ::testing::WithParamInterface<TransformationsParams>::GetParam();
        CNNNetworkImplPtr usedNetwork;

        std::vector<std::pair<int, float>> referenceValues;
        if (transformationsParam.params.updatePrecisions &&
            (transformationsParam.transformationsInPluginEnabled || transformationsParam.transformationsInTestEnabled)) {
            if (transformationsParam.modelParams.referenceOutputDataWithTransformations.size() == 1) {
                referenceValues = transformationsParam.modelParams.referenceOutputDataWithTransformations[0];
            } else {
                referenceValues = InferenceEngine::with_cpu_x86_avx512f() ?
                                  transformationsParam.modelParams.referenceOutputDataWithTransformations[1] :
                                  transformationsParam.modelParams.referenceOutputDataWithTransformations[0];
            }
        } else {
            if (transformationsParam.modelParams.referenceOutputDataWithoutTransformations.size() == 1) {
                referenceValues = transformationsParam.modelParams.referenceOutputDataWithoutTransformations[0];
            } else {
                referenceValues = InferenceEngine::with_cpu_x86_avx512f() ?
                                  transformationsParam.modelParams.referenceOutputDataWithoutTransformations[1] :
                                  transformationsParam.modelParams.referenceOutputDataWithoutTransformations[0];
            }
        }

        network_params p{
                "CPU",
                transformationsParam.modelParams.irFilePath,
                transformationsParam.modelParams.dataFilePath,
                referenceValues
        };

        classify(p,
                 transformationsParam.batchSize,
                 1.f,
                 transformationsParam,
                 usedNetwork,
                 transformationsParam.modelParams.inputs,
                 transformationsParam.modelParams.transformations);

        if (transformationsParam.modelParams.validation != nullptr) {
            transformationsParam.modelParams.validation(transformationsParam, usedNetwork);
        }
    }
};
