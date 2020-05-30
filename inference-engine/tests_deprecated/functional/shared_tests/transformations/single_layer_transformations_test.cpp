// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "common/validation.hpp"
#include "tests_common_func.hpp"
#include <cpp/ie_cnn_net_reader.h>

TBlob<uint8_t>::Ptr SingleLayerTransformationsTest::generateWeights(const CNNNetwork& network) {
    std::vector<Blob::Ptr> blobs;
    const auto net_precision = network.getPrecision();

    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);
    for (CNNLayerPtr layer : sortedLayers) {
        auto weightableLayer = std::dynamic_pointer_cast<WeightableLayer>(layer);
        const std::string& type = layer->type;
        if ((weightableLayer == nullptr) && !CaselessEq<std::string>()(type, "Const")) {
            continue;
        }

        size_t blobSize = 0lu;
        if (CaselessEq<std::string>()(type, "Convolution")) {
            const size_t kernelSize = CNNNetworkHelper::getKernelSize(*layer);
            const size_t inputChannelsCount = CNNNetworkHelper::getInputChannelsCount(*layer);
            const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(*layer);
            blobSize = kernelSize * inputChannelsCount * outputChannelsCount;
        } else if (CaselessEq<std::string>()(type, "Const")) {
            const std::vector<size_t>& dims = layer->outData[0]->getDims();
            blobSize = std::accumulate(dims.begin(), dims.end(), 1lu, std::multiplies<size_t>());
        } else if (CaselessEq<std::string>()(type, "ScaleShift")) {
            blobSize = 2 * layer->outData[0]->getDims()[1]; // weights and biases
        }

        Blob::Ptr weights = CNNNetworkHelper::makeNewBlobPtr({ net_precision, { blobSize }, C });
        weights->allocate();
        fillDataWithInitValue(weights, 1.23f);
        blobs.push_back(weights);

        if (CaselessEq<std::string>()(type, "Convolution")) {
            Blob::Ptr bias = CNNNetworkHelper::makeNewBlobPtr({ net_precision, { CNNNetworkHelper::getOutputChannelsCount(*layer) }, C });
            bias->allocate();
            fillDataWithInitValue(bias, 3.21f);
            blobs.push_back(bias);
        }
    }
    size_t totalSize = 0lu;
    for (auto& blob : blobs) totalSize += (blob->byteSize());

    TBlob<uint8_t>::Ptr modelBlob = make_shared_blob<uint8_t>({ Precision::U8, { totalSize }, C });
    modelBlob->allocate();
    uint8_t* modelBlobBuffer = modelBlob->buffer().as<uint8_t *>();
    for (Blob::Ptr blob : blobs) {
        memcpy(modelBlobBuffer, blob->buffer().as<uint8_t *>(), blob->byteSize());
        modelBlobBuffer += blob->byteSize();
    }

    return modelBlob;
}

// TODO: not completed
void SingleLayerTransformationsTest::checkNetworkWithFakeQuantize(const CNNNetwork& network) {
    size_t total_size_in_bytes = 0;
    std::vector<Blob::Ptr> blob_to_model;

    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);
    for (CNNLayerPtr layer : sortedLayers) {
        if ((layer->type != "Convolution") && (layer->type != "Const")) {
            continue;
        }
    }
}

// TODO: not completed
void SingleLayerTransformationsTest::checkNetworkWithQuantize(const CNNNetwork& network) {
    size_t total_size_in_bytes = 0;
    std::vector<Blob::Ptr> blob_to_model;

    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);
    for (CNNLayerPtr layer : sortedLayers) {
        if ((layer->type != "Convolution") && (layer->type != "Const")) {
            continue;
        }
    }
}

//void SingleLayerTransformationsTest::sortBlobs(CNNLayer& layer) {
//    auto it = layer.blobs.begin();
//    if (it == layer.blobs.end()) {
//        THROW_IE_EXCEPTION << "there is no blobs";
//    }

//    const auto size = it->second->size();
//    const auto byteSize = it->second->byteSize();
//    if ((it->second->size() != 2) || (it->second->byteSize() != 16)) {
//        THROW_IE_EXCEPTION << "not supported - under development";
//    }

//    float* buffer = it->second->buffer().as<float*>();
//    if (buffer[0] > buffer[1]) {
//        const float tmp = buffer[0];
//        buffer[0] = buffer[1];
//        buffer[1] = tmp;
//    }
//}

CNNNetwork SingleLayerTransformationsTest::createNetwork() {
    SingleLayerTransformationsTestParams p = ::testing::WithParamInterface<SingleLayerTransformationsTestParams>::GetParam();
    std::string model = p.model->getModel(p);

    Core reader;
    auto weights_fake = make_shared_blob<uint8_t>(TensorDesc(Precision::U8,
            SizeVector({std::numeric_limits<uint32_t>::max()/2}), Layout::C));
    weights_fake->allocate();
    CNNNetwork network = reader.ReadNetwork(model, weights_fake);

    auto modelBlob = generateWeights(network);
    return reader.ReadNetwork(model, modelBlob);
}

std::unordered_map<std::string, InferenceEngine::Blob::Ptr> SingleLayerTransformationsTest::infer(
        CNNNetwork& network,
        std::unordered_map<std::string, Blob::Ptr>& inputBlobs,
        Core & core,
        const std::string & device_name,
        ExecutableNetwork & executableNetwork,
        InferRequest & inferRequest) {
    const SingleLayerTransformationsTestParams p = ::testing::WithParamInterface<SingleLayerTransformationsTestParams>::GetParam();

    std::map<std::string, std::string> config;
    config.emplace(PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE, PluginConfigParams::NO);
    //config.emplace(PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT, "SingleLayerTransformationsTest");

    executableNetwork = core.LoadNetwork(network, device_name, config);
    inferRequest = executableNetwork.CreateInferRequest();

    for (auto& item : inputBlobs) {
        inferRequest.SetBlob(item.first.c_str(), item.second);
    }

    inferRequest.Infer();

    const std::map<std::string, DataPtr> outputsInfo = network.getOutputsInfo();
    std::unordered_map<std::string, InferenceEngine::Blob::Ptr> outputs_blob_map;
    for (auto& info : outputsInfo) {
        Blob::Ptr output_blob = inferRequest.GetBlob(info.first.c_str());
        outputs_blob_map.insert({info.first, output_blob});
    }

    return outputs_blob_map;
}

void SingleLayerTransformationsTest::compareInDetails(
        InferenceEngine::Blob &res,
        InferenceEngine::Blob &ref,
        const size_t maxDifferenceCounts,
        float max_diff) {
    float *res_ptr = res.buffer().as<float*>();
    size_t res_size = res.size();

    float *ref_ptr = ref.buffer().as<float*>();
    size_t ref_size = ref.size();

    ASSERT_EQ(res_size, ref_size);

    size_t differenceCount = 0;
    std::stringstream log;
    for (size_t i = 0; i < ref_size; i++) {
        const float difference = fabs((res_ptr[i] - ref_ptr[i]) / ref_ptr[i]) * 100.0;
        if ((difference >= max_diff) && (fabs(res_ptr[i] - ref_ptr[i]) > 0.0003)) {
            log << "i=" << i << ": " << res_ptr[i] << " VS " << ref_ptr[i] << ": " << difference << "%, " << fabs(res_ptr[i] - ref_ptr[i]) << std::endl;

            differenceCount++;
            if (differenceCount > maxDifferenceCounts) {
                std::cout << log.str();
                std::cout << differenceCount << " differences are detected" << std::endl;
                ASSERT_TRUE(difference < max_diff);
                break;
            }
        }
    }
}

void SingleLayerTransformationsTest::SetUp() {
    try {
        const SingleLayerTransformationsTestParams p = ::testing::WithParamInterface<SingleLayerTransformationsTestParams>::GetParam();
        // TODO: ONNX enabling
        CNNNetwork network = createNetwork();

        const auto inputsInfo = network.getInputsInfo();
        std::unordered_map<std::string, Blob::Ptr> inputBlobs;
        for (auto& inputInfo : inputsInfo) {
            const TensorDesc& desc = inputInfo.second->getTensorDesc();
            Blob::Ptr input = CNNNetworkHelper::makeNewBlobPtr(desc);
            input->allocate();

            fillData(input, 4.f);
            p.model->initInput(input);

            inputBlobs.insert(std::pair<std::string, Blob::Ptr>(inputInfo.first, input));
        }

        p.model->resetTransformation(network);

        //network.serialize(
        //    p.model->getName() + "_original.xml",
        //    p.model->getName() + "_original.bin");

        Core core;
        ExecutableNetwork executableNetwork;
        InferRequest inferRequest;
        const auto originalOutputMap = infer(network, inputBlobs, core,
                p.device_name, executableNetwork, inferRequest);

        const std::vector<bool> updatePrecisionsValues = { false };
        const std::vector<bool> quantizeOutputsValues = { true, false };
        const std::vector<bool> weightsToConstValues = { true, false };
        const std::vector<LayerTransformation::QuantizedTensorAlignment> quantizedTensorAlignmentOnActivationsValues = {
            LayerTransformation::QuantizedTensorAlignment::None,
            LayerTransformation::QuantizedTensorAlignment::UpdateLevel
        };
        const std::vector<LayerTransformation::QuantizedTensorAlignment> quantizedTensorAlignmentOnWeightsValues = {
            LayerTransformation::QuantizedTensorAlignment::None,
            //LayerTransformation::QuantizedTensorAlignment::Mixed
        };
        const std::vector<bool> roundQuantizedValues = { false, true };
        const std::vector<bool> updateBiasesValues = { true, false };
        const std::vector<bool> supportAsymmetricQuantizationValues = { true /*, false*/ };
        const std::vector<std::vector<Precision>> precisionOnActivationsValues = {
            // TODO: just to debug
            { Precision::I8 },
            { Precision::I8, Precision::U8 },
            { Precision::U8 },
            { Precision::U8, Precision::I8 }
        };
        const std::vector<std::vector<Precision>> precisionOnWeightsValues = { { Precision::I8 } };

        for (const bool updatePrecision : updatePrecisionsValues) {
            for (const bool quantizeOutputs : quantizeOutputsValues) {
                for (const bool weightsToConst : weightsToConstValues) {
                    for (const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnActivations : quantizedTensorAlignmentOnActivationsValues) {
                        for (const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnWeights : quantizedTensorAlignmentOnWeightsValues) {
                            for (const bool roundQuantizedValue : roundQuantizedValues) {
                                for (const bool updateBiases : updateBiasesValues) {
                                    for (const bool supportAsymmetricQuantization : supportAsymmetricQuantizationValues) {
                                        for (const std::vector<Precision> precisionOnActivations : precisionOnActivationsValues) {
                                            for (const std::vector<Precision> precisionOnWeights : precisionOnWeightsValues) {
                                                network = createNetwork();

                                                p.model->resetTransformation(network);
                                                auto param = LayerTransformation::Params(
                                                    updatePrecision,
                                                    quantizeOutputs,
                                                    weightsToConst,
                                                    quantizedTensorAlignmentOnActivations,
                                                    quantizedTensorAlignmentOnWeights,
                                                    roundQuantizedValue,
                                                    updateBiases,
                                                    supportAsymmetricQuantization,
                                                    precisionOnActivations,
                                                    precisionOnWeights);

                                                const bool validate = p.model->transform(network, param);

#ifdef DISPLAY_RESULTS
                                                // TODO: separate each usecase to standalone parameterized test
                                                std::cout << std::endl <<
                                                    "\tupdatePrecision=" << (param.updatePrecisions ? "true" : "false") << std::endl <<
                                                    "\tquantizeOutputs=" << (param.quantizeOutputs ? "true" : "false") << std::endl <<
                                                    "\tweightsToConst=" << (param.weightsToConst ? "true" : "false") << std::endl <<
                                                    "\tquantizedTensorAlignmentOnActivations=" << param.quantizedTensorAlignmentOnActivations << std::endl <<
                                                    "\tquantizedTensorAlignmentOnWeights=" << param.quantizedTensorAlignmentOnWeights << std::endl <<
                                                    "\troundQuantizedValues: " << (param.roundQuantizedValues ? "true" : "false") << std::endl <<
                                                    "\tupdateBiases: " << (param.updateBiases ? "true" : "false") << std::endl <<
                                                    "\tsupportAsymmetricQuantization: " << (param.supportAsymmetricQuantization ? "true" : "false") << std::endl <<
                                                    "\tprecisionsOnActivations: " << param.precisionsOnActivations << std::endl <<
                                                    "\tprecisionsOnWeights: " << param.precisionsOnWeights << std::endl <<
                                                    "\tnetworkPrecision=" << p._network_precision << std::endl;
#endif

                                                //network.serialize(
                                                //    p.model->getName() + "_transformed.xml",
                                                //    p.model->getName() + "_transformed.bin");

                                                if (validate) {
                                                    LowPrecisionTransformationValidation::validate(
                                                        network,
                                                        param,
                                                        p.model->getNotTransformedLayers());
                                                }

                                                ExecutableNetwork executableNetworkTransformed;
                                                InferRequest inferRequestTransformed;
                                                const auto transformedOutput = infer(network, inputBlobs, core, p.device_name, executableNetworkTransformed, inferRequestTransformed);

                                                //compareInDetails(originalOutputMap, *transformedOutput, 70, 0.5);
                                                auto net_precision = network.getPrecision();
                                                for (auto& originalOutput : originalOutputMap) {
                                                    const auto& name = originalOutput.first;
                                                    const auto outSize = originalOutput.second->size();

                                                    auto transformed = CNNNetworkHelper::getFloatData(transformedOutput.find(name)->second);
                                                    auto original = CNNNetworkHelper::getFloatData(originalOutput.second);

                                                    const float threshold = p.model->getThreshold(p.device_name, net_precision, param);
                                                    const float zeroThreshold = p.model->getZeroThreshold();

                                                    const auto outName = transformedOutput.find(name);
                                                    if (outName == transformedOutput.end()) {
                                                        THROW_IE_EXCEPTION << "Original output name " + name + " doesn't exist in transformed model";
                                                    }

                                                    relative_compare(
                                                        CNNNetworkHelper::getFloatData(outName->second).get(),
                                                        CNNNetworkHelper::getFloatData(originalOutput.second).get(),
                                                        outSize,
                                                        threshold,
                                                        updatePrecision ? "failed with precisions" : "failed without precisions",
                                                        zeroThreshold);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } catch (const InferenceEngine::details::InferenceEngineException &e) {
        FAIL() << e.what();
    }
}
