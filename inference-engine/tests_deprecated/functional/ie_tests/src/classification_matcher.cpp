// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "classification_matcher.hpp"
#include <gtest/gtest.h>
#include <xml_helper.hpp>
#include "details/ie_cnn_network_iterator.hpp"

using namespace Regression ;
using namespace Regression :: Matchers ;

ClassificationMatcher::ClassificationMatcher(RegressionConfig &config)
    : BaseMatcher(config) {
    // Get file names for files with weights and labels
    std::string binFileName = testing::FileUtils::fileNameNoExt(config._path_to_models) + ".bin";

    auto cnnNetwork = config.ie_core->ReadNetwork(config._path_to_models, binFileName);

    std::string labelFileName = testing::FileUtils::fileNameNoExt(config._path_to_models) + ".labels";

    // Try to read labels file
    readLabels(labelFileName);

    if (config._reshape) {
        auto inputShapes = cnnNetwork.getInputShapes();
        inputShapes.begin()->second[0] = config.batchSize;

        cnnNetwork.reshape(inputShapes);
    } else if (config.batchSize != 1) {
        cnnNetwork.setBatchSize(config.batchSize);
    }

    _inputsInfo = cnnNetwork.getInputsInfo();
    _outputsInfo = cnnNetwork.getOutputsInfo();
    for (auto &info : _inputsInfo) {
        if (config._inputPrecision != InferenceEngine::Precision::UNSPECIFIED) {
            info.second->setPrecision(config._inputPrecision);
        }
    }

    for (auto &info : _outputsInfo) {
        if (config._outputPrecision != Precision::UNSPECIFIED) {
            info.second->setPrecision(config._outputPrecision);
        } else {
            info.second->setPrecision(config.modelPrecision);
        }
    }

    if (config.useDynamicBatching) {
        config.plugin_config[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
        cnnNetwork.setBatchSize(config.batchSize);
    }

    for (int i=0; i < config._numNetworks; i++) {
        auto loadedExecutableNetwork = config.ie_core->LoadNetwork(cnnNetwork, config._device_name, config.plugin_config);
        InferenceEngine::ExecutableNetwork executableNetwork;
        if (config.useExportImport) {
            std::stringstream stream;
            loadedExecutableNetwork.Export(stream);
            executableNetwork = config.ie_core->ImportNetwork(stream);
        } else {
            executableNetwork = loadedExecutableNetwork;
        }
        _executableNetworks.push_back(executableNetwork);
    }

    top = (-1 == config.topKNumbers) ? 5 : config.topKNumbers;
}

void ClassificationMatcher::to(const std::vector <Regression::Reference::ClassificationScoringResultsForTests> &expected) {
    checkResultNumber = 0;
    match(std::min(top, expected.size()));
    checkResult(checkResultNumber, expected);
    checkResultNumber++;
}

void ClassificationMatcher::to(std::string modelType) {
    auto batchSize = config.batchSize;

    if (config.useDynamicBatching) {
        batchSize = config.dynBatch;
    }

    checkImgNumber(batchSize);
    ASSERT_NO_FATAL_FAILURE(match(10));  // This method produces top-10 reference results.
    for (size_t i = 0; i < config._paths_to_images.size(); i++) {
        const size_t last_slash_idx = config._paths_to_images[i].find_last_of(kPathSeparator);
        if (std::string::npos != last_slash_idx) {
            config._paths_to_images[i].erase(0, last_slash_idx + 1);
        }
        if (Regression::Reference::values.find(modelType + "_" + config._paths_to_images[i]) ==
            Regression::Reference::values.end()) {
            FAIL() << "Reference result for " << modelType + "_" + config._paths_to_images[i] << " cannot be found";
        }
        ASSERT_NO_FATAL_FAILURE(checkResult(i, Regression::Reference::values[modelType + "_" + config._paths_to_images[i]]));
    }
    checkResultNumber++;
}


void ClassificationMatcher::readLabels(std::string labelFilePath) {
    std::fstream fs(labelFilePath, std::ios_base::in);
    if (fs.is_open()) {
        std::string line;
        while (getline(fs, line)) {
            config.labels.push_back(TestsCommon::trim(line));
        }
    } else {
        THROW_IE_EXCEPTION << "cannot open label file: " << labelFilePath;

    }
}

int ClassificationMatcher::getIndexByLabel(const std::string &label) {
    auto result = std::find(begin(config.labels), end(config.labels), label);
    if (result == config.labels.end()) {
        THROW_IE_EXCEPTION << "cannot locate index for label : " << label;
    }
    return static_cast<int>(std::distance(begin(config.labels), result));
}

std::string ClassificationMatcher::getLabel(unsigned int index) {
    if (config.labels.empty()) {
        return "label #" + std::to_string(index);
    }
    if (index >= config.labels.size()) {
        THROW_IE_EXCEPTION << "index out of labels file: " << index;
    }

    return config.labels[index];
}

void ClassificationMatcher::checkResult(size_t checkNumber,
                                         const std::vector <Regression::Reference::ClassificationScoringResultsForTests> &expected) {
    if (checkNumber >= _results.size()) {
        FAIL() << "Expected number of results(" << checkNumber << ") is more than real number of results: "
               << _results.size();
    }
    auto result = _results.at(checkNumber);

    std::map<std::string, float> expected_map;
    int expectedSize = expected.size();
    int resultSize = result.size();

    if (config.topKNumbers != -1) {
        expectedSize = config.topKNumbers;
        resultSize = config.topKNumbers;
    }

    for (int i = 0; i < expectedSize; ++i) {
        expected_map[expected[i].getLabel()] = expected[i].getProbability();
    }

    for (int i = 0; i < resultSize; ++i) {
        if (expected_map.count(result[i].getLabel())) {
            ASSERT_NEAR(result[i].getProbability(), expected_map[result[i].getLabel()], config.nearValue)
                                << "Failed for label \"" << result[i].getLabel() << "\" index "  << i;
            expected_map.erase(result[i].getLabel());
        } else {
            // Label which not in expected list can be below last expected element
            ASSERT_LE(result[i].getProbability(), expected.back().getProbability() + config.nearValue)
                                << "Label \"" << result[i].getLabel() << "\" not found or cannot be in expected list";
        }
    }

    if (expected_map.size() != 0) {
        for (auto & elem: expected_map) {
            std::cout << "Label \"" << elem.first << "\" with probability="
                      << elem.second << " not found in result list" << std::endl;
        }
        FAIL();
    }
}

void ClassificationMatcher::match(size_t top) {
    for (int i = 0; i != _executableNetworks.size(); i++) {
        match_n(top, i);
    }
}

namespace {

template <class T>
inline void TopResults(unsigned int n, TBlob<T>& input, std::vector<unsigned>& output) {
    SizeVector dims = input.getTensorDesc().getDims();
    size_t input_rank = dims.size();
    if (!input_rank || !dims[0]) THROW_IE_EXCEPTION << "Input blob has incorrect dimensions!";
    size_t batchSize = dims[0];
    std::vector<unsigned> indexes(input.size() / batchSize);

    n = static_cast<unsigned>(std::min<size_t>((size_t)n, input.size()));

    output.resize(n * batchSize);

    for (size_t i = 0; i < batchSize; i++) {
        size_t offset = i * (input.size() / batchSize);
        T* batchData = input.data();
        batchData += offset;

        std::iota(std::begin(indexes), std::end(indexes), 0);
        std::partial_sort(std::begin(indexes), std::begin(indexes) + n, std::end(indexes),
                          [&batchData](unsigned l, unsigned r) {
                              return batchData[l] > batchData[r];
                          });
        for (unsigned j = 0; j < n; j++) {
            output.at(i * n + j) = indexes.at(j);
        }
    }
}

}

void ClassificationMatcher::match_n(size_t top, int index) {
    try {
        InferenceEngine::IInferRequest::Ptr inferRequest;
        if (_executableNetworks[index]->CreateInferRequest(inferRequest, &_resp) != OK) {
            THROW_IE_EXCEPTION << "Can not create infer request: " << _resp.msg;
        }
        std::string prevImageName = "";

        auto batchSize = config.batchSize;

        if (config.useDynamicBatching) {
            batchSize = config.dynBatch;
            InferenceEngine::ResponseDesc resp;
            inferRequest->SetBatch(batchSize, &resp);
        }

        if (config._paths_to_images.size() % batchSize != 0) {
            THROW_IE_EXCEPTION << "Can not process all input images("<< config._paths_to_images.size()
                               <<") using given batch size of " << batchSize;
        }
        // loading images in batches
        for (int i = 0; i < config._paths_to_images.size(); i += batchSize) {

            // has same image names
            bool areImagesSame = false;
            if (i > 0)  {
                areImagesSame = true;
                for (int j = i;j != i + batchSize; j++) {
                    if (config._paths_to_images[j] != config._paths_to_images[j - batchSize]) {
                        areImagesSame = false;
                        break;
                    }
                }
            }
            if (!areImagesSame) {
                for (int j = 0; j != batchSize; j++) {
                    const auto & imageName  = config._paths_to_images[i + j];

                    InferenceEngine::Blob::Ptr inputBlob;
                    if (inferRequest->GetBlob(_inputsInfo.begin()->first.c_str(), inputBlob, &_resp) != OK) {
                        THROW_IE_EXCEPTION << "Can not get input with name: " << _inputsInfo.begin()->first
                                           << " error message: " << _resp.msg;
                    }
                    loadImage(imageName, inputBlob, true, j);
                }
            }

            StatusCode status = inferRequest->Infer(&_resp);
            if (status != OK) {
                THROW_IE_EXCEPTION << "Can not do infer: " << _resp.msg;
            }

            InferenceEngine::Blob::Ptr outputBlobPtr;
            if (inferRequest->GetBlob(_outputsInfo.begin()->first.c_str(), outputBlobPtr, &_resp) != OK) {
                THROW_IE_EXCEPTION << "Can not get output with name: " << _outputsInfo.begin()->first
                                   << " error message: " << _resp.msg;
            }

            InferenceEngine::TBlob<float>::Ptr outputFP32;
                if (outputBlobPtr->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP16) {
                    TensorDesc desc = { InferenceEngine::Precision::FP32, outputBlobPtr->getTensorDesc().getDims(),
                        outputBlobPtr->getTensorDesc().getLayout() };
                    outputFP32 = make_shared_blob<float>(desc);
                    outputFP32->allocate();
                    PrecisionUtils::f16tof32Arrays(outputFP32->buffer().as<float *>(), outputBlobPtr->cbuffer().as<short *>(), outputBlobPtr->size());
                } else if (outputBlobPtr->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
                    outputFP32 = dynamic_pointer_cast<InferenceEngine::TBlob<float>>(outputBlobPtr);
                } else {
                    THROW_IE_EXCEPTION << "Unsupported output format for test. Supported FP16, FP32";
                }

            vector<unsigned> topClassesIndexes;
            TopResults<float>(top, *outputFP32, topClassesIndexes);
            std::vector<float> probabilities(outputFP32->buffer().as<float *>(),
                                             outputFP32->buffer().as<float *>() + outputFP32->size());

            saveResults(topClassesIndexes, probabilities, top);
        }
    } catch (InferenceEngine::details::InferenceEngineException &e) {
        FAIL() << e.what();
    } catch (std::exception &e) {
        FAIL() << e.what();
    }
}

void ClassificationMatcher::saveResults(const std::vector<unsigned> &topIndexes, const std::vector<float> &probs, size_t top) {

    for(auto idx = topIndexes.begin(); idx != topIndexes.end();) {
        std::vector<Reference::LabelProbability> topResults;
        for (int i = 0; i != top; i++) {
            Reference::LabelProbability labelProb(*idx, probs[*idx], getLabel(*idx));
            std::cout << "index=" << labelProb.getLabelIndex() << ", probability=" << labelProb.getProbability()
                      << ", class=" << labelProb.getLabel() << "\n";
            topResults.push_back(labelProb);
            idx++;
        }
        _results.push_back(topResults);
    }
}
