// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <format_reader_ptr.h>
#include <gtest/gtest.h>
#include <legacy/graph_tools.hpp>
#include "raw_matcher.hpp"
#include <precision_utils.h>

namespace Regression {
namespace Matchers {

void RawMatcher::match() {
    try {
        // Read network
        std::string binFileName = testing::FileUtils::fileNameNoExt(config._path_to_models) + ".bin";
        std::cout << config._path_to_models << std::endl;
        auto cnnNetwork = config.ie_core->ReadNetwork(config._path_to_models, binFileName);

        InferenceEngine::InputsDataMap networkInputs;
        networkInputs = cnnNetwork.getInputsInfo();
        if (networkInputs.size() == 0) {
            THROW_IE_EXCEPTION << "No inputs detected.";
        }

        if (config._paths_to_images.size() % ( config.batchSize * networkInputs.size()) != 0) {
            std::cerr << "[WARNING]: Can not process all input images("<< config._paths_to_images.size()
                      <<") using given batch size of " << config.batchSize << ". Batch size will be equal 1." << std::endl;
            config.batchSize = 1;
        }

        InferenceEngine::DataPtr inputData = cnnNetwork.getInputsInfo().begin()->second->getInputData();
        InferenceEngine::SizeVector inputDims = inputData->getTensorDesc().getDims();

        if (config._reshape) {
            auto inputShapes = cnnNetwork.getInputShapes();
            inputShapes.begin()->second[0] = config.batchSize;

            cnnNetwork.reshape(inputShapes);
        } else if (config.batchSize != 1) {
            cnnNetwork.setBatchSize(config.batchSize);
        }

        // TODO(amalyshe) quick dirty solution which might not cover all topologies,
        // but covers only networks having one input passing to one layer
        CNNLayerPtr layer;
        for (auto input : networkInputs) {
            InputInfo::Ptr q = input.second;
            if (config._inputPrecision) q->setPrecision(config._inputPrecision);
            DataPtr p = q->getInputData();
            IE_SUPPRESS_DEPRECATED_START
            layer = getInputTo(p).begin()->second;
            IE_SUPPRESS_DEPRECATED_END
        }

        {
            // Set output precision
            InferenceEngine::OutputsDataMap out;
            out = cnnNetwork.getOutputsInfo();
            for (auto &&item : out) {
                Blob::Ptr output;
                auto  outputName = item.first;
                auto& outBlob    = item.second;
                if (config._outputPrecision) outBlob->setPrecision(config._outputPrecision);
                if (config._outputBlobPrecision.count(outputName)) outBlob->setPrecision(config._outputBlobPrecision[outputName]);
            }
        }

        if (!config.deviceMapping.empty()) {
            IE_SUPPRESS_DEPRECATED_START
            CNNNetDFS(layer, [&](const CNNLayerPtr &layer) {
                auto it = config.deviceMapping.find(layer->name);
                if (it != config.deviceMapping.end()) {
                    layer->affinity = it->second;
                } else {
                    layer->affinity = "CPU";
                }
            });
            IE_SUPPRESS_DEPRECATED_END
        }

        // Read image
        std::vector<std::shared_ptr<unsigned char>> imagesData;
        unsigned int actualNetSize = 0;
        for (auto & imageName : config._paths_to_images) {
            FormatReader::ReaderPtr reader(imageName.c_str());
            if (reader.get() == nullptr) {
                THROW_IE_EXCEPTION << "[ERROR]: Image " + imageName + " cannot be read!";
            }
            actualNetSize += reader->size();
            // Store image data

            size_t width = 0, height = 0;
            SizeVector dims = inputData->getTensorDesc().getDims();
            if (dims.size() == 3) {
                height = dims.at(1);
                width = dims.at(2);
            } else if (dims.size() == 4) {
                height = dims.at(2);
                width = dims.at(3);
            } else if (dims.size() == 5) {
                height = dims.at(3);
                width = dims.at(4);
            } else {
                THROW_IE_EXCEPTION << inputData->getName() << " has unsupported layout " << inputData->getTensorDesc().getLayout();
            }

            std::shared_ptr<unsigned char> data(reader->getData(width, height));
            if (data.get() != nullptr) {
                imagesData.push_back(data);
            } else {
                THROW_IE_EXCEPTION << "Invalid image '" << imageName << "'";
            }
        }

        auto out2 = cnnNetwork.getOutputsInfo();
        for (auto &&item : out2) {
            if (config._outputPrecision) item.second->setPrecision(config._outputPrecision);
            if (config._outputBlobPrecision.count(item.first)) {
                item.second->setPrecision(config._outputBlobPrecision[item.first]);
            }
        }

        auto loadedExecutableNetwork = config.ie_core->LoadNetwork(cnnNetwork, config._device_name, config.plugin_config);
        InferenceEngine::ExecutableNetwork executableNetwork;
        if (config.useExportImport) {
            std::stringstream stream;
            loadedExecutableNetwork.Export(stream);
            executableNetwork = config.ie_core->ImportNetwork(stream);
        } else {
            executableNetwork = loadedExecutableNetwork;
        }
        auto inferRequest = executableNetwork.CreateInferRequest();

        InferenceEngine::BlobMap inputBlobs;

        auto allocateBlob = [](const InferenceEngine::TensorDesc& desc) {
            InferenceEngine::Blob::Ptr blob;
            switch (desc.getPrecision()) {
                case InferenceEngine::Precision::FP32 :
                    blob = InferenceEngine::make_shared_blob<float>(desc);
                    break;
                case InferenceEngine::Precision::FP16 :
                case InferenceEngine::Precision::Q78 :
                case InferenceEngine::Precision::I16 :
                    blob = InferenceEngine::make_shared_blob<int16_t>(desc);
                    break;
                case InferenceEngine::Precision::U8 :
                    blob = InferenceEngine::make_shared_blob<uint8_t>(desc);
                    break;
                default:
                    THROW_IE_EXCEPTION << "Unsupported blob precision: " << desc.getPrecision();
            }
            blob->allocate();

            return blob;
        };

        for(auto&& inputInfo : cnnNetwork.getInputsInfo()) {
            std::string inputName = inputInfo.first;

            if (!inferRequest) {
                // Allocate blobs
                inputBlobs[inputName] = allocateBlob(inputInfo.second->getTensorDesc());
            } else {
                inputBlobs[inputName] = inferRequest.GetBlob(inputName);
            }
        }

        {
            InferenceEngine::OutputsDataMap out;
            out = cnnNetwork.getOutputsInfo();
            for (auto &&item : out) {
                Blob::Ptr output;
                auto  outputName = item.first;
                if (!inferRequest) {
                    output = allocateBlob(item.second->getTensorDesc());
                } else {
                    // TODO(amalyshe): we need to return GetBlob eventually after the fix bug in mkldnnplugin
                    output = inferRequest.GetBlob(outputName);
                    // output = allocateBlob(item.second->getTensorDesc());
                    // inferRequest.SetBlob(outputName, output);
                }
                outputBlobs[outputName] = output;
            }
        }

        // loading images in batches
        for (int i = 0; i < config._paths_to_images.size(); i += config.batchSize * inputBlobs.size()) {
            int k = 0;
            for(auto&& input: inputBlobs) {
                for (int j = 0; j != config.batchSize; j++) {
                    const auto & imageName  = config._paths_to_images[i + j + k];
                    loadImage(imageName, input.second, true, j);
                }
                k++;
            }

            if (config.isAsync) {
                inferRequest.StartAsync();
                inferRequest.Wait(IInferRequest::WaitMode::RESULT_READY);
            } else {
                inferRequest.Infer();
            }

            // Get performance info
            if (config.perfInfoPtr != nullptr) {
                *config.perfInfoPtr = inferRequest.GetPerformanceCounts();
            }
        }
    } catch (details::InferenceEngineException &e) {
        FAIL() << e.what();
    }
    catch (std::exception &e) {
        FAIL() << e.what();
    }
}

void RawMatcher::checkResult(const std::map<std::string, std::map<size_t, float>> &allExpected) {
    auto prepareResults = [&](const Blob::Ptr& output) {
        std::vector<float> tmp_buffer;

        if (output->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP16) {
            tmp_buffer.resize(output->size(), 0.f);
            PrecisionUtils::f16tof32Arrays(tmp_buffer.data(),
                                           output->buffer().as<int16_t*>(),
                                           output->size());
        } else {
            assert(output->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32);
            tmp_buffer.resize(output->size(), 0.f);
            std::copy_n(output->buffer().as<float*>(), output->size(), tmp_buffer.begin());
        }

        return tmp_buffer;
    };
    if(config.print) {
        std::cout << "{";
        for(auto&& out : outputBlobs) {
            Blob::Ptr& output = out.second;
            auto results = prepareResults(output);
            std::cout << "{{\""  << out.first <<"\", {\n";
            for(std::size_t i = 0; i < output->size(); i += (output->size() + config.printNum - 1)/ config.printNum) {
                std::cout << "{" << i <<", "<< results[i] << "},\n";
            }
            std::cout << "}}},\n";
        }
        std::cout << "};" << std::endl;
    } else {
        std::stringstream strm;
        auto generateInfo = [&](const std::vector<float>& results, const std::map<std::size_t, float>& expected) {
            double meanRelative = 0;
            double maxAbsolute = 0;
            double maxRelative = 0;
            strm << std::endl << std::setw(15) << "Position" << std::setw(15) <<
                "Expected" << std::setw(15) <<
                "Actual" << std::setw(15) <<
                "Absolute" << std::setw(15) <<
                "Relative,%" << std::endl;
            for (auto e : expected) {
                double absolute = fabs(e.second - results[e.first]);
                double relative = fabs(e.second - results[e.first]) / fabs(e.second);

                strm << std::setw(15) << e.first
                     << std::setw(15) << std::setprecision(6) << e.second
                     << std::setw(15) << std::setprecision(6) << results[e.first]
                     << std::setw(15) << std::setprecision(6) << absolute
                     << std::setw(15) << std::setprecision(6) << relative*100 << std::endl;
                meanRelative += relative;
                maxAbsolute = std::max(maxAbsolute, absolute);
                maxRelative = std::max(maxRelative, relative);
            }
            strm << "Max Absolute = " << maxAbsolute
                 << " Mean Relative = " << meanRelative*100/expected.size()
                 << " Max Relative = " << maxRelative*100  << '\n';
        };

        if(0 != config.nearValue) {
            for(auto expectedPair : allExpected) {
                Blob::Ptr output = outputBlobs[expectedPair.first];
                if (!output) {
                    FAIL() << "Was not able to find expected output " << expectedPair.first;
                }

                auto results = prepareResults(output);

                const std::map<size_t, float> &expected = expectedPair.second;

                for (auto e : expected) {
                    if (fabs(e.second - results[e.first]) > config.nearValue) {
                        strm << "In blob " << expectedPair.first
                             << " element at " << e.first << " index expected to be " << e.second << " but in fact it is "
                             << results[e.first] <<
                             " Delta = " << (fabs(e.second - results[e.first]));
                        generateInfo(results, expected);
                        FAIL() << strm.str();
                    }
                }
            }
        }
        if(0 != config.meanRelativeError) {
            for(auto expectedPair : allExpected) {
                Blob::Ptr output = outputBlobs[expectedPair.first];
                if (!output) {
                    FAIL() << "Was not able to find expected output " << expectedPair.first;
                }
                auto results = prepareResults(output);

                std::map<std::size_t, float>& expected = expectedPair.second;

                double meanRelative = 0;
                for (auto e : expected) {
                    double eps = fabs(e.second - results[e.first]) / fabs(e.second);
                    meanRelative += eps;
                }
                meanRelative /= expected.size();
                meanRelative *= 100;

                if (meanRelative > config.meanRelativeError) {
                    strm << "In blob " << expectedPair.first
                         << " Mean Relative Error = " << meanRelative
                         << " Expected Mean Relative Error = " << config.meanRelativeError;
                    generateInfo(results, expected);
                    FAIL() << strm.str();
                }
            }
        }
        if(0 != config.maxRelativeError) {
            for(auto expectedPair : allExpected) {
                Blob::Ptr output = outputBlobs[expectedPair.first];
                if (!output) {
                    FAIL() << "Was not able to find expected output " << expectedPair.first;
                }
                auto results = prepareResults(output);

                std::map<std::size_t, float>& expected = expectedPair.second;

                double maxRelative = 0;
                std::size_t maxPos = 0;
                for (auto e : expected) {
                    double eps = fabs(e.second - results[e.first]) / fabs(e.second);
                    if(eps > maxRelative) {
                        maxRelative = eps;
                        maxPos = e.first;
                    }
                }
                maxRelative *= 100;

                if (maxRelative > config.maxRelativeError) {
                    strm << "In blob " << expectedPair.first << " element at " << maxPos << " index"
                         << " expected to be " << expected[maxPos] << " but in fact it is " << results[maxPos]
                         << " Max Relative Error = " << maxRelative
                         << " Expected Max Relative Error = " << config.maxRelativeError;
                    generateInfo(results, expected);
                    FAIL() << strm.str();
                }
            }
        }
    }
}

}
} //  namespace matchers
