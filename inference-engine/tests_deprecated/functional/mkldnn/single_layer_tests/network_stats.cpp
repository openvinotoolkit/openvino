// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cfloat>
#include <fstream>
#include <limits>
#include <memory>

#include <pugixml.hpp>

#include <format_reader_ptr.h>

#include "network_stats.h"
#include <samples/slog.hpp>

using namespace InferenceEngine;

class DataStats {
public:
    template <typename T>
    static void GetDataMinMax(const T* data, size_t count, T& min, T& max);

    template <typename T>
    static void GetDataAverage(const T* data, size_t count, T& ave);

    template <typename T>
    static void GetDataAbsMax(const T* data, size_t count, T& max);

    template <typename T>
    static T GetAbsMax(T min, T max);
};

template <typename T>
void DataStats::GetDataMinMax(const T* data, size_t count, T& min, T& max) {
    for (size_t i = 0; i < count; i++) {
        T val = data[i];

        if (min > val) {
            min = val;
        }

        if (max < val) {
            max = val;
        }
    }
}

template <typename T>
void DataStats::GetDataAbsMax(const T* data, size_t count, T& max) {
    T min = FLT_MAX;

    GetDataMinMax(data, count, min, max);

    max = GetAbsMax(min, max);
}

template void DataStats::GetDataMinMax<float>(const float* data, size_t count, float& min, float& max);
template void DataStats::GetDataMinMax<uint8_t>(const uint8_t* data, size_t count, uint8_t& min, uint8_t& max);

template void DataStats::GetDataAbsMax<float>(const float* data, size_t count, float& max);

template <typename T>
void DataStats::GetDataAverage(const T* data, size_t count, T& ave) {
    ave = 0;

    for (size_t i = 0; i < count; i++) {
        ave += data[i];
    }

    ave /= count;
}

template void DataStats::GetDataAverage<float>(const float* data, size_t count, float& ave);

template <typename T>
T DataStats::GetAbsMax(T min, T max) {
    if (min < 0) {
        min *= -1;
    }

    if (max < 0) {
        max *= -1;
    }

    return (max > min) ? max : min;
}

template float DataStats::GetAbsMax<float>(float min, float max);


CNNLayerPtr NetworkStatsCollector::addScaleShiftBeforeLayer(std::string name, CNNLayer::Ptr beforeLayer, size_t port, std::vector<float> scale) {
    if (beforeLayer->insData.size() < port) {
        THROW_IE_EXCEPTION << "cannot find appropraite port for addScaleShiftBeforeLayer";
    }

    DataPtr pData = beforeLayer->insData[port].lock();
    LayerParams params;
    params.name = name;
    params.precision = Precision::FP32;
    params.type = "ScaleShift";
    CNNLayerPtr lptr = std::make_shared<ScaleShiftLayer>(params);
    ScaleShiftLayer *pScaleShift = dynamic_cast<ScaleShiftLayer *>(lptr.get());

    IE_ASSERT(4 == pData->getDims().size());
    std::size_t num_chanels = pData->getDims().at(1);
    SizeVector wdims({ num_chanels });

    if (scale.size() == 1) {
        scale.resize(wdims[0]);
        for (int i = 1; i < wdims[0]; i++) {
            scale[i] = scale[0];
        }
    }

    if (scale.size() != num_chanels) {
        THROW_IE_EXCEPTION << "Failed to add scaleshift before " << beforeLayer->name << " due to scales and layer output dims incossitency";
    }

    Blob::Ptr weights = nullptr;
    weights = make_shared_blob<float>({Precision::FP32, wdims, Layout::C});
    weights->allocate();
    float *buffer = weights->buffer().as<float *>();
    for (size_t i = 0; i < num_chanels; i++) {
        buffer[i] = scale[i];
    }
    pScaleShift->_weights = weights;


    SizeVector bdims({ num_chanels });
    Blob::Ptr biases = nullptr;
    biases = make_shared_blob<float>({Precision::FP32, bdims, Layout::C});
    biases->allocate();
    buffer = biases->buffer().as<float *>();
    for (size_t i = 0; i < num_chanels; i++) {
        buffer[i] = 0.f;
    }
    pScaleShift->_biases = biases;

    Data *edge2 = new Data(*pData.get());
    DataPtr newEdge(edge2);
    lptr->insData.push_back(pData);
    lptr->outData.push_back(newEdge);
    newEdge->setName(/*"EdgeAfter_" +*/ params.name);
    newEdge->getCreatorLayer() = lptr;
    newEdge->getInputTo().clear();
    newEdge->getInputTo()[beforeLayer->name] = beforeLayer;

    pData->getInputTo().erase(beforeLayer->name);
    pData->getInputTo()[params.name] = lptr;

    for (size_t i = 0; i < beforeLayer->insData.size(); i++) {
        DataPtr d = beforeLayer->insData[i].lock();
        if (d == pData) {
            beforeLayer->insData[i] = newEdge;
            break;
        }
    }
    return lptr;
}

NetworkStatsCollector::NetworkStatsCollector(const InferenceEngine::Core & ie, const std::string & deviceName) :
    _ie(ie), _deviceName(deviceName) {
}

NetworkStatsCollector::~NetworkStatsCollector() {
}

void NetworkStatsCollector::ReadNetworkAndSetWeights(const void *model, size_t size, const InferenceEngine::TBlob<uint8_t>::Ptr &weights, size_t batch) {
    /** Reading network model **/
    _network = _ie.ReadNetwork((const char*)model, weights);
    _network.setBatchSize(batch);
}

std::string FileNameNoExt(const std::string& filePath) {
    auto pos = filePath.rfind('.');

    if (pos == std::string::npos) {
        return filePath;
    }

    return filePath.substr(0, pos);
}

void NetworkStatsCollector::LoadNetwork(const std::string& modelPath, size_t batch) {
    /** Reading network model **/
    _network = _ie.ReadNetwork(modelPath);
    _network.setBatchSize(batch);
}

void NetworkStatsCollector::InferAndCollectStats(const std::vector<std::string>& images,
                                                 std::map<std::string, NetworkNodeStatsPtr>& netNodesStats) {
    slog::info << "Collecting statistics for layers:" << slog::endl;

    std::vector<CNNLayerPtr> layersAfterInputs;

    std::string hackPrefix = "scaleshifted_input:";

    std::map<std::string, std::string> inputsFromLayers;
    for (auto&& layer : _network) {
        if (layer->insData.size() > 0) {
            std::string inName = layer->input()->getName();
            for (auto&& input : _network.getInputsInfo()) {
                if (inName == input.first) {
                    layersAfterInputs.push_back(layer);
                    inputsFromLayers[hackPrefix + layer->name] = inName;
                }
            }
        }
    }

    for (auto&& layer : layersAfterInputs) {
        std::string firstInputName = hackPrefix + layer->name;
        auto scaleShiftLayer = addScaleShiftBeforeLayer(firstInputName, layer, 0, { 1.f });
        ((ICNNNetwork&)_network).addLayer(scaleShiftLayer);
    }

    // Adding output to every layer
    for (auto&& layer : _network) {
        slog::info << "\t" << layer->name << slog::endl;

        std::string layerType = _network.getLayerByName(layer->name.c_str())->type;
        if (/*layerType != "Split" &&*/ layerType != "Input") {
            _network.addOutput(layer->name);
        }
    }

    NetworkNodeStatsPtr nodeStats;

    const size_t batchSize = _network.getBatchSize();

    std::vector<std::string> imageNames;

    size_t rounded = images.size() - images.size() % batchSize;

    auto executable_network = _ie.LoadNetwork(_network, _deviceName);

    std::map<std::string, std::vector<float>> min_outputs, max_outputs;

    for (size_t i = 0; i < rounded; i += batchSize) {
        slog::info << "Inferring image " << i+1 << " of " << rounded << slog::endl;

        imageNames.clear();

        for (size_t img = 0; img < batchSize; img++) {
            imageNames.push_back(images[i + img]);
        }


        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(_network.getInputsInfo());

        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");
        auto inputInfoItem = *inputInfo.begin();

        /** Specifying the precision of input data provided by the user.
         * This should be called before load of the network to the device **/
        inputInfoItem.second->setPrecision(Precision::FP32);
        inputInfoItem.second->setLayout(Layout::NCHW);

        std::vector<std::shared_ptr<unsigned char>> imagesData;
        for (auto & i : imageNames) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            auto data_dims = inputInfoItem.second->getTensorDesc().getDims();
            std::shared_ptr<unsigned char> data(reader->getData(data_dims.back(), data_dims.at(data_dims.size() - 2)));
            if (data.get() != nullptr) {
                imagesData.push_back(data);
            }
        }
        if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

        OutputsDataMap outputInfo(_network.getOutputsInfo());
        for (auto itOut : outputInfo) {
            itOut.second->setPrecision(Precision::FP32);
        }

        auto infer_request = executable_network.CreateInferRequest();

        // -------------------------------Set input data----------------------------------------------------
        /** Iterate over all the input blobs **/

        /** Creating input blob **/
        Blob::Ptr input = infer_request.GetBlob(inputInfoItem.first);
        if (!input) {
            throw std::logic_error("Invalid input blob " + inputInfoItem.first + " pointer");
        }

        /** Filling input tensor with images. First b channel, then g and r channels **/
        auto input_dims = input->getTensorDesc().getDims();
        size_t num_chanels = input_dims.at(1);
        size_t image_size = input_dims.at(input_dims.size() - 2) * input_dims.back();

        auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

        /** Iterate over all input images **/
        for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
            /** Iterate over all pixel in image (b,g,r) **/
            for (size_t pid = 0; pid < image_size; pid++) {
                /** Iterate over all channels **/
                for (size_t ch = 0; ch < num_chanels; ++ch) {
                    /**          [images stride + channels stride + pixel id ] all in bytes            **/
                    data[image_id * image_size * num_chanels + ch * image_size + pid ] = imagesData.at(image_id).get()[pid*num_chanels + ch];
                }
            }
        }

        infer_request.Infer();


        for (auto itOut : outputInfo) {
            auto outBlob = infer_request.GetBlob(itOut.first);

            std::string outName = itOut.first;
            if (inputsFromLayers.find(itOut.first) != inputsFromLayers.end()) {
                outName = inputsFromLayers[itOut.first];
            }

            size_t N, C, statCount;
            auto output_dims = outBlob->getTensorDesc().getDims();
            if (output_dims.size() == 4 && outBlob->getTensorDesc().getLayout() == Layout::NCHW) {
                N = output_dims[0];
                C = output_dims[1];
                statCount = C;
            } else if (output_dims.size() == 2 && outBlob->getTensorDesc().getLayout() == Layout::NC) {
                N = output_dims[0];
                C = output_dims[1];
                statCount = 1;
            } else {
                slog::warn << "Only NCHW and NC layouts are supported. Skipping layer \"" << outName << "\"" << slog::endl;
                continue;
            }


            if (netNodesStats.find(outName) == netNodesStats.end()) {
                nodeStats = NetworkNodeStatsPtr(new NetworkNodeStats(statCount));

                netNodesStats[outName] = nodeStats;
            } else {
                nodeStats = netNodesStats[outName];
            }

            // Counting min/max outputs per channel
            for (size_t n = 0; n < N; n++) {
                if (output_dims.size() == 4) {
                    size_t _HW = output_dims.back() * output_dims.at(output_dims.size() - 2);
                    for (size_t c = 0; c < C; c++) {
                        if (outBlob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
                            float* ptr = &outBlob->buffer().as<float*>()[(n * C + c) * _HW];

                            float min = nodeStats->_minOutputs[c];
                            float max = nodeStats->_maxOutputs[c];
                            DataStats::GetDataMinMax<float>(ptr, _HW, min, max);
                            nodeStats->_minOutputs[c] = min;
                            nodeStats->_maxOutputs[c] = max;
                        } else if (outBlob->getTensorDesc().getPrecision() == InferenceEngine::Precision::U8) {
                            uint8_t* ptr = &outBlob->buffer().as<uint8_t*>()[(n * C + c) * _HW];

                            uint8_t min = nodeStats->_minOutputs[c];
                            uint8_t max = nodeStats->_maxOutputs[c];
                            DataStats::GetDataMinMax<uint8_t>(ptr, _HW, min, max);
                            nodeStats->_minOutputs[c] = min;
                            nodeStats->_maxOutputs[c] = max;
                        } else {
                            throw std::logic_error(std::string("Unsupported precision: ") + outBlob->getTensorDesc().getPrecision().name());
                        }
                    }
                } else if (output_dims.size() == 2) {
                    if (outBlob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
                        float* ptr = &outBlob->buffer().as<float*>()[n * C];

                        float min = nodeStats->_minOutputs[0];
                        float max = nodeStats->_maxOutputs[0];
                        DataStats::GetDataMinMax<float>(ptr, C, min, max);
                        nodeStats->_minOutputs[0] = min;
                        nodeStats->_maxOutputs[0] = max;
                    } else if (outBlob->getTensorDesc().getPrecision() == InferenceEngine::Precision::U8) {
                        uint8_t* ptr = &outBlob->buffer().as<uint8_t*>()[n * C];

                        uint8_t min = nodeStats->_minOutputs[0];
                        uint8_t max = nodeStats->_maxOutputs[0];
                        DataStats::GetDataMinMax<uint8_t>(ptr, C, min, max);
                        nodeStats->_minOutputs[0] = min;
                        nodeStats->_maxOutputs[0] = max;
                    } else {
                        throw std::logic_error(std::string("Unsupported precision: ") + outBlob->getTensorDesc().getPrecision().name());
                    }
                }
            }
        }
    }
}