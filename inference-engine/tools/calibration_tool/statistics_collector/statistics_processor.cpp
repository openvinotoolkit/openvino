// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <deque>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <condition_variable>

#include <ie_extension.h>

#include "statistics_processor.hpp"
#include "cpp/ie_cnn_net_reader.h"
#include "details/caseless.hpp"
#include "ie_plugin_config.hpp"
#include "ie_parallel.hpp"
#include <extension/ext_list.hpp>
#include "samples/common.hpp"
#include "samples/console_progress.hpp"
#include "samples/slog.hpp"
#include "inference_engine/cpp_interfaces/ie_task_synchronizer.hpp"
#include "image_decoder.hpp"
#include "utils.hpp"

using namespace InferenceEngine::details;
using InferenceEngine::details::InferenceEngineException;


StatisticsCollector::StatisticsCollector(const std::string& deviceName,
                                        const std::string& custom_cpu_library,
                                        const std::string& custom_cldnn,
                                        const std::string& modelFilePath,
                                        const std::string& imagesPath,
                                        size_t img_number,
                                        size_t batch,
                                        const ct_preprocessingOptions& preprocessingOptions,
                                        const std::string& progress) :
        _img_number(img_number),
        _deviceName(deviceName),
        _imagesPath(imagesPath),
        _custom_cpu_library(custom_cpu_library),
        _modelFilePath(modelFilePath),
        _batch(batch),
        _preprocessingOptions(preprocessingOptions),
        _progress(progress),
        _filesFinished(false) {
    if (_modelFilePath.empty() || !isFile(_modelFilePath))
        THROW_IE_EXCEPTION << "Path to IR xml file is expected in simplified mode. Not a directory.";
    if (_imagesPath.empty() || (!isFile(_imagesPath) && !isDirectory(_imagesPath)))
        THROW_IE_EXCEPTION << "Invalig path to images.";
    if (!_custom_cpu_library.empty() && !isFile(_custom_cpu_library) && !isDirectory(_custom_cpu_library))
        THROW_IE_EXCEPTION << "Invalig CPU extension path.";
    if (!custom_cldnn.empty() && !isFile(custom_cldnn) && !isDirectory(custom_cldnn))
        THROW_IE_EXCEPTION << "Invalig GPU extension path.";
    _statData = std::make_shared<simpleDataStats>();
}

InferenceEngine::NetworkStatsMap StatisticsCollector::getStatistics(float threshold) {
    InferenceEngine::NetworkStatsMap netNodesStats;
    // go over all outputs and get aggregated statistics
    for (auto& outName : _statData->registeredLayers()) {
        NetworkNodeStatsPtr nodeStats;
        size_t channels = outName._channels;
        if (netNodesStats.find(outName._name) == netNodesStats.end()) {
            nodeStats = NetworkNodeStatsPtr(new NetworkNodeStats(channels));

            netNodesStats[outName._name] = nodeStats;
        } else {
            nodeStats = netNodesStats[outName._name];
        }
        for (size_t c = 0; c < channels; c++) {
            _statData->getDataMinMax(outName._name, c, nodeStats->_minOutputs[c], nodeStats->_maxOutputs[c], threshold);
        }
    }
    return netNodesStats;
}

void StatisticsCollector::collectStatistics() {
    InferenceEngine::CNNNetReader networkReader = InferenceEngine::CNNNetReader();
    networkReader.ReadNetwork(_modelFilePath);
    if (!networkReader.isParseSuccess())
        THROW_IE_EXCEPTION << "Could not load a model " << _modelFilePath;
    /** Extract model name and load weights **/
    std::string binFileName = fileNameNoExt(_modelFilePath) + ".bin";
    networkReader.ReadWeights(binFileName.c_str());
    auto network = networkReader.getNetwork();
    if (_batch == 0) {
        // Zero means "take batch value from the IR"
        _batch = network.getBatchSize();
    } else {
        // Not zero means "use the specified value"
        auto input_shapes = network.getInputShapes();
        std::string input_name;
        SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        input_shape[0] = _batch;
        input_shapes[input_name] = input_shape;
        network.reshape(input_shapes);
    }

    _cnn_network = std::make_shared<CNNNetwork>(network);

    for (auto &&layer : *_cnn_network) {
        const std::string& layerType = layer->type;
        if (!CaselessEq<std::string>()(layerType, "const") &&
            !CaselessEq<std::string>()(layerType, "split") &&
            !CaselessEq<std::string>()(layerType, "input")) {
            _cnn_network->addOutput(layer->name);
        }
    }

    // Read files from file system in parallel to reduce suspences
    _filesFinished = false;
    std::thread readFilesThread(fillBlobs, this);
    Process();
    readFilesThread.join();
}

void StatisticsCollector::Process(bool stream_output) {
    slog::info << "Collect inputs..." << slog::endl;
    auto inputInfo = _cnn_network->getInputsInfo();
    if (inputInfo.size() != 1) {
        THROW_IE_EXCEPTION << "Just networks with one input are supported.";
    }

    slog::info << "Loading Inference Engine" << slog::endl;
    Core ie;
    if (_deviceName.find("CPU") != std::string::npos && !_custom_cpu_library.empty()) {
        if (isFile(_custom_cpu_library)) {
            // CPU extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(_custom_cpu_library);
            ie.AddExtension(extension_ptr, "CPU");
            slog::info << "CPU Extension loaded: " << _custom_cpu_library << slog::endl;
        } else {
            slog::info << "Path to CPU Extension does not exist or it is a directory: '" <<
                _custom_cpu_library << "'" << slog::endl;
            slog::info << "Simplified mode required path to CPU Extension not a directory." << slog::endl;
        }
    }
    if (!_custom_cldnn.empty()) {
        // GPU extensions are loaded from an .xml description and OpenCL kernel files
        ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, _custom_cldnn}}, "GPU");
        slog::info << "GPU Extension loaded: " << _custom_cldnn << slog::endl;
    }

    slog::info << "Device info: " << slog::endl;
    std::cout << ie.GetVersions(_deviceName) << std::endl;

    ExecutableNetwork executable_network = ie.LoadNetwork(*_cnn_network, _deviceName,
            { { CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES) } });
    // Choose requests number depending on the number of layers to reduce memory consumption.
    size_t requests_num = parallel_get_max_threads() * 80 / executable_network.GetExecGraphInfo().layerCount();
    requests_num = requests_num == 0lu ? 1lu : requests_num;
    if (!CaselessEq<std::string>()(_deviceName, "cpu") && requests_num > MAX_NUMBER_OF_TASKS_IN_QUEUE)
        requests_num = MAX_NUMBER_OF_TASKS_IN_QUEUE;
    std::vector<InferRequest> inferRequests;

    for (size_t i = 0lu; i < requests_num; i++) {
        inferRequests.push_back(executable_network.CreateInferRequest());
    }
    for (auto& info : inputInfo) {
        const auto& outBlobDesc = inferRequests[0].GetBlob(info.first)->getTensorDesc();
        if ((outBlobDesc.getLayout() != Layout::NCHW) &&
            (outBlobDesc.getLayout() != Layout::NC) &&
            (outBlobDesc.getLayout() != Layout::C) &&
            (outBlobDesc.getLayout() != Layout::NCDHW)) {
            continue;
        } else {
            _statData->registerLayer(info.first, getTensorBatch(outBlobDesc), getTensorChannels(outBlobDesc));
        }
    }
    for (auto& info : _cnn_network->getOutputsInfo()) {
        const auto& outBlobDesc = inferRequests[0].GetBlob(info.first)->getTensorDesc();
        if ((outBlobDesc.getLayout() != Layout::NCHW) &&
            (outBlobDesc.getLayout() != Layout::NC) &&
            (outBlobDesc.getLayout() != Layout::C) &&
            (outBlobDesc.getLayout() != Layout::NCDHW)) {
            continue;
        } else {
            _statData->registerLayer(info.first, getTensorBatch(outBlobDesc), getTensorChannels(outBlobDesc));
        }
    }

    const std::string firstInputName = inputInfo.begin()->first;
    std::condition_variable condVar;
    std::mutex infers_mutex;
    bool finished = false;
    size_t processedImages = 0lu;

    slog::info << "Start inference" << slog::endl;
    auto start = std::chrono::system_clock::now();
    for (auto& inferRequest : inferRequests) {
        bool requestInitialized = false;
        while (!_filesFinished || !_blobs.empty()) {
            Blob::Ptr newblob = popBlob();
            if (newblob != nullptr) {
                inferRequest.SetBlob(firstInputName, newblob);
                requestInitialized = true;
                break;
            }
        }
        if (!requestInitialized)
            break;
        inferRequest.SetCompletionCallback(
                [&] {
                    collectCalibrationStatistic(inferRequest);
                    if (_consoleProgress)
                        _consoleProgress->addProgress(_batch);
                    processedImages++;
                    if (!_filesFinished || !_blobs.empty()) {
                        while (!_filesFinished || !_blobs.empty()) {
                            Blob::Ptr newblob = popBlob();
                            if (newblob != nullptr) {
                                inferRequest.SetBlob(firstInputName, newblob);
                                inferRequest.StartAsync();
                                break;
                            }
                        }
                    } else {
                        finished = true;
                        condVar.notify_one();
                    }
                });
        inferRequest.StartAsync();
    }
    std::unique_lock<std::mutex> lock(infers_mutex);
    condVar.wait(lock, [&]{
        if (!finished) return false;
        for (auto& inferRequest : inferRequests)
            inferRequest.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
        return true;
    });

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    slog::info << slog::endl << processedImages << " objects processed in "
            << elapsed_seconds.count() << " seconds" << slog::endl;
    if (_consoleProgress)
        _consoleProgress->finish();
    slog::info << "Inference passed successfully" << slog::endl;
}

void StatisticsCollector::collectCalibrationStatistic(InferenceEngine::InferRequest& inferRequest) {
    for (auto& info : _statData->registeredLayers()) {
        auto outBlob = inferRequest.GetBlob(info._name);
        const auto& outBlobDesc = outBlob->getTensorDesc();
        const size_t N = info._batch;
        const size_t C = info._channels;

#define ADD_STATISTICS for (size_t n = 0lu; n < N; n++) { \
    size_t nC = n * C; \
    for (size_t c = 0lu; c < C; c++) { \
        _statData->addStatistics(info._name, c, &ptr[(nC + c) * HW], HW); \
    } \
}
        size_t HW = 1lu;
        if (outBlobDesc.getLayout() == Layout::NCHW)
            HW = getTensorWidth(outBlobDesc) * getTensorHeight(outBlobDesc);
        if (outBlobDesc.getPrecision() == Precision::FP32) {
            float* ptr = outBlob->buffer().as<float *>();
            ADD_STATISTICS
        } else if (outBlobDesc.getPrecision() == Precision::FP16) {
            short* ptr = outBlob->buffer().as<short *>();
            ADD_STATISTICS
        } else if (outBlobDesc.getPrecision() == Precision::U8) {
            uint8_t* ptr = outBlob->buffer().as<uint8_t *>();
            ADD_STATISTICS
        } else {
            throw std::logic_error(std::string("Unsupported precision: ") + outBlobDesc.getPrecision().name());
        }
    }
}

#define MAKE_SHARED_BLOB(precision)\
case InferenceEngine::Precision::precision  : {\
    new_blob = make_shared_blob<PrecisionTrait<Precision::precision>::value_type>(inputDesc); \
    break;\
}

void StatisticsCollector::fillBlobs(StatisticsCollector* collectorInstance) {
    auto datasetEntries = getDatasetEntries(collectorInstance->_imagesPath, collectorInstance->_img_number);
    if (datasetEntries.empty()) {
        THROW_IE_EXCEPTION << "No applicable dataset files were found by path '"
                << collectorInstance->_imagesPath
                << "'. Check the dataset file or folder.";
    }
    size_t img_number = datasetEntries.size();
    size_t progress_step = 1lu;
    if (collectorInstance->_progress == "print")
        progress_step = 100lu;
    collectorInstance->_consoleProgress = std::make_shared<ConsoleProgress>(img_number);

    TensorDesc inputDesc = collectorInstance->_cnn_network->getInputsInfo().begin()->second->getTensorDesc();
    const Precision::ePrecision inputPrecision = inputDesc.getPrecision();

    PreprocessingOptions preprocessingOptions;
    if (CaselessEq<std::string>()(collectorInstance->_preprocessingOptions._pp_type, "none")) {
    } else if (CaselessEq<std::string>()(collectorInstance->_preprocessingOptions._pp_type, "resizecrop")) {
        if (collectorInstance->_preprocessingOptions._pp_size == 0lu ||
                (collectorInstance->_preprocessingOptions._pp_width == 0 && collectorInstance->_preprocessingOptions._pp_height == 0)) {
            THROW_IE_EXCEPTION << "Size must be specified for preprocessing type " << collectorInstance->_preprocessingOptions._pp_type;
        }
        size_t ppWidth = collectorInstance->_preprocessingOptions._pp_width > 0lu ?
            collectorInstance->_preprocessingOptions._pp_width : collectorInstance->_preprocessingOptions._pp_size;
        size_t ppHeight = collectorInstance->_preprocessingOptions._pp_height > 0lu ?
            collectorInstance->_preprocessingOptions._pp_height : collectorInstance->_preprocessingOptions._pp_size;
        preprocessingOptions = PreprocessingOptions(false, ResizeCropPolicy::ResizeThenCrop, ppWidth, ppHeight);
    } else if (CaselessEq<std::string>()(collectorInstance->_preprocessingOptions._pp_type, "resize") ||
            collectorInstance->_preprocessingOptions._pp_type.empty()) {
        preprocessingOptions.resizeCropPolicy = ResizeCropPolicy::Resize;
    } else {
        THROW_IE_EXCEPTION << "Unknown preprocessing type: " << collectorInstance->_preprocessingOptions._pp_type;
    }

    auto iter = datasetEntries.begin();
    size_t ipics = 0lu;
    while (iter != datasetEntries.end()) {
        Blob::Ptr new_blob;
        switch (inputPrecision) {
            MAKE_SHARED_BLOB(FP32);
            MAKE_SHARED_BLOB(FP16);
            default:
                THROW_IE_EXCEPTION << "Cannot process blob for precision: " << inputPrecision;
        }
        new_blob->allocate();
        size_t b = 0lu;
        for (; b < collectorInstance->_batch && iter != datasetEntries.end() && ipics + b < img_number ; b++, iter++) {
            try {
                ImageDecoder::insertIntoBlob(*iter, b, *new_blob, preprocessingOptions);
            } catch (const InferenceEngineException &iex) {
                slog::warn << "Can't read file " << *iter << slog::endl;
                // Could be some non-image file in directory
                b--;
                continue;
            }
        }
        collectorInstance->addBlob(new_blob);
        ipics += collectorInstance->_batch;
    }
    collectorInstance->_filesFinished = true;
}

void StatisticsCollector::saveIRWithStatistics(const std::string& originalName,
                      const std::string& outModelName,
                      const InferenceEngine::NetworkStatsMap& statMap,
                      const std::string& output_precision) {
    CNNNetReader networkReader;
    networkReader.ReadNetwork(originalName);
    if (!networkReader.isParseSuccess())
        THROW_IE_EXCEPTION << "Cannot load a failed Model";

    /** Extract model name and load weights **/
    std::string binFileName = fileNameNoExt(originalName) + ".bin";
    networkReader.ReadWeights(binFileName.c_str());

    bool precision_as_is = false;
    Precision target_precision;
    if (output_precision.empty())
        precision_as_is = true;
    else if (CaselessEq<std::string>()(output_precision, "fp32"))
        target_precision = Precision::FP32;
    else if (CaselessEq<std::string>()(output_precision, "fp16"))
        target_precision = Precision::FP16;
    else
        THROW_IE_EXCEPTION << "Unsupported precision '" << output_precision << "'";

    auto network = networkReader.getNetwork();
    for (auto &&layer : network) {
        if (layer->precision == Precision::FP32 && (CaselessEq<std::string>()(layer->type, "convolution") ||
                    CaselessEq<std::string>()(layer->type, "fullyconnected")))
            layer->params["quantization_level"] = "I8";
        if (!precision_as_is) {
            auto wLayer = dynamic_cast<InferenceEngine::WeightableLayer *>(layer.get());
            if (wLayer) {
                auto weights_precision = wLayer->_weights->getTensorDesc().getPrecision();
                if (wLayer->_weights && target_precision != weights_precision) {
                    if (target_precision == Precision::FP16 &&
                            weights_precision == Precision::FP32) {
                        wLayer->_weights = convertBlobFP32toFP16(wLayer->_weights);
                    } else if (target_precision == Precision::FP32 &&
                            weights_precision == Precision::FP16) {
                        wLayer->_weights = convertBlobFP16toFP32(wLayer->_weights);
                    } else {
                        THROW_IE_EXCEPTION << "Weights for the layer '" << wLayer->name
                                << "' have unsupported precision '" << weights_precision.name() << "'";
                    }
                }

                auto biases_precision = wLayer->_biases->getTensorDesc().getPrecision();
                if (wLayer->_biases && target_precision != biases_precision) {
                    if (target_precision == Precision::FP16 &&
                            biases_precision == Precision::FP32) {
                        wLayer->_biases = convertBlobFP32toFP16(wLayer->_biases);
                    } else if (target_precision == Precision::FP32 &&
                            biases_precision == Precision::FP16) {
                        wLayer->_biases = convertBlobFP16toFP32(wLayer->_biases);
                    } else {
                        THROW_IE_EXCEPTION << "Biases for the layer '" << wLayer->name
                                << "' have unsupported precision '" << biases_precision.name() << "'";
                    }
                }
            }
            layer->precision = target_precision;
        }
    }

    ICNNNetworkStats* pstats = nullptr;
    StatusCode s = ((ICNNNetwork&)network).getStats(&pstats, nullptr);
    if (s == StatusCode::OK && pstats) {
        pstats->setNodesStats(statMap);
    }

    slog::info << "Write network with statistics to " << outModelName << ".(xml|bin) IR file\n";
    network.serialize(outModelName + ".xml", outModelName + ".bin");
}

void StatisticsCollector::collectStatisticsToIR(const std::string& outModelName, const std::string& output_precision) {
    collectStatistics();
    saveIRWithStatistics(_modelFilePath, outModelName, getStatistics(), output_precision);
}
