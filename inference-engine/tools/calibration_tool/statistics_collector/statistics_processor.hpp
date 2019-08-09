// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include "inference_engine.hpp"
#include "samples/console_progress.hpp"
#include "data_stats.hpp"
#include "ie_blob.h"

struct ct_preprocessingOptions {
    std::string _pp_type;
    size_t _pp_size;
    size_t _pp_width;
    size_t _pp_height;
};

class StatisticsCollector {
public:
    StatisticsCollector(const std::string& deviceName,
                        const std::string& custom_cpu_library,
                        const std::string& custom_cldnn,
                        const std::string& modelFilePath,
                        const std::string& imagesPath,
                        size_t img_number,
                        size_t batch,
                        const ct_preprocessingOptions& preprocessingOptions,
                        const std::string& progress = "");

    /**
     * Initializes state to collect accuracy of network and collect statistic
     * of activations. The statistic of activations is stored in _statData and has all max/min for all
     * layers and for all pictures
     * The inference of all pictures and real collect of the statistic happen  during call of
     * Processor::Process()
     */
    void collectStatistics();
    void collectStatisticsToIR(const std::string& outModelName, const std::string& output_precision);

    /**
     * Statistic collected in the collectFP32Statistics is processed with threshold passed as a parameter
     * for this method. All values for each layers and for all pictures are sorted and number of min/max
     * values which  exceed threshold is thrown off
     * @param threshold - parameter for thrown off outliers in activation statistic
     * @return InferenceEngine::NetworkStatsMap - mapping of layer name to NetworkNodeStatsPtr
     */
    InferenceEngine::NetworkStatsMap getStatistics(float threshold = 100.f);

    void saveIRWithStatistics(const std::string& originalName,
                      const std::string& outModelName,
                      const InferenceEngine::NetworkStatsMap& statMap,
                      const std::string& output_precision);

protected:
    /**
     * This function should be called each Infer for each picture
     * It collects activation values statistic
     */
    void collectCalibrationStatistic(InferenceEngine::InferRequest& inferRequest);

    void Process(bool stream_output = false);

    static void fillBlobs(StatisticsCollector* collectorInstance);

    inline void addBlob(InferenceEngine::Blob::Ptr& blob) {
        _blobs_mutex.lock();
        _blobs.push_back(blob);
        _blobs_mutex.unlock();
    }
    inline InferenceEngine::Blob::Ptr popBlob() {
        InferenceEngine::Blob::Ptr blob;
        _blobs_mutex.lock();
        if (!_blobs.empty()) {
            blob = _blobs[0];
            _blobs.pop_front();
        }
        _blobs_mutex.unlock();
        return blob;
    }

    std::string _deviceName;
    std::string _custom_cpu_library;
    std::string _custom_cldnn;
    std::string _modelFilePath;
    std::string _imagesPath;
    std::string _progress;
    size_t _img_number;
    size_t _batch;
    std::shared_ptr<InferenceEngine::CNNNetwork> _cnn_network;
    std::shared_ptr<dataStats> _statData;
    ct_preprocessingOptions _preprocessingOptions;
    std::deque<InferenceEngine::Blob::Ptr> _blobs;
    std::mutex _blobs_mutex;
    std::shared_ptr<ConsoleProgress> _consoleProgress;
    bool _filesFinished;
};
