// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>
#include "inference_engine.hpp"
#include "ClassificationProcessor.hpp"
#include "SSDObjectDetectionProcessor.hpp"
#include "data_stats.h"
#include <map>
#include <memory>

/**
 * Calibrator class representing unified stages for calibration of any kind of networks
*/
class Int8Calibrator {
public:
    /**
     * Intermediate structure storing of data for measurements of by-layer statistic of accuracy drop
     */
    struct SingleLayerData {
        InferenceEngine::InferRequest _request;
        std::string _outputName;
        std::string _outputI8Name;
        std::vector<float> _int8Accuracy;
    };

    /**
     * Initializes state to collect accuracy of FP32 network and collect statistic
     * of activations. The statistic of activations is stored in _statData and has all max/min for all
     * layers and for all pictures
     * The inference of all pictures and real collect of the statistic happen  during call of
     * Processor::Process()
     */
    void collectFP32Statistic();

    /**
     * Initializes a state to collect intermediate numeric accuracy drop happening during quantization of
     * certain layer to int8. The numeric accuracy drop is measured using NRMSD metric.
     *
     * For this purpose it creates dedicated network for certain layer, initializes this
     * network by statistic that cause execute dedicated network in int8 mode.
     *
     * In addition to  original network we create full original network executed in FP32 mode, and
     * register all layers as output ones.
     * Information from these layers is used as
     *  a) input to dedicated layer networks
     *  b) comparison for NRMSD algorithm between I8 and FP32 calc
     *
     *  The inference of all pictures and real collect of the drop happen during call of
     * Processor::Process()
     * @param stat
     */
    void collectByLayerStatistic(const InferenceEngine::NetworkStatsMap &stat);

    /**
     * Initialize state to collect accuracy drop in int8 mode to be compared later vs FP32 accuracy
     * metric.
     *
     * The inference of all pictures and real collect of the accuracy happen during call of
     * Processor::Process()
     *
     * @param stat - The statistic for normalization
     * @param layersToInt8 - list of layers planned to be executed in int8. if layer is absent in this
     *                     map, it is assumed that it will be executed in int8
     * @param convertFullyConnected - should the FullyConnected layers be converted into Int8 or not
     */
    void validateInt8Config(const InferenceEngine::NetworkStatsMap &stat,
                                    const std::map<std::string, bool>& layersToInt8,
                                    bool convertFullyConnected);

    /**
     * Statistic collected in the collectFP32Statistic is processed with threshold passed as a parameter
     * for this method. All values for each layers and for all pictures are sorted and number of min/max
     * values which  exceed threshold is thrown off
     * @param threshold - parameter for thrown off outliers in activation statistic
     * @return InferenceEngine::NetworkStatsMap - mapping of layer name to NetworkNodeStatsPtr
     */
    InferenceEngine::NetworkStatsMap getStatistic(float threshold);

    /**
     * returns by-layer accuracy drop container
     */
    std::map<std::string, float> layersAccuracyDrop();

protected:
    /**
     * This function should be called from final callibrator after and each Infer for each picture
     * It calculates by layer accuracy drop and as well it also collect activation values statistic
     */
    void collectCalibrationStatistic(size_t pics);

    /**
     * This function should be called from calibration class after Infer of all picture
     * It calculates average NRMSD based accuracy drop for each layer and fills _layersAccuracyDrop
     */
    void calculateLayersAccuracyDrop();

    bool _collectByLayer = false;
    bool _collectStatistic = true;
    InferencePlugin _pluginI8C;
    std::string _modelFileNameI8C;
    InferenceEngine::CNNNetReader networkReaderC;
    InferenceEngine::InferRequest _inferRequestI8C;
    int _cBatch = 0;

    size_t _nPictures;

private:
    /**
     * helper function for getting statistic for input layers. For getting statistic for them, we are
     * adding scalshift just after the input with scale == 1 and shift == 0
     */
    CNNLayerPtr addScaleShiftBeforeLayer(std::string name, InferenceEngine::CNNLayer::Ptr beforeLayer,
                                         size_t port, std::vector<float> scale);

    /**
     * Returns Normalized root-mean-square deviation metric for two blobs passed to the function
     */
    float compare_NRMSD(InferenceEngine::Blob::Ptr res, InferenceEngine::Blob::Ptr ref);

    /**
     * Creates dedicated i8 network around selected layer. Currently this network beside layer itself
     * has to have ReLU and ScaleShift layers.
     * Since Inference Engine API mostly directed to the loading of network from IR, we need to create
     * such IR first, read through stream and modify network to correspond required parameters
     */
    InferenceEngine::CNNNetwork createICNNNetworkForLayer(InferenceEngine::CNNLayer::Ptr layerToClone,
                                                          bool hasReLU);

    std::map<std::string, float> _layersAccuracyDrop;
    std::vector<InferenceEngine::ExecutableNetwork> _singleLayerNetworks;
    std::map<std::string, SingleLayerData> _singleLayerRequests;
    std::map<std::string, std::string> _inputsFromLayers;
    AggregatedDataStats _statData;
};

/**
 * This class represents the only one generalized metric which will be used for comparison of
 * accuracy drop
 */
struct CalibrationMetrics : public ClassificationProcessor::InferenceMetrics {
public:
    float AccuracyResult = 0;
};

/**
 * Сalibration class for classification networks.
 * Responsible for proper post processing of results and calculate of Top1 metric which is used as
 * universal metric for accuracy and particiapted in verification of accuracy drop
 */
class ClassificationCalibrator : public ClassificationProcessor, public Int8Calibrator {
public:
    ClassificationCalibrator(int nPictures, const std::string &flags_m, const std::string &flags_d,
                             const std::string &flags_i, int flags_b,
                              InferenceEngine::InferencePlugin plugin, CsvDumper &dumper, const std::string &flags_l,
                              PreprocessingOptions preprocessingOptions, bool zeroBackground);

    shared_ptr<InferenceMetrics> Process(bool stream_output = false) override;
};


/**
* Calibration class for SSD object detection networks.
* Responsible for proper post processing of results and calculate of mAP metric which is used as
* universal metric for accuracy and participated in verification of accuracy drop
*/
class SSDObjectDetectionCalibrator : public SSDObjectDetectionProcessor, public Int8Calibrator {
public:
    SSDObjectDetectionCalibrator(int nPictures, const std::string &flags_m, const std::string &flags_d,
                                 const std::string &flags_i, const std::string &subdir, int flags_b,
                                 double threshold,
                                 InferencePlugin plugin, CsvDumper &dumper,
                                 const std::string &flags_a, const std::string &classes_list_file);

    shared_ptr<InferenceMetrics> Process(bool stream_output = false) override;
};
