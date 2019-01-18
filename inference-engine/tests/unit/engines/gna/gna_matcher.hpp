// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <limits>
#include <inference_engine/graph_tools.hpp>
#include "gtest/gtest.h"
#include "inference_engine.hpp"
#include "gna/gna_config.hpp"
#include "gna_plugin.hpp"
#include "gna-api.h"
#include "test_irs.hpp"
#include "dnn.h"


#define withConfig(key, value) withGNAConfig(GNA_CONFIG_KEY(key), value)
#define ASSERT_NO_THROW_IE_EXCEPTION(expr) \
try {\
expr;\
}catch(std::exception & e) {\
    FAIL() << e.what();\
}\
catch(...) {\
    FAIL() << "unknown exception";\
}

/**
 * GNA unit tests environment
 */
class GnaPluginTestEnvironment {
 public:
    struct NnetPrecision {
        InferenceEngine::Precision input_precision;
        InferenceEngine::Precision output_precision;
        InferenceEngine::Precision weights_precision;
        InferenceEngine::Precision biases_precision;
    };
    enum MatchWhat {
        exactNNetStructure,
        matchNone,
        matchProcType,
        matchPrecision,
        matchPwlInserted,
        matchConvInserted,
        matchMaxPoolingInserted,
        matchPwlQuantizeMetrics,
        matchCopyInserted,
        matchDiagonalInserted,
        saveArgs
    };
    std::vector<MatchWhat> whatToMatch;
    enum {
        kUnset = -1,
        kAnyNotNull= -2
    };
    InferenceEngine::TargetDevice target_device =
                            InferenceEngine::TargetDevice::eGNA;
    int matchQuantity = kUnset;
    int numberOfStates = kUnset;
    bool matchInserted = true;
    NnetPrecision nnet_precision;
    float quantization_presicion_threshold = 1.0f;
    uint16_t quantization_segments_threshold = UINT16_MAX;
    uint32_t type = 0;
    std::string model;
    std::string exportedModelFileName;
    bool exportNetworkOnly = false;
    std::function<void (InferenceEngine::CNNNetwork &)> cb;
    std::map<std::string, std::string> config;
    bool matchThrows = false;
    uint32_t proc_type = static_cast<intel_gna_proc_t>(GNA_SOFTWARE & GNA_HARDWARE);
    std::string importedModelFileName;
    bool is_profiling_enabled = false;
    bool matchOutput = false;
    bool is_setup_of_omp_theads_expected = false;
    std::vector<float> input_init;
    std::vector<float> expected_output;
};

class GNATestBase {
 public:
    virtual ~GNATestBase() = default;
};

template <class T>
class GNATestConfigurability : public GNATestBase{
 protected:
    bool needNextMatcher = true;
    GnaPluginTestEnvironment _env;
    GnaPluginTestEnvironment::MatchWhat & getMatcher() {
        if (needNextMatcher) {
            needNextMatcher = false;
            _env.whatToMatch.push_back({});
        }
        return _env.whatToMatch.back();
    }
 public:
    GNATestConfigurability(GnaPluginTestEnvironment env) : _env(env) {
    }
    T & And() {
        needNextMatcher = true;
        return *dynamic_cast<T*>(this);
    }
    template <class VType>
    T & withGNAConfig(const std::string keyName, const VType &value) {
        std::stringstream ss;
        ss << value;
        _env.config[keyName] = ss.str();
        return *dynamic_cast<T*>(this);
    }
    T & withGNADeviceMode(std::string value) {
        _env.config[GNA_CONFIG_KEY(DEVICE_MODE)] = value;
        return *dynamic_cast<T*>(this);
    }
    T & withAcceleratorThreadsNumber(std::string value) {
        _env.config[GNA_CONFIG_KEY(LIB_N_THREADS)] = value;
        return *dynamic_cast<T*>(this);
    }
    T & throws() {
        _env.matchThrows = true;
        return *dynamic_cast<T*>(this);
    }
    T & profiling_counters() {
        _env.is_profiling_enabled = true;
        _env.config[CONFIG_KEY(PERF_COUNT)] = InferenceEngine::PluginConfigParams::YES;
        return *dynamic_cast<T*>(this);
    }

    T & enable_omp_multithreading() {
        _env.is_setup_of_omp_theads_expected = true;
        _env.config[CONFIG_KEY(SINGLE_THREAD)] = InferenceEngine::PluginConfigParams::NO;
        return *dynamic_cast<T*>(this);
    }
};

/**
 * @brief matches loadnetwork + infer + call to gna_api propagate
 */
class GNAPropagateMatcher : public GNATestConfigurability<GNAPropagateMatcher> {
 public:
    using base = GNATestConfigurability<GNAPropagateMatcher>;
    using base::base;
    using base::getMatcher;

    ~GNAPropagateMatcher() {
        match();
    }

    GNAPropagateMatcher & called() {
        // inserting default matcher that matches any propagate_forward call
        getMatcher();
        return *this;
    }

    GNAPropagateMatcher & called_with() {
        return *this;
    }

    GNAPropagateMatcher & called_without() {
        _env.matchInserted = false;
        return *this;
    }

    GNAPropagateMatcher & called_with_input_and_expected_output(std::vector<float>& input_data,
                                                                std::vector<float>& expect) {
        _env.matchOutput = true;
        _env.input_init = input_data;
        _env.expected_output = expect;
        return *this;
    }

    GNAPropagateMatcher & once() {
        _env.matchQuantity = 1;
        return *this;
    }

    GNAPropagateMatcher & twice() {
        _env.matchQuantity = 2;
        return *this;
    }

    GNAPropagateMatcher & args(std::string args) {
        return *this;
    }

    GNAPropagateMatcher & exact_nnet_structure(intel_nnet_type_t * pNet) {

        getMatcher() = GnaPluginTestEnvironment::exactNNetStructure;
        original_nnet = pNet;
        return *this;
    }

    GNAPropagateMatcher & pwl_inserted_into_nnet() {
        getMatcher() = GnaPluginTestEnvironment::matchPwlInserted;
        return *this;
    }

    GNAPropagateMatcher & max_pooling_inserted_into_nnet() {
        getMatcher() = GnaPluginTestEnvironment::matchMaxPoolingInserted;
        return *this;
    }

    GNAPropagateMatcher & succeed() {
        return *this;
    }

    GNAPropagateMatcher & convolution_inserted_into_nnet() {
        getMatcher() = GnaPluginTestEnvironment::matchConvInserted;
        return *this;
    }


    GNAPropagateMatcher & pwl_quantization_activation(uint32_t activation_type) {
        getMatcher() = GnaPluginTestEnvironment::matchPwlQuantizeMetrics;
        _env.type = activation_type;
        return *this;
    }

    GNAPropagateMatcher & pwl_quantization_precision_threshold(float threshold) {
        getMatcher() = GnaPluginTestEnvironment::matchPwlQuantizeMetrics;
        _env.quantization_presicion_threshold = threshold;
        return *this;
    }

    GNAPropagateMatcher & pwl_quantization_segments_threshold(uint16_t threshold) {
        getMatcher() = GnaPluginTestEnvironment::matchPwlQuantizeMetrics;
        _env.quantization_segments_threshold = threshold;
        return *this;
    }

    GNAPropagateMatcher & diagonal_inserted_into_nnet() {
        getMatcher() = GnaPluginTestEnvironment::matchDiagonalInserted;
        return *this;
    }

    GNAPropagateMatcher & copy_inserted_into_nnet() {
        getMatcher() = GnaPluginTestEnvironment::matchCopyInserted;
        return *this;
    }

    GNAPropagateMatcher & nnet_input_precision(const InferenceEngine::Precision &precision) {
        getMatcher() = GnaPluginTestEnvironment::matchPrecision;
        _env.nnet_precision.input_precision = precision;
        return *this;
    }
    GNAPropagateMatcher & nnet_ouput_precision(const InferenceEngine::Precision &precision) {
        getMatcher() = GnaPluginTestEnvironment::matchPrecision;
        _env.nnet_precision.output_precision = precision;
        return *this;
    }
    GNAPropagateMatcher & nnet_weights_precision(const InferenceEngine::Precision &precision) {
        getMatcher() = GnaPluginTestEnvironment::matchPrecision;
        _env.nnet_precision.weights_precision = precision;
        return *this;
    }
    GNAPropagateMatcher & nnet_biases_precision(const InferenceEngine::Precision &precision) {
        getMatcher() = GnaPluginTestEnvironment::matchPrecision;
        _env.nnet_precision.biases_precision = precision;
        return *this;
    }

    GNAPropagateMatcher & proc_type(uint32_t proc_type) {
        getMatcher() = GnaPluginTestEnvironment::matchProcType;
        _env.proc_type = proc_type;
        return * this;
    }

    GNAPropagateMatcher & to(intel_nnet_type_t *savedNet) {
        this->savedNet = savedNet;
        return *this;
    }

    GNAPropagateMatcher & onCPU() {
        _env.target_device = InferenceEngine::TargetDevice::eCPU;
        return *this;
    }
 protected:
    void match();
    intel_nnet_type_t * original_nnet = nullptr;
    intel_nnet_type_t * savedNet = nullptr;
};


/**
 * @brief GNAPlugin matches creation only case
 */
class GNAPluginCreationMatcher : public GNATestConfigurability<GNAPluginCreationMatcher> {
 public:
    using base = GNATestConfigurability<GNAPluginCreationMatcher>;
    using base::base;

    GNAPluginCreationMatcher & gna_plugin() {
        return * this;
    }
    ~GNAPluginCreationMatcher () {
        match();
    }
 protected:
    void match();
};

/**
 * @brief GNAPlugin matches creation only case
 */
class GNAPluginAOTMatcher : public GNATestConfigurability<GNAPluginAOTMatcher> {
 public:
    using base = GNATestConfigurability<GNAPluginAOTMatcher>;
    using base::base;

    ~GNAPluginAOTMatcher() {
        match();
    }
 protected:
    void match();
};

/**
 * @brief xnn api tests
 */
class GNADumpXNNMatcher : public GNATestConfigurability<GNADumpXNNMatcher> {
 public:
    using base = GNATestConfigurability<GNADumpXNNMatcher>;
    using base::base;

    ~GNADumpXNNMatcher() {
        if (match_in_dctor) {
            match();
        }
    }
    GNADumpXNNMatcher& called() {
        return *this;
    }
 protected:

    bool match_in_dctor = true;
    void load(GNAPluginNS::GNAPlugin & plugin);
    void match();
};

/**
 * @brief xnn api tests
 */
class GNAQueryStateMatcher : public GNADumpXNNMatcher {
 public:
    using base = GNADumpXNNMatcher;
    using base::base;

    ~GNAQueryStateMatcher() {
        if (match_in_dctor) {
            match();
            match_in_dctor = false;
        }
    }
    void isEmpty() {
        _env.numberOfStates = 0;
    }
    void isNotEmpty() {
        _env.numberOfStates = GnaPluginTestEnvironment::kAnyNotNull;
    }

 protected:
    void match();
};



/**
 * @brief base for test fixture
 */
class GNATest : public ::testing::Test, public GNATestConfigurability<GNATest>  {
    using base = GNATestConfigurability<GNATest>;
    using base::_env;
    std::list<std::vector<uint8_t>> dataUsedInMatchers;
    std::list<std::shared_ptr<GNATestBase>> returnedMatchers;

 public:
    template <class T>
    T & storage () {
        dataUsedInMatchers.push_back(std::vector<uint8_t >(sizeof(T)));
        return *reinterpret_cast<T*> (&dataUsedInMatchers.back().front());
    }
    GNATest()  : base(GnaPluginTestEnvironment()) {}
    GNATest & as() {
        return *this;
    }
    GNATest & model() {
        return *this;
    }
    GNATest & assert_that() {
        return *this;
    }
    GNATest & export_network(std::string modelName) {
        _env.model = modelName;
        _env.exportNetworkOnly = true;
        return *this;
    }
    GNATest & save_args() {
        getMatcher() = GnaPluginTestEnvironment::saveArgs;
        return *this;
    }

    GNATest & onInfer1AFModel() {
        _env.model = GNATestIRs::Fc2DOutputModel();
        return *this;
    }
    GNATest & onLoad(std::string _model) {
        _env.model = _model;
        return *this;
    }
    GNATest & afterLoadingModel(std::string _model) {
        _env.model = _model;
        return *this;
    }

    GNAQueryStateMatcher & queryState() {
        returnedMatchers.push_back(std::make_shared<GNAQueryStateMatcher>(_env));
        // clearing env;
        _env = GnaPluginTestEnvironment();
        return dynamic_cast<GNAQueryStateMatcher&>(*returnedMatchers.back());
    }

    /**importing indicates no infer happened ata all **/
    GNAPropagateMatcher & importingModelFrom(std::string fileName) {
        _env.importedModelFileName = fileName;
        returnedMatchers.push_back(std::make_shared<GNAPropagateMatcher>(_env));
        // clearing env;
        _env = GnaPluginTestEnvironment();
        return dynamic_cast<GNAPropagateMatcher&>(*returnedMatchers.back());
    }
    GNATest & importedFrom(std::string fileName) {
        _env.importedModelFileName = fileName;
        return *this;
    }
    GNATest & onInferModel(std::string _model = "",
                           std::function<void (InferenceEngine::CNNNetwork &)> _cb = [](InferenceEngine::CNNNetwork & net){}) {
        _env.model = _model;
        _env.cb = _cb;
        return *this;
    }
    GNATest & gna() {
        return *this;
    }
    GNATest & from() {
        return *this;
    }
    GNATest & inNotCompactMode() {
        _env.config[GNA_CONFIG_KEY(COMPACT_MODE)] = CONFIG_VALUE(NO);
        return *this;
    }
    GNATest & withUniformPWLAlgo() {
        base::_env.config[GNA_CONFIG_KEY(PWL_UNIFORM_DESIGN)] = CONFIG_VALUE(YES);
        return *this;
    }
    GNAPropagateMatcher& propagate_forward() {
        returnedMatchers.push_back(std::make_shared<GNAPropagateMatcher>(_env));
        //clearing env;
        _env = GnaPluginTestEnvironment();
        return dynamic_cast<GNAPropagateMatcher&>(*returnedMatchers.back());
    }
    GNADumpXNNMatcher& dumpXNN() {
        returnedMatchers.push_back(std::make_shared<GNADumpXNNMatcher>(_env));
        //clearing env;
        _env = GnaPluginTestEnvironment();
        return dynamic_cast<GNADumpXNNMatcher&>(*returnedMatchers.back());
    }
    GNATest & withNanScaleFactor() {
        base::_env.config[GNA_CONFIG_KEY(SCALE_FACTOR)] = std::to_string(std::numeric_limits<float>::quiet_NaN());
        return *this;
    }
    GNATest & withInfScaleFactor() {
        base::_env.config[GNA_CONFIG_KEY(SCALE_FACTOR)] = std::to_string(std::numeric_limits<float>::infinity());
        return *this;
    }
    GNAPluginCreationMatcher creating() {
        return _env;
    }

    GNAPluginAOTMatcher & to (std::string fileName) {
        _env.exportedModelFileName = fileName;
        returnedMatchers.push_back(std::make_shared<GNAPluginAOTMatcher>(_env));
        //clearing env;
        _env = GnaPluginTestEnvironment();
        return dynamic_cast<GNAPluginAOTMatcher&>(*returnedMatchers.back());
    }

    static void fillWeights(InferenceEngine::Blob::Ptr weights, float value = 1) {
        std::fill_n(weights->buffer().as<float*>(), weights->byteSize()/sizeof(float), value);
    }
};
