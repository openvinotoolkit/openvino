// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <exception>
#include <string>
#include <map>
#include <list>
#include <memory>
#include <utility>
#include <vector>
#include <random>

#include <gtest/gtest.h>
#include <legacy/graph_tools.hpp>
#include <ngraph/function.hpp>
#include <ie_precision.hpp>
#include <ie_blob.h>
#include <ie_plugin_config.hpp>
#include <cpp/ie_cnn_network.h>

#include <backend/dnn_types.h>
#include <backend/gna_types.h>
#include <gna/gna_config.hpp>
#include <gna_plugin.hpp>
#include <gna_lib_ver_selector.hpp>

#include "test_irs.hpp"

#define withConfig(key, value) withGNAConfig(GNA_CONFIG_KEY(key), value)
#define ASSERT_NO_THROW_IE_EXCEPTION(expr) \
do {\
try {\
expr;\
}catch(std::exception & e) {\
    FAIL() << e.what();\
}\
catch(...) {\
    FAIL() << "unknown exception";\
}}while(false)

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
        saveArgs,
        matchInputData,
        fillOutputValues,
        matchAffineWeightsTranspose,
        matchAffineWeights,
        matchAffineWeightsSize,
        saveAffineWeights,
    };
    enum {
        kUnset = -1,
        kAnyNotNull= -2
    };
    struct  MatcherData {
        MatchWhat type = matchNone;
        int matchQuantity = kUnset;
    };
    std::vector<MatcherData> whatToMatch;

    int numberOfStates = kUnset;
    bool matchInserted = true;
    NnetPrecision nnet_precision;
    float quantization_presicion_threshold = 1.0f;
    uint16_t quantization_segments_threshold = UINT16_MAX;
    uint32_t type = 0;
    std::string model;
    std::shared_ptr<ngraph::Function> ngraph_model;
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
    std::vector<int16_t> input_processed;
    InferenceEngine::Precision input_precision = InferenceEngine::Precision::FP32;
    std::map<std::string, std::vector<float>> input_init;
    std::vector<std::vector<float>> expected_output;
    int16_t fillValue = 0;
    std::vector<float> weightsFillPattern;
    std::map<std::string, std::vector<float>> weightsByLayerFillPattern;
    std::pair<int, int> transposeArgs;
    std::pair<int, int> transposedArgsForSaving;
    std::vector<uint16_t>* transposedData;
    std::vector<DnnActivationType> pwlsToMatchWith;
    size_t matched_weight_size = 0;
    size_t nCopyLayersToMatch = -1;
};

class GNATestBase {
 public:
    virtual ~GNATestBase() = default;
 protected:

    #define USE_RANDOM_SEED 0
    static unsigned int const random_seed;

    template<typename T>
    std::vector<T> generate_random_1d(size_t a, T min, T max) {
        static std::default_random_engine generator(random_seed);
        std::uniform_real_distribution<T> distribution(min, max);
        std::vector<T> v(a);

        for (size_t i = 0; i < a; ++i) {
            v[i] = (T)distribution(generator);
        }
        return v;
    }
};

template <class T>
class GNATestConfigurability : public GNATestBase{
 protected:
    bool needNextMatcher = true;
    GnaPluginTestEnvironment _env;
    GnaPluginTestEnvironment::MatcherData& getMatcher() {
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
    T & withGNAConfig(const std::string &keyName, const VType &value) {
        std::stringstream ss;
        ss << value;
        _env.config[keyName] = ss.str();
        return *dynamic_cast<T*>(this);
    }
    T & onCPU() {
        _env.config[GNA_CONFIG_KEY(DEVICE_MODE)] = GNA_CONFIG_VALUE(SW_FP32);
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

    GNAPropagateMatcher & returns() {
        return *this;
    }

    GNAPropagateMatcher & that() {
        return *this;
    }

    GNAPropagateMatcher & result() {
        return *this;
    }

    GNAPropagateMatcher & called_with() {
        return *this;
    }

    GNAPropagateMatcher & called_without() {
        _env.matchInserted = false;
        return *this;
    }
    /**
     * @brief gna_propagate_forward will fill all output pointers of 16 bits with this value
     */
    GNAPropagateMatcher & filledWith(int16_t valueToFill) {
        _env.fillValue = valueToFill;
        getMatcher().type = GnaPluginTestEnvironment::fillOutputValues;
        return *this;
    }

    GNAPropagateMatcher & input(const std::string & inputName, const std::vector<float>& inputData) {
        _env.input_init[inputName] = inputData;
        return *this;
    }

    GNAPropagateMatcher & inputScale(const std::string & inputName, float scaleFactor) {
        _env.config[std::string(GNA_CONFIG_KEY(SCALE_FACTOR)) + "_" + inputName] = std::to_string(scaleFactor);
        return *this;
    }

    GNAPropagateMatcher & called_with_input_and_expected_output(const std::vector<float>& input_data,
                                                                const std::vector<float>& expect) {
        _env.matchOutput = true;
        _env.input_init["any_input_name"] = input_data;
        _env.expected_output .push_back(expect);
        return *this;
    }

    GNAPropagateMatcher &  called_with_input(std::vector<float>& input_data) {
        _env.input_init["any_input_name"] = input_data;
        return *this;
    }

    GNAPropagateMatcher &  equals_to(std::vector<float>& expect) {
        _env.matchOutput = true;
        _env.expected_output.push_back(expect);
        return *this;
    }

    GNAPropagateMatcher & once() {
        return times(1);
    }

    GNAPropagateMatcher & twice() {
        return times(2);
    }

    GNAPropagateMatcher & times(int n) {
        getMatcher().matchQuantity = n;
        return *this;
    }

    GNAPropagateMatcher & args(std::string args) {
        return *this;
    }

    GNAPropagateMatcher & exact_nnet_structure(gna_nnet_type_t * pNet) {

        getMatcher().type = GnaPluginTestEnvironment::exactNNetStructure;
        original_nnet = pNet;
        return *this;
    }

    GNAPropagateMatcher & pwl_inserted_into_nnet() {
        getMatcher().type = GnaPluginTestEnvironment::matchPwlInserted;
        return *this;
    }

    GNAPropagateMatcher & pwls_inserted_into_nnet(const std::vector<DnnActivationType> &pwls) {
        getMatcher().type = GnaPluginTestEnvironment::matchPwlInserted;
        _env.pwlsToMatchWith = pwls;
        return *this;
    }

    GNAPropagateMatcher & max_pooling_inserted_into_nnet() {
        getMatcher().type = GnaPluginTestEnvironment::matchMaxPoolingInserted;
        return *this;
    }

    GNAPropagateMatcher & succeed() {
        return *this;
    }

    GNAPropagateMatcher & convolution_inserted_into_nnet() {
        getMatcher().type = GnaPluginTestEnvironment::matchConvInserted;
        return *this;
    }


    GNAPropagateMatcher & pwl_quantization_activation(uint32_t activation_type) {
        getMatcher().type = GnaPluginTestEnvironment::matchPwlQuantizeMetrics;
        _env.type = activation_type;
        return *this;
    }

    GNAPropagateMatcher & pwl_quantization_precision_threshold(float threshold) {
        getMatcher().type = GnaPluginTestEnvironment::matchPwlQuantizeMetrics;
        _env.quantization_presicion_threshold = threshold;
        return *this;
    }

    GNAPropagateMatcher & pwl_quantization_segments_threshold(uint16_t threshold) {
        getMatcher().type = GnaPluginTestEnvironment::matchPwlQuantizeMetrics;
        _env.quantization_segments_threshold = threshold;
        return *this;
    }

    GNAPropagateMatcher & diagonal_inserted_into_nnet() {
        getMatcher().type = GnaPluginTestEnvironment::matchDiagonalInserted;
        return *this;
    }

    GNAPropagateMatcher &preprocessed_input_data(std::vector<float> input_init, std::vector<int16_t> input_processed,
                                                 InferenceEngine::Precision inputPrecision) {
        getMatcher().type = GnaPluginTestEnvironment::matchInputData;
        _env.input_processed = std::move(input_processed);
        _env.input_init["placeholder"] = std::move(input_init);
        _env.input_precision = inputPrecision;
        return *this;
    }

    GNAPropagateMatcher & copy_inserted_into_nnet() {
        getMatcher().type = GnaPluginTestEnvironment::matchCopyInserted;
        return *this;
    }

    GNAPropagateMatcher & affine_weights_transpozed(std::pair<int, int> &&transpozedArgs) {
        getMatcher().type = GnaPluginTestEnvironment::saveAffineWeights;
        _env.transposedArgsForSaving = std::move(transpozedArgs);

        return *this;
    }

    GNAPropagateMatcher & affine_weights() {
        getMatcher().type = GnaPluginTestEnvironment::saveAffineWeights;
        return *this;
    }

    GNAPropagateMatcher & affine_weights_eq(std::vector<uint16_t> & sourceWeights) {
        getMatcher().type = GnaPluginTestEnvironment::matchAffineWeights;
        _env.transposedData = &sourceWeights;
        return *this;
    }


    GNAPropagateMatcher & affine_weights_transposed(std::vector<uint16_t> & sourceWeights, std::pair<int,int> transposeData) {
        getMatcher().type = GnaPluginTestEnvironment::matchAffineWeightsTranspose;
        _env.transposeArgs = transposeData;
        _env.transposedData = &sourceWeights;
        return *this;
    }

    GNAPropagateMatcher & nnet_input_precision(const InferenceEngine::Precision &precision) {
        getMatcher().type = GnaPluginTestEnvironment::matchPrecision;
        _env.nnet_precision.input_precision = precision;
        return *this;
    }
    GNAPropagateMatcher & nnet_ouput_precision(const InferenceEngine::Precision &precision) {
        getMatcher().type = GnaPluginTestEnvironment::matchPrecision;
        _env.nnet_precision.output_precision = precision;
        return *this;
    }
    GNAPropagateMatcher & nnet_weights_precision(const InferenceEngine::Precision &precision) {
        getMatcher().type = GnaPluginTestEnvironment::matchPrecision;
        _env.nnet_precision.weights_precision = precision;
        return *this;
    }
    GNAPropagateMatcher & nnet_biases_precision(const InferenceEngine::Precision &precision) {
        getMatcher().type = GnaPluginTestEnvironment::matchPrecision;
        _env.nnet_precision.biases_precision = precision;
        return *this;
    }

    GNAPropagateMatcher & proc_type(uint32_t proc_type) {
        getMatcher().type = GnaPluginTestEnvironment::matchProcType;
        _env.proc_type = proc_type;
        return * this;
    }

    GNAPropagateMatcher & to(gna_nnet_type_t *savedNet) {
        this->savedNet = savedNet;
        return *this;
    }

    GNAPropagateMatcher & to(std::vector<uint16_t> & sourceWeights) {
        _env.transposedData = &sourceWeights;
        return *this;
    }

 protected:
    void match();
    gna_nnet_type_t * original_nnet = nullptr;
    gna_nnet_type_t * savedNet = nullptr;
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
    void load(std::shared_ptr<GNAPluginNS::GNAPlugin> & plugin);
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
 * @brief weights matcher has specific weights matching methods
 */
class GNAWeightsMatcher : public GNAPropagateMatcher {
 public:
    using base = GNAPropagateMatcher;
    using base::base;

    GNAWeightsMatcher & size() {
        getMatcher().type = GnaPluginTestEnvironment::matchAffineWeightsSize;
        return *this;
    }
    GNAWeightsMatcher & equals_to(size_t weights_size) {
        if (getMatcher().type == GnaPluginTestEnvironment::matchAffineWeightsSize) {
            _env.matched_weight_size = weights_size;
        }
        return *this;
    }
};



/**
 * @brief base for test fixture
 */
template <class U = ::testing::Test>
class GNATest : public U, public GNATestConfigurability<GNATest<U>>  {
    using base = GNATestConfigurability<GNATest<U>>;
    using base::_env;
    class XStorage {
     public:
        std::vector<uint8_t> data;
        std::function<void (void *)> destroyer;
       ~XStorage() {
           destroyer(&data.front());
       }
    };
    std::list<XStorage> dataUsedInMatchers;
    std::list<std::shared_ptr<GNATestBase>> returnedMatchers;

 public:
    template <class T>
    T & storage () {
        dataUsedInMatchers.push_back({std::vector<uint8_t >(sizeof(T)), [](void * toDestroy) {
            reinterpret_cast<T*>(toDestroy)->~T();
        }});

        auto ptr = reinterpret_cast<T*> (&dataUsedInMatchers.back().data.front());
        // sad to say we are not using destructors here so data might leak
        new(ptr) T;

        return *ptr;
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
        base::getMatcher().type = GnaPluginTestEnvironment::saveArgs;
        return *this;
    }
    GNATest & save() {
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
    GNATest & afterLoadingModel(std::shared_ptr<ngraph::Function> ngraph_model) {
        _env.ngraph_model = ngraph_model;
        return *this;
    }

    GNAWeightsMatcher & affine_weights() {
        returnedMatchers.push_back(std::make_shared<GNAWeightsMatcher>(_env));
        _env = GnaPluginTestEnvironment();
        return dynamic_cast<GNAWeightsMatcher&>(*returnedMatchers.back());
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

    GNATest & onInferNgraphModel(std::shared_ptr<ngraph::Function> ngraph_model) {
        _env.ngraph_model = ngraph_model;
        return *this;
    }

    GNATest &  withWeigthsPattern(std::string layerName, std::vector<float> && initializer) {
        _env.weightsByLayerFillPattern[layerName] = std::move(initializer);
        return *this;
    }

    GNATest &  withWeigthsPattern(std::vector<float> && initializer) {
        _env.weightsFillPattern = std::move(initializer);
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
};

void fillWeights(InferenceEngine::Blob::Ptr weights, std::vector<float> pattern = {1.f});
