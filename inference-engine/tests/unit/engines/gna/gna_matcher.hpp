//
// Copyright 2016-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
        saveArgs,
        matchInputData,
        fillOutputValues,
        matchAffineWeightsTranspose,
        matchAffineWeights,
        saveAffineWeights
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
    GNAPluginNS::Policy policy;
    bool matchThrows = false;
    uint32_t proc_type = static_cast<intel_gna_proc_t>(GNA_SOFTWARE & GNA_HARDWARE);
    std::string importedModelFileName;
    bool is_profiling_enabled = false;
    bool matchOutput = false;
    bool is_setup_of_omp_theads_expected = false;
    std::vector<int16_t> input_processed;
    InferenceEngine::Precision input_precision = InferenceEngine::Precision::FP32;
    std::map<std::string, std::vector<float>> input_init;
    std::vector<float> expected_output;
    int16_t fillValue = 0;
    std::vector<float> weightsFillPattern;
    std::pair<int, int> transposeArgs;
    std::pair<int, int> transposedArgsForSaving;
    std::vector<uint16_t>* transposedData;
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
    T & withGNAConfig(const std::string &keyName, const VType &value) {
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

    GNAPropagateMatcher & returns() {
        return *this;
    }

    GNAPropagateMatcher & And() {
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
        getMatcher() = GnaPluginTestEnvironment::fillOutputValues;
        return *this;
    }

    GNAPropagateMatcher & equal_to(const std::vector<float>& expect) {
        _env.matchOutput = true;
        _env.expected_output = expect;
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

    GNAPropagateMatcher &preprocessed_input_data(std::vector<float> input_init, std::vector<int16_t> input_processed,
                                                 InferenceEngine::Precision inputPrecision) {
        getMatcher() = GnaPluginTestEnvironment::matchInputData;
        _env.input_processed = std::move(input_processed);
        _env.input_init["placeholder"] = std::move(input_init);
        _env.input_precision = inputPrecision;
        return *this;
    }

    GNAPropagateMatcher & copy_inserted_into_nnet() {
        getMatcher() = GnaPluginTestEnvironment::matchCopyInserted;
        return *this;
    }


    GNAPropagateMatcher & affine_weights_transpozed(std::pair<int, int> &&transpozedArgs) {
        getMatcher() = GnaPluginTestEnvironment::saveAffineWeights;
        _env.transposedArgsForSaving = std::move(transpozedArgs);

        return *this;
    }

    GNAPropagateMatcher & affine_weights() {
        getMatcher() = GnaPluginTestEnvironment::saveAffineWeights;
        return *this;
    }

    GNAPropagateMatcher & affine_weights_eq(std::vector<uint16_t> & sourceWeights) {
        getMatcher() = GnaPluginTestEnvironment::matchAffineWeights;
        _env.transposedData = &sourceWeights;
        return *this;
    }


    GNAPropagateMatcher & affine_weights_transposed(std::vector<uint16_t> & sourceWeights, std::pair<int,int> transposeData) {
        getMatcher() = GnaPluginTestEnvironment::matchAffineWeightsTranspose;
        _env.transposeArgs = transposeData;
        _env.transposedData = &sourceWeights;
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

    GNAPropagateMatcher & to(std::vector<uint16_t> & sourceWeights) {
        _env.transposedData = &sourceWeights;
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
        getMatcher() = GnaPluginTestEnvironment::saveArgs;
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

    static void fillWeights(InferenceEngine::Blob::Ptr weights, std::vector<float> pattern = {1.f}) {
        float * p = weights->buffer().as<float *>();
        float * pEnd = p + weights->byteSize() / sizeof(float);

        for(; p!=pEnd ;) {
            for (int i = 0; i != (weights->byteSize() / sizeof(float) / 3) + 1; i++) {
                for (int j = 0; j != pattern.size() && p != pEnd; j++, p++) {
                    *p = pattern[j];
                }
            }
        }
    }
};
