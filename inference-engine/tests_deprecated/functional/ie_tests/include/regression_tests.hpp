// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <tests_common.hpp>
#include <tests_file_utils.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <iterator>

#include <streambuf>

#include <format_reader_ptr.h>

#include "regression_reference.hpp"
#include "regression_config.hpp"

#include "net_model.hpp"
#include "segmentation_matcher.hpp"
#include "custom_matcher.hpp"
#include "raw_matcher.hpp"
#include "classification_matcher.hpp"
#include "object_detection_matcher.hpp"
#include "optimized_network_matcher.hpp"

#include "functional_test_utils/plugin_cache.hpp"

#ifdef near
#undef near
#endif


namespace Regression {
using namespace Matchers;

/**
 * @brief wether to reset plugin after feeding this input, default is false.
 */

#define afterReset(ACTOR) setCustomInput([&](const Regression::InferenceContext & _) -> \
    Regression::RegressionConfig::InputFetcherResult{return {true, false, false};})

#define withCustomInput(ACTOR) setCustomInput([&](const Regression::InferenceContext & _) -> \
    Regression::RegressionConfig::InputFetcherResult{ACTOR; return {};})

#define withCustomOutput(ACTOR) setCustomOutput([&](const Regression::InferenceContext & _){ACTOR;})
#define withCustomModel(ACTOR) setCustomModel([&](const Regression::InferenceContext & _){ACTOR;})


enum EMean {
    eNo,
    eValues,
    eImage
};

static std::string format_mean(EMean isMean) {
    switch (isMean) {
        case eNo:return "_no_mean";

        case eImage:return "_mf";

        case eValues:return "";
    }
    return nullptr;
}

inline std::ostream &operator<<(std::ostream &os, EMean mean) {
    return os << format_mean(mean);
}

template<typename M>
class ModelSelector {

    template<typename T>
    friend class ModelSelector; // every B<T> is a friend of A


    enum EPrecision {
        eq78, efp32, efp16, ei16, ei8
    };

    enum EGroup {
        eNoGroup, eGroup
    };


    static std::string format_precision(EPrecision precision) {
        switch (precision) {
            case efp32:return "fp32";

            case eq78:return "q78";

            case efp16:return "fp16";

            case ei16:return "i16";

            case ei8: return "i8";
        }
        return nullptr;
    }

    static std::string format_group(EGroup isGroup) {
        switch (isGroup) {
            case eNoGroup:return "";

            case eGroup:return "_group";
        }
        return nullptr;
    }

    friend std::ostream &operator<<(std::ostream &os, EPrecision precision) {
        return os << format_precision(precision);
    }

    friend std::ostream &operator<<(std::ostream &os, EGroup group) {
        return os << format_group(group);
    }


    Model model, statFile;
    RegressionConfig config;
    EMean isMean = eValues;
    EPrecision precision = eq78;
    EGroup isGroup = eNoGroup;

 private:
    std::string prepareModelMatching() {
        std::stringstream path_to_input;
        path_to_input << TestDataHelpers::get_data_path();
        path_to_input << kPathSeparator
                      << model.resolution() << kPathSeparator;
        for (auto & fileName : config._paths_to_images) {
            fileName = path_to_input.str() + fileName;
        }

        if (model.folderName().empty() || model.fileName().empty()) {
            return "";
        }
        ModelsPath path_to_model;
        std::stringstream prc;
        path_to_model << kPathSeparator
                      << model.folderName() << kPathSeparator
                      << model.fileName() << "_" << precision << isMean << isGroup << "." << model.extension();

        return path_to_model.str();
    }

    std::string prepareStatMatching() {
        if (statFile.fileName() == "") return "";
        ModelsPath path_to_stat;
        path_to_stat << kPathSeparator
                      << statFile.folderName() << kPathSeparator
                      << statFile.fileName();

        return path_to_stat.str();
    }

    ModelSelector() = default;

    std::string getReferenceResultsLabel() {
        std::stringstream ss;
        for (auto&& v: config.ie_core->GetVersions(config._device_name)) {
            const InferenceEngine::Version& version = v.second;
            if (nullptr != version.description) {
                ss << version.description;
                break;
            }
        }
        std::string pluginName = ss.str();
        if (pluginName.empty())
            std::cerr << "getReferenceResultsLabel() failed for device: \"" << config._device_name << "\"" << std::endl;

        return pluginName + "_" + model.folderName() + format_mean(isMean)
                 + "_" + format_precision(precision) + format_group(isGroup);
    }

    bool loadBlobFile(const char* fname, std::vector<char>& outData)
    {
        if (!fname)
            return false;
        FILE *f = fopen(fname, "rb");
        if (!f) {
            return false;
        }
        fseek(f, 0, SEEK_END);
        int fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        outData.resize(fsize);
        size_t bytesRead = fread(outData.data(), 1, fsize, f);
        if (bytesRead != fsize) {
            std::cout << "cannot read file" << std::endl;
            return false;
        }
        fclose(f);

        return true;
    }
 public :

    explicit ModelSelector(const RegressionConfig& config) : config(config) {}

    template <class T>
    explicit ModelSelector(T *oldSelector) {
        config = oldSelector->config;
    }

    ModelSelector &And(const std::string &fileName) {
        config._paths_to_images.push_back(fileName);
        return *this;
    }

    ModelSelector &And(const std::vector<std::string> &filesNamesVector) {
        config._paths_to_images.insert(config._paths_to_images.end(), filesNamesVector.begin(), filesNamesVector.end());
        return *this;
    }

    ModelSelector &on(const std::string &fileName) {
        config._paths_to_images.push_back(fileName);
        return *this;
    }

    ModelSelector &print(const std::size_t printNum = 10) {
        config.print = true;
        config.printNum = printNum;
        return *this;
    }

    ModelSelector &useExportImport() {
        config.useExportImport = true;
        return *this;
    }

    /// @breif - tile last batch
    ModelSelector &onN_infers(int nTimesCopyInputImages) {
        if (config._paths_to_images.size() != config.batchSize) {
            THROW_IE_EXCEPTION << "number of input images:"
                               << config._paths_to_images.size() << " not equal to batch size: " << config.batchSize;
        }
        auto first_image =  config._paths_to_images.end();
        std::advance(first_image, -config.batchSize);

        std::vector<std::string> data_for_last_infer(first_image, config._paths_to_images.end());

        for (;nTimesCopyInputImages > 0; nTimesCopyInputImages--) {
            config._paths_to_images.insert(config._paths_to_images.end(), data_for_last_infer.begin(), data_for_last_infer.end());
        }
        return *this;
    }
    /**
     * @brief - tile last input image
     * @param nTimesCopyLastImagePlusOne = number of times last image will be tiled + 1
     * @deprecated
     */
    ModelSelector &times(int nTimesCopyLastImagePlusOne) {
        tile(nTimesCopyLastImagePlusOne - 1);
        return *this;
    }
    /**
     * @brief - tile last input image
     * @param nTimesCopyLastImage = number of times last image will be tiled
     * @deprecated
     */
    ModelSelector &tile(int nTimesCopyLastImage) {
        if (config._paths_to_images.empty()) {
            return *this;
        }
        auto lastImage = config._paths_to_images.back();
        for (;nTimesCopyLastImage > 0; nTimesCopyLastImage--) {
            config._paths_to_images.push_back(lastImage);
        }
        return *this;
    }

    ModelSelector &onModel(
        std::string _folderName,
        std::string _fileName,
        std::string _resolutionName) {
        model = {_folderName, _fileName, _resolutionName};
        return *this;
    }

    ModelSelector &onArkInput() {
        model = {model.folderName(), model.fileName(), "ark"};
        return *this;
    }

    ModelSelector &onFP32() {
        precision = efp32;
        config.modelPrecision = Precision::FP32;
        return *this;
    }

    ModelSelector &onI16() {
        precision = ei16;
        config.modelPrecision = Precision::I16;
        return *this;
    }

    ModelSelector &onFP16() {
        precision = efp16;
        config.modelPrecision = Precision::FP16;
        return *this;
    }

    ModelSelector &onQ78() {
        precision = eq78;
        config.modelPrecision = Precision::Q78;
        return *this;
    }

    ModelSelector& onI8() {
        precision = ei8;
        config.modelPrecision = Precision::I8;
        return *this;
    }

    ModelSelector &withInputPrecision(InferenceEngine::Precision p) {
        config._inputPrecision = p;
        return *this;
    }

    ModelSelector &withOutputPrecision(InferenceEngine::Precision p) {
        config._outputPrecision = p;
        return *this;
    }

    ModelSelector &withOutputPrecision(std::map<std::string, InferenceEngine::Precision> p) {
        static_assert(std::is_same<M, RawMatcher>::value, "Output precision per blob implemented only in RawMatcher");
        config._outputBlobPrecision = p;
        return *this;
    }

    template <class Q = M>
    typename enable_if<std::is_base_of<OptimizedNetworkDumper, Q>::value, bool>::type
    needInput() const {
        return false;
    }

    template <class Q = M>
    typename enable_if<!std::is_base_of<OptimizedNetworkDumper, Q>::value, bool>::type
    needInput() const {
        return true;
    }

    ModelSelector &withBatch() {
        config.batchMode = true;
        return *this;
    }

    ModelSelector &withBatch(int nBatchSize) {
        config.batchSize = nBatchSize;
        // assumption made that inputs already gets provided to matcher
        if (config._paths_to_images.empty() && needInput()) {
            THROW_IE_EXCEPTION << "withBatch token should follow after setting up inputs";
        }
        if (config._paths_to_images.size() < nBatchSize) {
            tile(nBatchSize - config._paths_to_images.size());
        }

        return *this;
    }

    ModelSelector &withDynBatch(int nLimit, int nBatchSize) {
        config.batchMode = true;
        config.useDynamicBatching = true;
        config.batchSize = nLimit;
        config.dynBatch = nBatchSize;
        return *this;
    }

    ModelSelector &withAsyncInferRequests(int nRequests) {
        config._nrequests = nRequests;
        return *this;
    }

    ModelSelector &onMultipleNetworks(int nNetworks) {
        config._numNetworks = nNetworks;
        return *this;
    }

    ModelSelector &setMean(EMean mean) {
        isMean = mean;
        return *this;
    }

    ModelSelector &withoutMean() {
        isMean = eNo;
        return *this;
    }

    ModelSelector &withMeanValues() {
        isMean = eValues;
        return *this;
    }

    ModelSelector &withMeanImage() {
        isMean = eImage;
        return *this;
    }

    ModelSelector &withGroup() {
        isGroup = eGroup;
        return *this;
    }

    ModelSelector withTopK(int topKNumbers) {
        config.topKNumbers = topKNumbers;
        return *this;
    }

    ModelSelector &withPluginConfig(const std::map<std::string, std::string> & plugin_config) {
        config.plugin_config = plugin_config;
        return *this;
    }

    ModelSelector &addPluginConfig(const std::map<std::string, std::string> & plugin_config) {
        config.plugin_config.insert(plugin_config.begin(), plugin_config.end());
        return *this;
    }

    ModelSelector &withPluginConfigOption(std::string key, std::string value) {
        config.plugin_config[key] = value;
        return *this;
    }

    ModelSelector & withImportedExecutableNetworkFrom(std::string location) {
        config._path_to_aot_model = location;
        return *this;
    }

    template <class T>
    ModelSelector &modifyConfig(const T & modifier) {
        modifier(config);
        return *this;
    }

    ModelSelector & usingAsync() {
        config.isAsync = true;
        return *this;
    }

    ModelSelector &fromLayer(const std::string & name) {
        config.outputLayer = name;
        return *this;
    }

    ModelSelector& doReshape(bool reshape = true) {
        config._reshape = reshape;
        return *this;
    }

    // type define when class in one of building method converted to new one or not
#define CUSTOM_TYPE\
    typename std::conditional<std::is_base_of<CustomMatcher, M>::value,\
    ModelSelector<M>&,\
    ModelSelector<CustomMatcher>>::type

 private :
    template <class A, class Q = M>
    typename enable_if<std::is_base_of<CustomMatcher, Q>::value, CUSTOM_TYPE>::type modify_config(const A& action) {
        action(config);
        return *this;
    }

    template <class A, class Q = M>
    typename enable_if<!std::is_base_of<CustomMatcher, Q>::value, CUSTOM_TYPE>::type modify_config(const A& action) {
        ModelSelector<CustomMatcher> newSelector(this);
        action(newSelector.config);
        return newSelector;
    }

 public:

    template <class T>
    CUSTOM_TYPE  setCustomModel(const T& model_maker) {
        return modify_config([&](RegressionConfig & this_config) {
            this_config.make_model = model_maker;
        });
    }

    template <class T>
    CUSTOM_TYPE setCustomInput(const T & fetcher) {
        return modify_config([&](RegressionConfig & this_config) {
            this_config.fetch_input.push_back(fetcher);
        });
    }

    template <class T>
    CUSTOM_TYPE setCustomOutput(const T & fetcher) {
        return modify_config([&](RegressionConfig & this_config) {
            this_config.fetch_result = fetcher;
        });
    }

    template <class T >
    M equalsTo(const std::initializer_list<T> & rhs) {
        config.referenceOutput.insert(config.referenceOutput.end(), rhs.begin(), rhs.end());
        return near(0.0);
    }

    template <class T >
    M near(double nearValue, const TBlob<T> & rhs) {
        config.nearValue = nearValue;
        for (const auto & v : rhs) {
            config.referenceOutput.push_back(v);
        }
        config._path_to_models = prepareModelMatching();
        config._stat_file = prepareStatMatching();
        return M(config);
    }

    M to(Blob::Ptr rhs) {
        config.outputBlob = rhs;
        config._path_to_models = prepareModelMatching();
        config._stat_file = prepareStatMatching();
        return M(config);
    }


    template <class T >
    M near(double nearValue, const initializer_list<TBlob<T>> & rhs) {
        config.nearValue = nearValue;

        for (auto && frame : rhs) {
            for (auto && data : frame) {
                config.referenceOutput.push_back(data);
            }
        }
        config._path_to_models = prepareModelMatching();
        config._stat_file = prepareStatMatching();
        return M(config);
    }

    template <class T >
    M near_avg(double nearAvgValue, const TBlob<T> & rhs) {
        config.nearAvgValue = nearAvgValue;
        return near(0.0, rhs);
    }

    M near(double nearValue, double meanRelativeError = 0, double maxRelativeError = 0) {
        config.nearValue = nearValue;
        config.meanRelativeError = meanRelativeError;
        config.maxRelativeError = maxRelativeError;
        config._path_to_models = prepareModelMatching();
        config._stat_file = prepareStatMatching();
        return M(config);
    }

    void equalToReferenceWithDelta(double nearValue) {
        config.nearValue = nearValue;
        config._path_to_models = prepareModelMatching();
        config._stat_file = prepareStatMatching();
        M(config).to(getReferenceResultsLabel());
    }

    template <class T>
    M equalToReference(const TBlob<T> & rhs) {
        for (const auto & v : rhs) {
            config.referenceOutput.push_back(v);
        }
        config._path_to_models = prepareModelMatching();
        config._stat_file = prepareStatMatching();
        return M(config, true);
    }

    // place holder to run the matcher without providing any reference
    void possible() {
        config._path_to_models = prepareModelMatching();
        config._stat_file = prepareStatMatching();
        auto tmp = M(config);
        ASSERT_NO_FATAL_FAILURE(tmp.match());
    }
};

/**
 * @class PluginVersion
 * @brief A PluginVersion class stores plugin version and initialization status
 */
struct PluginVersion : public InferenceEngine::Version {
    bool initialized = false;

    explicit PluginVersion(const InferenceEngine::Version *ver) {
        if (nullptr == ver) {
            return;
        }
        InferenceEngine::Version::operator=(*ver);
        initialized = true;
    }

    operator bool() const noexcept {
        return initialized;
    }
};

class Builder {
private:
    std::shared_ptr<InferenceEngine::Core> ie;
    RegressionConfig config;

public:
    Builder(std::shared_ptr<InferenceEngine::Core> _ie) : ie(_ie) {
        config.ie_core = ie;

#ifndef NDEBUG
        auto devices = ie->GetAvailableDevices();
        std::cout << "Available devices (" << devices.size() << "):" << std::endl;
        for (auto&& d : devices) {
            std::cout << "Device: " << d << std::endl;
            for (auto&& v : ie->GetVersions(d))
                std::cout << "\t" << v.first << " : " << PluginVersion(&v.second) << std::endl;
        }
#endif
    }

    Builder & usingDevice(const std::string & device_name) {
        config._device_name = device_name;
        return *this;
    }

    Builder& setPerfInfo(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& map) {
        config.perfInfoPtr = &map;
        config.plugin_config[CONFIG_KEY(PERF_COUNT)] = CONFIG_VALUE(YES);
        return *this;
    }

    Builder& setDeviceMapping(const std::map<std::string, std::string> &deviceMapping) {
        config.deviceMapping = deviceMapping;
        return *this;
    }

    ModelSelector<ClassificationMatcher> classificationResults() {
        return ModelSelector<ClassificationMatcher>(config);
    }

    ModelSelector<ClassificationMatcher> classificationResultsFor(const std::vector<std::string> & input) {
        return ModelSelector<ClassificationMatcher>(config).And(input);
    }

    ModelSelector<OptimizedNetworkMatcher> dumpedOptimizedNetwork() {
        return ModelSelector<OptimizedNetworkMatcher>(config);
    }

    ModelSelector<OptimizedNetworkDumper> dumpOptimizedNetworkTo(const std::string & file) {
        config._path_to_aot_model = file;
        return ModelSelector<OptimizedNetworkDumper>(config);
    }

    ModelSelector<ClassificationMatcher> classificationResultsFor(const std::string &input = { }) {
        auto selector = ModelSelector<ClassificationMatcher>(config);
        if (!input.empty()) {
            selector.And(input);
        }
        return selector;
    }

    ModelSelector<SegmentationMatcher> segmentationResultsFor(const std::string &fileName) {
        return ModelSelector<SegmentationMatcher>(config).And(fileName);
    }
    ModelSelector<RawMatcher> rawResultsFor(const std::string &fileName) {
        return ModelSelector<RawMatcher>(config).And(fileName);
    }
    ModelSelector<ObjectDetectionMatcher> objectDetectionResultsFor(const std::string &fileName) {
        return ModelSelector<ObjectDetectionMatcher>(config).And(fileName);
    }
    ModelSelector<ObjectDetectionMatcher> objectDetectionResults() {
        return ModelSelector<ObjectDetectionMatcher>(config);
    }
    ModelSelector<ObjectDetectionMatcher> objectDetectionResultsFor(const vector<std::string> &filesNamesVector) {
        return ModelSelector<ObjectDetectionMatcher>(config).And(filesNamesVector);
    }
};

class RegressionTests : public TestsCommon {
public:
    // to force overload
    virtual std::string getDeviceName() const = 0;

    Builder please() {
        std::shared_ptr<Core> ie = PluginCache::get().ie(getDeviceName());
        Builder b(ie);
        b.usingDevice(getDeviceName());
        return b;
    }
};

}

#define assertThat() SCOPED_TRACE("");please()
#define saveAfterInfer() SCOPED_TRACE("");please()
