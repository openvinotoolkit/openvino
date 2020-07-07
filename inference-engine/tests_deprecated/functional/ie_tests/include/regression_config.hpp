// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ie_blob.h>
#include <ie_precision.hpp>
#include <ie_core.hpp>

namespace Regression {
using namespace std;

class InferenceContext {
    InferenceEngine::BlobMap _inputs;
    std::vector<InferenceEngine::BlobMap> _outputs;
    std::vector<std::string> _fileNames;
    std::string _modelPath;
    InferenceEngine::Precision _prec;
    int _frameNumber = 0;
    int _inputIndex = 0;
 public:
    std::string modelFilePath() const {
        return _modelPath;
    }
    std::vector<std::string> fileNames() const {
        return _fileNames;
    }

    void setModelPath(const std::string &path) {
        _modelPath = path;
    }

    void setModelPrecision(InferenceEngine::Precision prec) {
        _prec = prec;
    }

    InferenceEngine::Precision getModelPrecision() const {
        return _prec;
    }

    void  setFileNames(const std::vector<std::string> fileNames)  {
        _fileNames = fileNames;
    }

    void setInput(std::string name, InferenceEngine::Blob::Ptr input) {
        _inputs[name] = input;
    }

    void setOutput(std::string name, InferenceEngine::Blob::Ptr output) {

        outputs()[name] = output;
    }

    InferenceEngine::Blob::Ptr getOutput(std::string name) {
        return outputs()[name];
    }

    const InferenceEngine::BlobMap& inputs() const {
        return _inputs;
    }

    const InferenceEngine::BlobMap& outputs() const {
        return const_cast<InferenceContext*>(this)->outputs();
    }

    std::vector<InferenceEngine::BlobMap>& allOutputs() {
        return _outputs;
    }

    InferenceEngine::BlobMap& outputs() {
        if (_outputs.empty()) {
            _outputs.push_back(InferenceEngine::BlobMap());
        }
        return _outputs.front();
    }

    InferenceEngine::BlobMap& newOutputs() {
        _outputs.push_back(InferenceEngine::BlobMap());
        return _outputs.back();
    }

    void setFrameNumber(int num) {
        _frameNumber = num;
    }

    int getFrameNumber() const {
        return _frameNumber;
    }

    void setInputIdx(int num) {
        _inputIndex = num;
    }

    size_t getInputIdx() const {
        return _inputIndex;
    }

    std::string currentInputFile() const {
        if (fileNames().empty()) {
            return "";
        }
        return fileNames()[std::min(getInputIdx(), fileNames().size()-1)];
    }

    const InferenceEngine::Blob::Ptr currentInputs() const {
        auto input = _inputs.begin();
        std::advance(input, getInputIdx());
        return input->second;
    }

};

struct RegressionConfig {
    struct InputFetcherResult {
        bool reset = false;
        bool fetchMore = false;
        bool fetched = true;
        bool hasResult = true;
        int frameNumber = 0;
        InputFetcherResult() = default;
        InputFetcherResult(bool reset, bool fetchMore=false, bool fetched=true, int frameNumber = 0, bool hasResult = true)
                : reset(reset), fetchMore(fetchMore), fetched(fetched), hasResult(hasResult) {}
    };
    using input_fetcher = std::function<InputFetcherResult (const InferenceContext & )>;
    using model_maker = std::function<void(const InferenceContext & )>;
    using result_fetcher = std::function<InferenceEngine::Blob::Ptr(const InferenceContext & )>;

    std::vector<input_fetcher> fetch_input;
    result_fetcher fetch_result;
    model_maker make_model;
    string _path_to_models;
    string _path_to_aot_model;
    vector<string> _paths_to_images;
    string _device_name;
    string _firmware;
    string _tmp_firmware;
    vector<string> labels;
    double nearValue = 0.0;
    double nearAvgValue = 0.0;
    double maxRelativeError = 0.0;
    double meanRelativeError = 0.0;
    bool batchMode = false;
    bool compactMode = true;
    bool int8Mode = false;
    bool isAsync = false;
    int batchSize = 1;
    //number of async infer requests to create
    int _nrequests = 1;
    int topKNumbers = -1;
    int _numNetworks = 1;

    bool useDynamicBatching = false;
    int dynBatch = -1;
    bool print = false;
    bool useExportImport = false;
    std::size_t printNum = 0;

    vector<float> referenceOutput;
    vector<uint8_t> referenceBin;

    InferenceEngine::Blob::Ptr outputBlob;
    std::string outputLayer;
    InferenceEngine::Precision _inputPrecision;
    InferenceEngine::Precision modelPrecision;
    InferenceEngine::Precision _outputPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::map<std::string, InferenceEngine::Precision> _outputBlobPrecision;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>* perfInfoPtr = nullptr;
    std::map<std::string, std::string> plugin_config;
    std::map<std::string, std::string> deviceMapping;

    std::shared_ptr<InferenceEngine::Core> ie_core;

    bool _reshape = false;
};

enum InputFormat {
    RGB = 0,
    BGR = 1
};

}
