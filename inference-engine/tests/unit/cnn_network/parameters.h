// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <queue>
#include <random>

enum class ParametersValues {
    ZERO,
    INT_POSITIVE,
    INT_NEGATIVE,
    FLOAT_POSITIVE,
    FLOAT_NEGATIVE,
    STRING
};
enum class ParameterRange {
    SET,
    SINGLE
};
using GoodBadParams = std::pair<std::vector<ParametersValues>, std::vector<ParametersValues>>;
using Params = std::map<std::string, std::pair<ParameterRange, GoodBadParams>>;

Params operator + (const Params& val1, const Params& val2) {
    Params result;
    result.insert(val1.begin(), val1.end());
    result.insert(val2.begin(), val2.end());
    return result;
}

class Parameters {
private:
    // Common for Convolution, Deconvolution, Pooling layers
    Params common {
            // Parameter name, range, type of good values, type of bad
            {"stride-x", {ParameterRange::SINGLE,
                                                  {{ParametersValues::INT_POSITIVE},
                                                                                                             {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"stride-y", {ParameterRange::SINGLE,
                                                  {{ParametersValues::INT_POSITIVE},
                                                                                                             {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"kernel-x", {ParameterRange::SINGLE,
                                                  {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                                                                                             {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"kernel-y", {ParameterRange::SINGLE, {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE}, {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"pad-x", {ParameterRange::SINGLE,
                                                  {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                                                                                             {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"pad-y", {ParameterRange::SINGLE,
                                                  {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                                                                                             {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}}
    };
    Params conv {
            // Parameter name, range, type of good values, type of bad
            {"dilation-x", {ParameterRange::SINGLE,
                                   {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                           {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"dilation-y", {ParameterRange::SINGLE,
                                   {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                           {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"output", {ParameterRange::SINGLE,
                                   {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                           {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"group", {ParameterRange::SINGLE,
                                   {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                           {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
    };
    Params pooling {
            // Parameter name, range, type of good values, type of bad
            {"pool-method", {ParameterRange::SINGLE,
                                    {{ParametersValues::STRING},
                                            {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"exclude-pad", {ParameterRange::SINGLE,
                                    {{ParametersValues::STRING},
                                            {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}}
    };
    Params detectionOutput {
            // Parameter name, range, type of good values, type of bad
            {"num_classes", {ParameterRange::SINGLE,
                                    {{ParametersValues::INT_POSITIVE},
                                            {ParametersValues::ZERO, ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"background_label_id", {ParameterRange::SINGLE,
                                    {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                            {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"top_k", {ParameterRange::SINGLE,
                                    {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                            {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"variance_encoded_in_target", {ParameterRange::SINGLE,
                                    {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                            {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"keep_top_k", {ParameterRange::SINGLE,
                                    {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                            {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"num_orient_classes", {ParameterRange::SINGLE,
                                    {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                            {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"code_type", {ParameterRange::SINGLE,
                                    {{ParametersValues::STRING},
                                            {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"share_location", {ParameterRange::SINGLE,
                                    {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                            {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"interpolate_orientation", {ParameterRange::SINGLE,
                                    {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                            {ParametersValues::STRING}}}},
            {"nms_threshold", {ParameterRange::SINGLE,
                                    {{ParametersValues::FLOAT_POSITIVE},
                                            {ParametersValues::FLOAT_NEGATIVE, ParametersValues::STRING}}}},
            {"confidence_threshold", {ParameterRange::SINGLE,
                                    {{ParametersValues::FLOAT_POSITIVE},
                                            {ParametersValues::FLOAT_NEGATIVE, ParametersValues::STRING}}}}
    };
    Params crop {
            {"axis", {ParameterRange::SET,
                             {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                     {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"offset", {ParameterRange::SET,
                             {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                     {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"dim", {ParameterRange::SET,
                             {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                     {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"crop_begin", {ParameterRange::SET,
                             {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                     {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"crop_end", {ParameterRange::SET,
                             {{ParametersValues::ZERO, ParametersValues::INT_POSITIVE},
                                     {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
    };
    Params interp {
            {"height", {ParameterRange::SINGLE,
                               {{ParametersValues::INT_POSITIVE, ParametersValues::ZERO},
                                       {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
            {"factor", {ParameterRange::SINGLE,
                               {{ParametersValues::FLOAT_POSITIVE},
                                       {ParametersValues::ZERO, ParametersValues::FLOAT_NEGATIVE, ParametersValues::STRING}}}},
            {"shrink_factor", {ParameterRange::SINGLE,
                               {{ParametersValues::FLOAT_POSITIVE},
                                       {ParametersValues::ZERO, ParametersValues::FLOAT_NEGATIVE, ParametersValues::STRING}}}},
            {"zoom_factor", {ParameterRange::SINGLE,
                               {{ParametersValues::FLOAT_POSITIVE},
                                       {ParametersValues::ZERO, ParametersValues::FLOAT_NEGATIVE, ParametersValues::STRING}}}},
            {"width", {ParameterRange::SINGLE,
                               {{ParametersValues::INT_POSITIVE, ParametersValues::ZERO},
                                       {ParametersValues::INT_NEGATIVE, ParametersValues::STRING}}}},
    };
    std::map<std::string, std::map<std::string, std::vector<std::string>>> stringParams {
            {"Eltwise", {{"operation", {"sum", "max", "mul"}}}},
            {"LRN", {{"region", {"across", "same"}}}},
            {"Activation", {{"type", {"sigmoid", "tanh", "elu", "relu6"}}}},
            {"Pooling", {{"pool_method", {"max", "avg"}}, {"exlude_pad", {"true", "false"}}}},
            {"Resample", {{"type", {"caffe.ResampleParameter.LINEAR", "caffe.ResampleParameter.CUBIC",
                                    "caffe.ResampleParameter.NEAREST"}}}},
            {"DetectionOutput", {{"code_type", {"caffe.PriorBoxParameter.CENTER_SIZE", "caffe.PriorBoxParameter.CORNER"}}}}
    };
    std::map<std::string, Params> layerParamsNames {
            // Layer name, Corresponding params names
            {"Convolution", common + conv},
            {"Deconvolution", common + conv},
            {"Pooling", common + pooling},
            {"DetectionOutput", detectionOutput},
            {"Crop", crop},
            {"Interp", interp}
    };
    const int zero = 0;
    std::string type;
    std::mt19937 gen;
    std::uniform_int_distribution<int> distIntPositive;
    std::uniform_int_distribution<int> distIntNegative;
    std::uniform_real_distribution<float> distFloatNegative;
    std::uniform_real_distribution<float> distFloatPositive;
    std::queue<std::string> paramWasInvalid;
public:
    Parameters() {}
    Parameters(const std::string& type) : gen(static_cast<unsigned long>(std::chrono::system_clock::now().time_since_epoch().count())),
                                          distIntPositive(1, 100),
                                          distIntNegative(-100, -1),
                                          distFloatNegative(-10.0, -0.1),
                                          distFloatPositive(0.1, 10.0) {
        this->type = type;
        Params param = getParametersByLayerName();
        for (auto iter : param) {
            paramWasInvalid.push(iter.first);
        }
    }
    Params getParametersByLayerName() {
        return layerParamsNames[type];
    }

    std::vector<std::string> getDifferentParamValues(const std::vector<ParametersValues>& valuesType,
                                                     const std::vector<std::string>& stringValues) {
        int magicNumber = 10;
        std::vector<std::string> paramsValues = {};
        for (auto i : valuesType) {
            switch(i) {
                case ParametersValues::ZERO: {
                    paramsValues.push_back("0");
                    break;
                }
                case ParametersValues::INT_POSITIVE: {
                    for (int j = 0; j < magicNumber; ++j) {
                        paramsValues.push_back(std::to_string(distIntPositive(gen)));
                    }
                    break;
                }
                case ParametersValues::INT_NEGATIVE: {
                    for (int j = 0; j < magicNumber; ++j) {
                        paramsValues.push_back(std::to_string(distIntNegative(gen)));
                    }
                    break;
                }
                case ParametersValues::FLOAT_POSITIVE: {
                    for (int j = 0; j < magicNumber; ++j) {
                        paramsValues.push_back(to_string_c_locale(distFloatPositive(gen)));
                    }
                    break;
                }
                case ParametersValues::FLOAT_NEGATIVE: {
                    for (int j = 0; j < magicNumber; ++j) {
                        paramsValues.push_back(to_string_c_locale(distFloatNegative(gen)));
                    }
                    break;
                }
                case ParametersValues::STRING: {
                    paramsValues.insert(paramsValues.begin(), stringValues.begin(), stringValues.end());
                    break;
                }
            }
        }

        return  paramsValues;
    }

    std::map<std::string, std::string> getValidParameters() {
        Params param = getParametersByLayerName();
        std::map<std::string, std::string> params;
        for (auto i : param) {
            params[i.first] = getCorrectParamValue(i.second, i.first);
        }
        return params;
    }

    std::string getCorrectParamValue(const std::pair<ParameterRange, GoodBadParams>& values,
                                     const std::string& paramName) {
        std::string parameter = "";
        ParameterRange howMany = values.first;
        std::vector<ParametersValues> valuesType = values.second.first;

        std::vector<std::string> paramsValues = getDifferentParamValues(valuesType, stringParams[type][paramName]);

        std::uniform_int_distribution<int> indexesDist(0, static_cast<int>(paramsValues.size() - 1));
        if (howMany == ParameterRange::SINGLE) {
            int index = indexesDist(gen);
            parameter = paramsValues[index];
        } else {
            int numOfDigits = indexesDist(gen);
            for (int i = 0; i < numOfDigits; i++) {
                parameter += paramsValues[i] + ", ";
            }
        }
        return parameter;
    }

    std::string getIncorrectParamValue(const std::pair<ParameterRange, GoodBadParams>& values) {
        std::string parameter = "";
        std::vector<ParametersValues> valuesType = values.second.second;

        std::vector<std::string> paramsValues = getDifferentParamValues(valuesType, {"foo", "bar"});
        std::uniform_int_distribution<int> indexesDist(0, static_cast<int>(paramsValues.size() - 1));
        int index = indexesDist(gen);
        parameter = paramsValues[index];

        return parameter;
    }

    std::map<std::string, std::string> getInvalidParameters() {
        std::map<std::string, std::string> params = getValidParameters();

        std::string paramName = paramWasInvalid.front();
        paramWasInvalid.pop();
        params[paramName] = getIncorrectParamValue(layerParamsNames[type][paramName]);
        return params;
    }

    int getNumOfParameters() {
        return static_cast<int>(layerParamsNames[type].size());
    }
};
