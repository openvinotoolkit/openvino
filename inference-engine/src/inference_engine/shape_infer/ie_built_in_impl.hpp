// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include <shape_infer/ie_ishape_infer_extension.hpp>
#include <description_buffer.hpp>

#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace InferenceEngine {

inline std::string GetParamAsString(const char* param, const std::map<std::string, std::string> & params) {
    auto it = params.find(param);
    if (it == params.end()) {
        THROW_IE_EXCEPTION << "No such parameter name '" << param << "'";
    }
    return (*it).second;
}

inline int GetParamAsInt(const char* param, const std::map<std::string, std::string> & params) {
    std::string val = GetParamAsString(param, params);
    try {
        return std::stoi(val);
    } catch (...) {
        THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer. Value "
                           << val << " cannot be casted to int.";
    }
}

inline bool GetParamAsBool(const char* param, const std::map<std::string, std::string> & params) {
    std::string val = GetParamAsString(param, params);
    std::string loweredCaseValue;
    std::transform(val.begin(), val.end(), std::back_inserter(loweredCaseValue), [](char value) {
        return static_cast<char>(std::tolower(value));
    });

    bool result = false;

    if (!(std::istringstream(loweredCaseValue) >> std::boolalpha >> result)) {
        // attempting parse using non alpha bool
        return (GetParamAsInt(param, params) != 0);
    }

    return result;
}

std::string GetParamAsString(const char* param, const char* def,
                             const std::map<std::string, std::string> & params) {
    auto it = params.find(param);
    if (it == params.end() || it->second.empty()) {
        return def;
    }
    return (*it).second;
}

int GetParamAsInt(const char* param, int def,
                  const std::map<std::string, std::string> & params) {
    std::string val = GetParamAsString(param, std::to_string(def).c_str(), params);
    try {
        return std::stoi(val);
    } catch (...) {
        THROW_IE_EXCEPTION << "Cannot parse parameter " << param << " from IR for layer. Value "
                           << val << " cannot be casted to int.";
    }
}

bool GetParamAsBool(const char* param, bool def,
                    const std::map<std::string, std::string> & params) {
    std::string val = GetParamAsString(param, std::to_string(def).c_str(), params);
    std::string loweredCaseValue;
    std::transform(val.begin(), val.end(), std::back_inserter(loweredCaseValue), [](char value) {
        return static_cast<char>(std::tolower(value));
    });

    bool result = false;

    if (!(std::istringstream(loweredCaseValue) >> std::boolalpha >> result)) {
        // attempting parse using non alpha bool
        return (GetParamAsInt(param, def, params) != 0);
    }

    return result;
}

inline unsigned int GetParamAsUInt(const char* param, const std::map<std::string, std::string> & params) {
    std::string val = GetParamAsString(param, params);
    std::string message = "Cannot parse parameter " + std::string(param) + " from IR for layer" +
                          ". Value " + val + " cannot be casted to unsigned int.";
    try {
        int value = std::stoi(val);
        if (value < 0) {
            THROW_IE_EXCEPTION << message;
        }
        return static_cast<unsigned int>(value);
    } catch (...) {
        THROW_IE_EXCEPTION << message;
    }
}

namespace ShapeInfer {

/**
 * @brief Base class for all built-in shape infer implementations. Contains common logic with validators and errors
 * handling
 */
class BuiltInShapeInferImpl : public IShapeInferImpl {
public:
    explicit BuiltInShapeInferImpl(const std::string& type): _type(type) { }

    virtual void inferShapesImpl(const std::vector<Blob::CPtr>& inBlobs,
                                 const std::map<std::string, std::string>& params,
                                 const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes) = 0;

    StatusCode inferShapes(const std::vector<Blob::CPtr>& inBlobs, const std::map<std::string, std::string>& params,
                           const std::map<std::string, Blob::Ptr>& blobs, std::vector<SizeVector>& outShapes,
                           ResponseDesc* resp) noexcept override {
        inShapes.clear();
        for (const auto& blob : inBlobs) {
            inShapes.push_back(blob->getTensorDesc().getDims());
        }
        outShapes.clear();
        try {
            inferShapesImpl(inBlobs, params, blobs, outShapes);
            return OK;
        } catch (const std::exception& ex) {
            return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
        } catch (...) {
            return InferenceEngine::DescriptionBuffer(UNEXPECTED) << "Unknown error";
        }
    }

protected:
    std::string _type;
    std::vector<SizeVector> inShapes;
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
