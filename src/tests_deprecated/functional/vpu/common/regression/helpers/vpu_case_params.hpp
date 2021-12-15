// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include <ostream>
#include <functional>

using String2StringMap = std::map<std::string, std::string>;
using AdditionCfgParamsFactory = std::function<String2StringMap()>;

//------------------------------------------------------------------------------
// class SourceParameterBase
//------------------------------------------------------------------------------

class SourceParameterBase {
public:
    //Constructors
    SourceParameterBase() = default;
    virtual ~SourceParameterBase() = default;
    inline SourceParameterBase(
            std::string model_name,
            std::string& img_name,
            double reference_delta);

    // Accessors
    inline std::string modelName() const;
    inline std::string imageName() const;
    inline double referenceDelta() const;

    // Operations
    inline virtual std::string name() const;

protected:
    //Data section
    std::string model_name_;
    std::string img_name_;
    double reference_delta_;
};

//------------------------------------------------------------------------------
// class ClassificationSrcParam
//------------------------------------------------------------------------------

class ClassificationSrcParam : public SourceParameterBase {
public:
    //Constructors
    ClassificationSrcParam() = default;
    inline ClassificationSrcParam(
            std::string model_name,
            std::string img_name,
            double reference_delta);

    friend std::ostream& operator<<(std::ostream& os, const ClassificationSrcParam& param) {
        return os << param.modelName() << ", " << param.imageName() <<
        ", " << std::to_string(param.referenceDelta());
    }
};

//------------------------------------------------------------------------------
// class CompilationParameter
//------------------------------------------------------------------------------

class CompilationParameter {
public:
    //Constructors
    CompilationParameter() = default;
    inline CompilationParameter(std::string name,
                                std::string path_to_network,
                                std::string path_to_weights);
    //Accessors
    inline std::string name() const;
    inline std::string pathToNetwork() const;
    inline std::string pathToWeights() const;

    friend std::ostream& operator<<(std::ostream& os, const CompilationParameter& param) {
        return os << param.name();
    }

private:
    //Data section
    std::string name_;
    std::string path_to_network_;
    std::string path_to_weights_;
};

//------------------------------------------------------------------------------
// Implementation of inline methods of class SourceParameterBase
//------------------------------------------------------------------------------

inline SourceParameterBase::SourceParameterBase(
        std::string model_name,
        std::string& img_name,
        double reference_delta):
        model_name_(model_name),
        img_name_(img_name),
        reference_delta_(reference_delta) {
}

inline std::string SourceParameterBase::modelName() const {
    return model_name_;
}

inline std::string SourceParameterBase::imageName() const {
    return img_name_;
}

inline double SourceParameterBase::referenceDelta() const {
    return reference_delta_;
}

inline std::string SourceParameterBase::name() const {
    return "ModelName=" + model_name_ +
           "_ImageName=" + img_name_;
}

//------------------------------------------------------------------------------
// Implementation of inline methods of class ClassificationSrcParam
//------------------------------------------------------------------------------

inline ClassificationSrcParam::ClassificationSrcParam(
        std::string model_name,
        std::string img_name,
        double reference_delta):
        SourceParameterBase(model_name, img_name, reference_delta) {
}

//------------------------------------------------------------------------------
// Implementation of inline methods of class CompilationParameter
//------------------------------------------------------------------------------

inline CompilationParameter::CompilationParameter(
        std::string name,
        std::string path_to_network,
        std::string path_to_weights):
        name_(name),
        path_to_network_(path_to_network),
        path_to_weights_(path_to_weights) {
}

inline std::string CompilationParameter::name() const {
    return name_;
}

inline std::string CompilationParameter::pathToNetwork() const {
    return path_to_network_;
}

inline std::string CompilationParameter::pathToWeights() const {
    return path_to_weights_;
}
