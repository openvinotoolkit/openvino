/*
// Copyright (c) 2016-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "kernel_selector_common.h"

#include <sstream>
#include <cmath>


namespace kernel_selector {

struct base_params;

using JitDefinitions = std::vector<std::pair<std::string, std::string>>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::string GetTypeName() { throw std::runtime_error("Implement me"); }
template <>
inline std::string GetTypeName<int8_t>() { return "char"; }
template <>
inline std::string GetTypeName<uint8_t>() { return "uchar"; }
template <>
inline std::string GetTypeName<int16_t>() { return "short"; }
template <>
inline std::string GetTypeName<uint16_t>() { return "ushort"; }
template <>
inline std::string GetTypeName<int32_t>() { return "int"; }
template <>
inline std::string GetTypeName<uint32_t>() { return "uint"; }
template <>
inline std::string GetTypeName<int64_t>() { return "long"; } 
template <>
inline std::string GetTypeName<uint64_t>() { return "ulong"; }
template <>
inline std::string GetTypeName<float>() { return "float"; }
template <>
inline std::string GetTypeName<double>() { return "double"; }

std::string toCLType(WeightsType wType);
std::string toCLType(Datatype dType);
std::string getMeanOpString(MeanOp op);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ToCodeString functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO improve to_code_string specializations
template<typename T>
std::string toCodeString(T val) { return std::to_string(val); }

inline std::string toCodeString(const std::string& val) { return val; }
inline std::string toCodeString(const char* val) { return val; }
inline std::string toCodeString(bool val) { return val ? "1" : "0"; }
std::string toCodeString(float val);
std::string toCodeString(double val);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// JitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename VecT, typename ValT, typename Func>
inline std::string toVectorString(const VecT& vec, const std::string& vectorType, size_t maxDim, ValT padFillingVal, Func fetchFunc)
{
    std::stringstream ss;
    ss << "(" << vectorType << " []){ ";
    for (size_t i = 0; i < vec.size(); i++)
        ss << toCodeString(fetchFunc(vec[i])) << ",";
    for (size_t i = vec.size(); i < maxDim; i++)
        ss << padFillingVal << ",";
    ss << " } ";
    return ss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// JitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class JitConstant 
{
protected:
    const std::string _name;
    JitConstant(const std::string& name):_name(name){}

public:
    virtual JitDefinitions GetDefinitions() const = 0;
    virtual ~JitConstant() {}
};

class simple_jit_constant : public JitConstant 
{
    const std::string _value;

public:
    simple_jit_constant(const std::string& name, const std::string& value)
        : JitConstant(name), _value(value) {}

    JitDefinitions GetDefinitions() const override 
    {
        return JitDefinitions{ {_name, _value} };
    }
};

template<typename T>
std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, T value) 
{
    return std::static_pointer_cast<JitConstant>(std::make_shared<simple_jit_constant>(name, toCodeString(value)));
}

std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const struct Tensor::DataTensor& value);
std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const struct Tensor::WeightsTensor& value);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// VectorDataJitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class VectorDataJitConstant : public JitConstant 
{
    const std::vector<T> _data;

public:
    VectorDataJitConstant(const std::string& name, const std::vector<T>& data) : JitConstant(name), _data(data) {}

    JitDefinitions GetDefinitions() const override 
    {
        JitDefinitions result{
            { _name + "_SIZE", toCodeString(_data.size()) },
            { _name, toVectorString(_data, GetTypeName<T>(), _data.size(), 1, [](const T& v) {return v; } ) },
        };
        return result;
    }
};

template <typename T>
inline  std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const std::vector<T>& value) 
{
    return std::static_pointer_cast<JitConstant>(std::make_shared<VectorDataJitConstant<T>>(name, value));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Size
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class SizeJitConstant : public JitConstant
{
    const Size<T> _size;

public:
    SizeJitConstant(const std::string& name, const Size<T>& size) : JitConstant(name), _size(size) {}

    JitDefinitions GetDefinitions() const override
    {
        JitDefinitions definitions{
            { _name + "_SIZE_X",        toCodeString(_size.x) },
            { _name + "_SIZE_Y",        toCodeString(_size.y) },
        };
        return definitions;
    }
};

template <typename T>
inline std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const Size<T>& value)
{
    return std::static_pointer_cast<JitConstant>(std::make_shared<SizeJitConstant<T>>(name, value));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DimTensor
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class DimVectorJitConstant : public JitConstant
{
    const DimTensor<T> _dims;

public:
    DimVectorJitConstant(const std::string& name, const DimTensor<T>& size) : JitConstant(name), _dims(size) {}

    JitDefinitions GetDefinitions() const override
    {
        JitDefinitions definitions{
            { _name + "_BATCH_NUM",   toCodeString(_dims.b) },
            { _name + "_FEATURE_NUM", toCodeString(_dims.f) },
            { _name + "_SIZE_Y",      toCodeString(_dims.y) },
            { _name + "_SIZE_X",      toCodeString(_dims.x) },
        };
        return definitions;
    }
};

template <typename T>
std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const DimTensor<T>& value)
{
    return std::make_shared<DimVectorJitConstant<T>>(name, value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// jit_constants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class JitConstants 
{
    std::vector<std::shared_ptr<JitConstant>> _constants;
public:
    JitConstants(std::initializer_list<std::shared_ptr<JitConstant>> constants) :_constants(constants) {}

    inline void AddConstant(std::shared_ptr<JitConstant> constant)
    {
        _constants.push_back(constant);
    }

    inline void AddConstants(const std::vector<std::shared_ptr<JitConstant>>& constants)
    {
        for (const auto& c : constants)
        {
            _constants.push_back(c);
        }
    }

    inline void Merge(const JitConstants& jit)
    {
        AddConstants(jit._constants);
    }

    JitDefinitions GetDefinitions() const;
};

JitConstants MakeActivationJitConstants(const base_activation_params& params, const std::string& suffix="");
JitConstants MakeBaseParamsJitConstants(const base_params& params);
JitConstants MakeLoopUnrollParamsJitConstants(uint32_t loopCount);
JitConstants MakeUnitTypeJitConstants(Datatype dataType);

}
