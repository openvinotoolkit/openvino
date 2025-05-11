// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector_common.h"

#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <locale>

namespace kernel_selector {

struct base_params;

using JitDefinitions = std::vector<std::pair<std::string, std::string>>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::string GetTypeName() {
    throw std::runtime_error("Implement me");
}
template <>
inline std::string GetTypeName<int8_t>() {
    return "char";
}
template <>
inline std::string GetTypeName<uint8_t>() {
    return "uchar";
}
template <>
inline std::string GetTypeName<int16_t>() {
    return "short";
}
template <>
inline std::string GetTypeName<uint16_t>() {
    return "ushort";
}
template <>
inline std::string GetTypeName<int32_t>() {
    return "int";
}
template <>
inline std::string GetTypeName<uint32_t>() {
    return "uint";
}
template <>
inline std::string GetTypeName<int64_t>() {
    return "long";
}
template <>
inline std::string GetTypeName<uint64_t>() {
    return "ulong";
}
template <>
inline std::string GetTypeName<float>() {
    return "float";
}
template <>
inline std::string GetTypeName<double>() {
    return "double";
}

std::string toCLType(WeightsType wType);
std::string toCLType(Datatype dType);
std::string getMeanOpString(MeanOp op);
inline std::string toVectorMulString(const std::vector<std::string>& vec) {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < vec.size(); i++) {
        ss << vec[i];
        if (i != vec.size() - 1)
            ss << "*";
    }
    ss << ")";
    return ss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ToCodeString functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO improve to_code_string specializations
template <typename T>
std::string toCodeString(T val) {
    std::stringstream ss;
    ss.imbue(std::locale("C"));
    ss << val;
    return ss.str();
}

inline std::string toCodeString(const std::string& val) { return val; }
inline std::string toCodeString(const char* val) { return val; }
inline std::string toCodeString(bool val) { return val ? "1" : "0"; }
std::string toCodeString(float val);
std::string toCodeString(double val);
std::string toCodeString(size_t val);
std::string toCodeString(uint8_t val);
std::string toCodeString(int8_t val);
std::string toCodeString(const Tensor::Dim& dim, size_t offset, bool padded = false, bool pad_is_dynamic = false, size_t dynamic_pad_offset = 0);
std::string toShapeInfoString(size_t arg_idx, size_t data_idx_at_6d, bool is_output = false, size_t num_of_inputs = 0);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// JitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename VecT, typename ValT, typename Func>
inline std::string toVectorInitString(const VecT& vec,
                                      const std::string& vectorType,
                                      size_t maxDim,
                                      ValT padFillingVal,
                                      Func fetchFunc) {
    std::stringstream ss;
    ss << "{ ";
    for (size_t i = 0; i < vec.size(); i++)
        ss << toCodeString(fetchFunc(vec[i])) << ",";
    for (size_t i = vec.size(); i < maxDim; i++)
        ss << padFillingVal << ",";
    ss << " } ";
    return ss.str();
}

template <typename VecT, typename ValT, typename Func>
inline std::string toVectorString(const VecT& vec,
                                  const std::string& vectorType,
                                  size_t maxDim,
                                  ValT padFillingVal,
                                  Func fetchFunc) {
    std::stringstream ss;
    if (vectorType.length())
        ss << "(" << vectorType << " [])";
    ss << toVectorInitString(vec, vectorType, maxDim, padFillingVal, fetchFunc);
    return ss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// JitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class JitConstant {
protected:
    const std::string _name;
    explicit JitConstant(const std::string& name) : _name(name) {}

public:
    std::string GetJitName() { return _name; }
    virtual JitDefinitions GetDefinitions() const = 0;
    virtual ~JitConstant() {}
};

class simple_jit_constant : public JitConstant {
    const std::string _value;

public:
    simple_jit_constant(const std::string& name, const std::string& value) : JitConstant(name), _value(value) {}

    JitDefinitions GetDefinitions() const override { return JitDefinitions{{_name, _value}}; }
};

template <typename T>
std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, T value) {
    return std::static_pointer_cast<JitConstant>(std::make_shared<simple_jit_constant>(name, toCodeString(value)));
}

std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const struct Tensor::DataTensor& value);
std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const struct Tensor::WeightsTensor& value);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// VectorDataJitConstant
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class VectorDataJitConstant : public JitConstant {
    const std::vector<T> _data;

public:
    VectorDataJitConstant(const std::string& name, const std::vector<T>& data) : JitConstant(name), _data(data) {}

    JitDefinitions GetDefinitions() const override {
        JitDefinitions result{
            {_name + "_SIZE", toCodeString(_data.size())},
            {_name + "_INIT", toVectorInitString(_data, GetTypeName<T>(), _data.size(), 1, [](const T& v) { return v; })},
            {_name, toVectorString(_data, GetTypeName<T>(), _data.size(), 1, [](const T& v) { return v; })},
        };
        return result;
    }
};

template <typename T>
inline std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const std::vector<T>& value) {
    return std::static_pointer_cast<JitConstant>(std::make_shared<VectorDataJitConstant<T>>(name, value));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Size
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class SizeJitConstant : public JitConstant {
    const Size<T> _size;

public:
    SizeJitConstant(const std::string& name, const Size<T>& size) : JitConstant(name), _size(size) {}

    JitDefinitions GetDefinitions() const override {
        JitDefinitions definitions{
            {_name + "_SIZE_X", toCodeString(_size.x)},
            {_name + "_SIZE_Y", toCodeString(_size.y)},
            {_name + "_SIZE_Z", toCodeString(_size.z)},
        };
        return definitions;
    }
};

template <typename T>
inline std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const Size<T>& value) {
    return std::static_pointer_cast<JitConstant>(std::make_shared<SizeJitConstant<T>>(name, value));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DimTensor
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class DimVectorJitConstant : public JitConstant {
    const DimTensor<T> _dims;

public:
    DimVectorJitConstant(const std::string& name, const DimTensor<T>& size) : JitConstant(name), _dims(size) {}

    JitDefinitions GetDefinitions() const override {
        JitDefinitions definitions{
            {_name + "_BATCH_NUM", toCodeString(_dims.b)},
            {_name + "_FEATURE_NUM", toCodeString(_dims.f)},
            {_name + "_SIZE_Y", toCodeString(_dims.y)},
            {_name + "_SIZE_X", toCodeString(_dims.x)},
            {_name + "_SIZE_Z", toCodeString(_dims.z)},
            {_name + "_SIZE_W", toCodeString(_dims.w)},
        };
        return definitions;
    }
};

template <typename T>
std::shared_ptr<JitConstant> MakeJitConstant(const std::string& name, const DimTensor<T>& value) {
    return std::make_shared<DimVectorJitConstant<T>>(name, value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// jit_constants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class JitConstants {
    std::vector<std::shared_ptr<JitConstant>> _constants;

public:
    JitConstants(std::initializer_list<std::shared_ptr<JitConstant>> constants) : _constants(constants) {}

    inline void AddConstant(std::shared_ptr<JitConstant> constant) { _constants.push_back(constant); }

    inline void AddConstants(const std::vector<std::shared_ptr<JitConstant>>& constants) {
        for (const auto& c : constants) {
            _constants.push_back(c);
        }
    }

    inline void Merge(const JitConstants& jit) { AddConstants(jit._constants); }

    inline void RemoveConstant(std::string name) {
        _constants.erase(
            std::remove_if(_constants.begin(),
                           _constants.end(),
                           [=](std::shared_ptr<JitConstant> x) -> bool { return x->GetJitName() == name; }),
            _constants.end());
    }

    JitDefinitions GetDefinitions() const;
};

// Historically, the whole kernel computation was performed in a single, UNIT,
// type and the activation function assumed to be done in that UNIT_TYPE. With
// the addition of different quantization schemes the kernels started to use
// multiple types and there might be no single UNIT type. Also it's not clear
// from the kernel-agnostic code in which type activation should be done.
//
// Simple solution for this is to make the ACTIVATION[_SUFFIX] jit macro accept
// an additional type parameter, but fixing all the existing implementations is
// costly, so in the meantime it's only done by explicitly specifying
// `use_type_parameter` to true and for the remaining kernels the old scheme
// will be used for now.
//
// Note, that we need the type to be the argument of the macro itself (as
// opposite to this function) so that the logic of choosing the activation type
// could be contained in the target code exclusively, without the need to do
// that processing on the host side. Otherwise it would be harder to read the
// target code as that would require looking into several place to understand
// the logic.
JitConstants MakeActivationJitConstants(const base_activation_params& params,
                                        Datatype output_dt,
                                        const std::string& suffix = "",
                                        bool use_type_parameter = false,
                                        bool disable_type_conversion = false);
JitConstants MakeActivationJitConstants(ActivationFunction activation_function,
                                        Datatype output_dt,
                                        const std::string& suffix,
                                        bool use_type_parameter,
                                        bool disable_type_conversion = false);
JitConstants MakeActivationJitConstants(std::vector<kernel_selector::base_activation_params> params,
                                        Datatype output_dt,
                                        const std::string& suffix = "",
                                        bool use_type_parameter = false,
                                        bool disable_type_conversion = false,
                                        bool convert_input_to_output_dt = false);
JitConstants MakeBaseParamsJitConstants(const base_params& params);
JitConstants MakeLoopUnrollParamsJitConstants(uint32_t loopCount);

// Generates macro CONST_LOOP(count, macro), where:
// count - should expand to integer with number of loop iterations;
// macro - is macro name that will be expanded at each iteration with current iteration number as single argument.
JitConstants MakeConstantLoopUnrollJitConstants(uint32_t loopCount);

JitConstants MakeTypeJitConstants(Datatype dataType, const std::string& macroName);
JitConstants MakeTypeJitConstants(WeightsType weightsType, const std::string& macroName);
inline JitConstants MakeUnitTypeJitConstants(Datatype dataType) { return MakeTypeJitConstants(dataType, "UNIT"); }
JitConstants make_int4_packed_type_jit_constant(const std::string& macro_name, WeightsType wt, size_t pack_size);

class FusedOpsCodeGenerator {
public:
    explicit FusedOpsCodeGenerator(fused_operation_desc desc) : desc(desc) {}

    struct idx_desc {
        std::string b;
        std::string f;
        std::string v;
        std::string u;
        std::string w;
        std::string z;
        std::string y;
        std::string x;
        size_t dims;
        explicit idx_desc(std::vector<std::string> idx, DataTensor t)
            : b("0"), f("0"), v("0"), u("0"), w("0"), z("0"), y("0"), x("0"), dims(0) {
            dims = idx.size();
            switch (dims) {
                case 1: f = idx[0]; break;
                case 2: b = idx[0]; f = idx[1]; break;
                case 3: b = idx[0]; f = idx[1]; y = idx[2]; break;
                case 4: b = idx[0]; f = idx[1]; y = idx[2]; x = idx[3]; break;
                case 5: b = idx[0]; f = idx[1]; z = idx[2]; y = idx[3]; x = idx[4]; break;
                case 6: b = idx[0]; f = idx[1]; w = idx[2]; z = idx[3]; y = idx[4]; x = idx[5]; break;
                case 7: b = idx[0]; f = idx[1]; u = idx[2]; w = idx[3]; z = idx[4]; y = idx[5]; x = idx[6]; break;
                case 8: b = idx[0]; f = idx[1]; v = idx[2]; u = idx[3]; w = idx[4]; z = idx[5]; y = idx[6]; x = idx[7]; break;
                default: throw std::runtime_error("More than 8 dimenstions is not supported in fused op generator");
            }

            if (t.Batch().v == 1) {
                b = "0";
            }
            if (t.Feature().v == 1) {
                f = "0";
            }
            if (t.V().v == 1) {
                v = "0";
            }
            if (t.U().v == 1) {
                u = "0";
            }
            if (t.W().v == 1) {
                w = "0";
            }
            if (t.Z().v == 1) {
                z = "0";
            }
            if (t.Y().v == 1) {
                y = "0";
            }
            if (t.X().v == 1) {
                x = "0";
            }
        }
    };

    JitConstants MakeFusedTensorJitConstants(const FusedOpsConfiguration& conf) const;
    JitConstants MakeInputDeclsJitConstants(const FusedOpsConfiguration& conf) const;
    JitConstants MakeLoadJitConstants(const FusedOpsConfiguration& conf, const DataTensor prim_output) const;
    JitConstants MakeOpJitConstants(const FusedOpsConfiguration& conf,
                                    const std::string in_var, const Datatype in_type,
                                    std::string& out_var) const;

    bool CanPreloadData(const FusedOpsConfiguration& conf) const;

    std::string GetTypeStr() const;
    std::string GetInputTensorName(size_t input_id) const;
    std::string GetOutputTensorName() const;
    std::string GetInputTypeName(size_t input_id, size_t vec_size) const;
    std::string GetJitLoad(const FusedOpsConfiguration& conf, size_t input_id, const DataTensor prim_output,
                           bool reuse_index = false, std::string reused_idx = "") const;
    std::string GetIdx(size_t input_id, idx_desc idx, bool should_be_safe) const;
    std::string GetInputPtrName(size_t input_id) const;
    std::string GetInputVarName(size_t input_id, bool is_shuffled = false, std::string shuffle_var = "") const;
    std::string GetOutputVarName(std::string input_var_name, size_t op_id) const;
    std::string ConvertToOutputType(std::string var, size_t vec_size = 1) const;
    std::string ConvertToType(std::string var, Datatype dt, size_t vec_size = 1) const;
    std::string CastToType(std::string var, Datatype dt, size_t vec_size = 1) const;
    std::string Broadcast(std::string var,  Datatype dt, size_t vec_size = 1) const;
    std::string ConvertToOutputTypeSat(std::string var, size_t vec_size = 1) const;
    std::string GetOutputType(size_t vec_size = 1) const;
    std::string GetType(Datatype dt, size_t vec_size = 1) const;

private:
    std::vector<size_t> GetRequiredInputs() const;

    fused_operation_desc desc;
};

}  // namespace kernel_selector
