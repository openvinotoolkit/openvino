// Copyright (C) 2019 Intel Corporationconvert2OutputVector
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <vector>
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/runtime/tensor.hpp>
#include <backend_manager.hpp>
#include <backend.hpp>

namespace ngraph {
namespace helpers {

template<ngraph::element::Type_t type>
struct nGraphTypesTrait {
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::boolean> {
    using value_type = bool;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::f64> {
    using value_type = double;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::f32> {
    using value_type = float;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::f16> {
    using value_type = ngraph::float16;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::bf16> {
    using value_type = ngraph::bfloat16;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::i8> {
    using value_type = int8_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::i16> {
    using value_type = int16_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::i32> {
    using value_type = int32_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::i64> {
    using value_type = int64_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::u8> {
    using value_type = uint8_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::u16> {
    using value_type = uint16_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::u32> {
    using value_type = uint32_t;
};
template<>
struct nGraphTypesTrait<ngraph::element::Type_t::u64> {
    using value_type = uint64_t;
};
enum PoolingTypes {
    MAX,
    AVG
};
enum ActivationTypes {
    None,
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu,
    Exp,
    Log,
    Sign,
    Abs,
    Gelu,
    Clamp,
    Negative,
    Acos,
    Asin,
    Atan,
    Cos,
    Cosh,
    Floor,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Elu,
    Erf,
    HardSigmoid,
    Selu,
    Ceiling
};

enum EltwiseTypes {
    ADD,
    MULTIPLY,
    SUBTRACT
};

enum ComparisonTypes {
    EQUAL,
    NOT_EQUAL,
    LESS,
    LESS_EQUAL,
    GREATER,
    GREATER_EQUAL
};

enum LogicalTypes {
    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_XOR,
    LOGICAL_NOT
};

enum SqueezeOpType {
    SQUEEZE,
    UNSQUEEZE
};

enum QuantizationGranularity {
    Pertensor,
    Perchannel
};

enum ReductionType {
    Mean,
    Max,
    Min,
    Prod,
    Sum,
    LogicalOr,
    LogicalAnd,
    LogicalXor
};

enum class InputLayerType {
    CONSTANT,
    PARAMETER,
};

std::ostream &operator<<(std::ostream &os, const ReductionType &m);

inline std::string quantizationGranularityToString(const QuantizationGranularity &granularity) {
    static std::map<QuantizationGranularity, std::string> names = {
            {Pertensor,  "Pertensor"},
            {Perchannel, "Perchannel"},
    };

    auto i = names.find(granularity);
    if (i != names.end())
        return i->second;
    else
        throw std::runtime_error("Unsupported QuantizationGranularity type");
}

inline std::ostream &operator<<(std::ostream &out, const QuantizationGranularity &granularity) {
    return out << quantizationGranularityToString(granularity);
}

ngraph::OutputVector convert2OutputVector(const std::vector<std::shared_ptr<ngraph::Node>> &nodes);

template<class opType>
inline ngraph::NodeVector castOps2Nodes(const std::vector<std::shared_ptr<opType>> &ops) {
    ngraph::NodeVector nodes;
    for (const auto &op : ops) {
        nodes.push_back(std::dynamic_pointer_cast<ngraph::Node>(op));
    }
    return nodes;
}

std::vector<std::vector<std::uint8_t>> interpreterFunction(const std::shared_ptr<Function> &function,
                                                           const std::vector<std::vector<std::uint8_t>> &inputs,
                                                           element::Type_t convertType = element::Type_t::undefined);

//
// This function compares two nGraph functions and requires them to have exactly one output
// Check nodes types
// Check number of inputs
// Check shapes of each Node
//
void CompareFunctions(const Function &actual, const Function &expected);


std::shared_ptr<Function> foldFunction(const std::shared_ptr<Function> &function,
                                       const std::vector<std::vector<std::uint8_t>> &inputs);

std::vector<std::vector<std::uint8_t>> getConstData(const std::shared_ptr<Function> &function,
                                                    element::Type_t convertType = element::Type_t::undefined);

std::shared_ptr<ngraph::Node> getNodeSharedPtr(const ngraph::NodeTypeInfo &type_info,
                                               const ngraph::OutputVector &outputVector);

std::vector<std::uint8_t> convertOutputPrecision(std::vector<std::uint8_t> &output,
                                                 const element::Type_t &fromPrecision,
                                                 const element::Type_t &toPrecision,
                                                 const size_t elementsCount);

std::ostream& operator<<(std::ostream & os, ngraph::helpers::EltwiseTypes type);

std::ostream& operator<<(std::ostream & os, ngraph::helpers::SqueezeOpType type);

std::ostream& operator<<(std::ostream& os, ngraph::helpers::InputLayerType type);

std::ostream& operator<<(std::ostream & os, ngraph::helpers::ComparisonTypes type);

std::ostream& operator<<(std::ostream & os, ngraph::helpers::LogicalTypes type);

}  // namespace helpers
}  // namespace ngraph
