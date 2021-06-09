// Copyright (C) 2018-2021 Intel Corporation
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

enum ROIPoolingTypes {
    ROI_MAX,
    ROI_BILINEAR
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
    Ceiling,
    PReLu,
    Mish,
    HSwish,
    SoftPlus,
    Swish,
    HSigmoid,
    RoundHalfToEven,
    RoundHalfAwayFromZero,
    GeluErf,
    GeluTanh
};

enum EltwiseTypes {
    ADD,
    MULTIPLY,
    SUBTRACT,
    DIVIDE,
    SQUARED_DIFF,
    POWER,
    FLOOR_MOD,
    MOD,
    ERF
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

enum MinMaxOpType {
    MINIMUM,
    MAXIMUM
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
    L1,
    L2
};

enum class DFTOpType {
    FORWARD,
    INVERSE
};

enum class InputLayerType {
    CONSTANT,
    PARAMETER,
};

enum class PadMode {
    CONSTANT,
    EDGE,
    REFLECT,
    SYMMETRIC,
};

enum class TensorIteratorBody {
    RNN,
    GRU,
    LSTM,
    // CNN todo: implement
};

enum class SequenceTestsMode {
    PURE_SEQ,
    PURE_SEQ_RAND_SEQ_LEN_CONST,
    PURE_SEQ_RAND_SEQ_LEN_PARAM,
    CONVERT_TO_TI_MAX_SEQ_LEN_CONST,
    CONVERT_TO_TI_MAX_SEQ_LEN_PARAM,
    CONVERT_TO_TI_RAND_SEQ_LEN_CONST,
    CONVERT_TO_TI_RAND_SEQ_LEN_PARAM,
};

enum class MemoryTransformation {
    NONE,
    LOW_LATENCY,
    LOW_LATENCY_REGULAR_API,
    LOW_LATENCY_V2,
    LOW_LATENCY_V2_REGULAR_API,
    LOW_LATENCY_V2_ORIGINAL_INIT
};

std::ostream &operator<<(std::ostream &os, const ReductionType &m);
std::ostream &operator<<(std::ostream &os, const PadMode &m);

bool is_tensor_iterator_exist(const std::shared_ptr<ngraph::Function> & func);

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

std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>
        interpreterFunction(const std::shared_ptr<Function> &function,
                            const std::vector<std::vector<std::uint8_t>> &inputs,
                            const std::vector<ngraph::element::Type> &inputTypes = {});

//
// This function compares two nGraph functions and requires them to have exactly one output
// Check nodes types
// Check number of inputs
// Check shapes of each Node
//
void CompareFunctions(const Function &actual, const Function &expected);


std::shared_ptr<Function> foldFunction(const std::shared_ptr<Function> &function,
                                       const std::vector<std::vector<std::uint8_t>> &inputs,
                                       const std::vector<ngraph::element::Type> &inputTypes = {});

std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> getConstData(const std::shared_ptr<Function> &function);

std::shared_ptr<ngraph::Node> getNodeSharedPtr(const ngraph::NodeTypeInfo &type_info,
                                               const ngraph::OutputVector &outputVector);

std::vector<std::uint8_t> convertOutputPrecision(const std::vector<std::uint8_t> &output,
                                                 const element::Type_t &fromPrecision,
                                                 const element::Type_t &toPrecision,
                                                 const size_t elementsCount);

std::ostream& operator<<(std::ostream & os, ngraph::helpers::EltwiseTypes type);

std::ostream& operator<<(std::ostream & os, ngraph::helpers::SqueezeOpType type);

std::ostream& operator<<(std::ostream& os, ngraph::helpers::InputLayerType type);

std::ostream& operator<<(std::ostream & os, ngraph::helpers::ComparisonTypes type);

std::ostream& operator<<(std::ostream & os, ngraph::helpers::LogicalTypes type);

std::ostream& operator<<(std::ostream & os, ngraph::op::v4::Interpolate::InterpolateMode type);

std::ostream& operator<<(std::ostream & os, ngraph::op::v4::Interpolate::CoordinateTransformMode type);

std::ostream& operator<<(std::ostream & os, ngraph::op::v4::Interpolate::NearestMode type);

std::ostream& operator<<(std::ostream & os, ngraph::op::v4::Interpolate::ShapeCalcMode type);

std::ostream& operator<<(std::ostream & os, TensorIteratorBody type);

std::ostream& operator<<(std::ostream & os, SequenceTestsMode type);

std::ostream& operator<<(std::ostream & os, MemoryTransformation type);

}  // namespace helpers
}  // namespace ngraph
