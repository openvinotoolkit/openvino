// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ostream>

#include "openvino/core/model.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/matrix_nms.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/multiclass_nms_base.hpp"

namespace ov {
namespace test {
namespace utils {

// clang-format off
enum ComparisonTypes {
    EQUAL,
    NOT_EQUAL,
    IS_FINITE,
    IS_INF,
    IS_NAN,
    LESS,
    LESS_EQUAL,
    GREATER,
    GREATER_EQUAL
};

enum ConversionTypes {
    CONVERT,
    CONVERT_LIKE
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

enum EltwiseTypes {
    ADD,
    MULTIPLY,
    SUBTRACT,
    DIVIDE,
    SQUARED_DIFF,
    POWER,
    FLOOR_MOD,
    MOD,
    ERF,
    BITWISE_AND,
    BITWISE_NOT,
    BITWISE_OR,
    BITWISE_XOR,
    RIGHT_SHIFT,
    LEFT_SHIFT
};

enum SqueezeOpType {
    SQUEEZE,
    UNSQUEEZE
};

enum class InputLayerType {
    CONSTANT,
    PARAMETER,
};

enum LogicalTypes {
    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_XOR,
    LOGICAL_NOT
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
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
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
    LogicalNot,
    RoundHalfToEven,
    RoundHalfAwayFromZero,
    GeluErf,
    GeluTanh,
    SoftSign,
    IsFinite,
    IsInf,
    IsNaN,
};

enum MinMaxOpType {
    MINIMUM,
    MAXIMUM
};

enum PoolingTypes {
    MAX,
    AVG
};

enum ROIPoolingTypes {
    ROI_MAX,
    ROI_BILINEAR
};

using ov::op::PadMode;

enum class SequenceTestsMode {
    PURE_SEQ,
    PURE_SEQ_RAND_SEQ_LEN_CONST,
    PURE_SEQ_RAND_SEQ_LEN_PARAM,
    CONVERT_TO_TI_MAX_SEQ_LEN_CONST,
    CONVERT_TO_TI_MAX_SEQ_LEN_PARAM,
    CONVERT_TO_TI_RAND_SEQ_LEN_CONST,
    CONVERT_TO_TI_RAND_SEQ_LEN_PARAM,
};

enum class DFTOpType {
    FORWARD,
    INVERSE
};

enum class QuantizationGranularity {
    Pertensor,
    Perchannel
};

enum class TensorIteratorBody {
    RNN,
    GRU,
    LSTM,
    // CNN todo: implement
};

enum class MemoryTransformation {
    NONE,
    LOW_LATENCY_V2,
    LOW_LATENCY_V2_REGULAR_API,
    LOW_LATENCY_V2_ORIGINAL_INIT
};
// clang-format on

std::ostream& operator<<(std::ostream& os, const ReductionType& m);

std::ostream& operator<<(std::ostream& os, ov::test::utils::EltwiseTypes type);

std::ostream& operator<<(std::ostream& os, ov::test::utils::SqueezeOpType type);

std::ostream& operator<<(std::ostream& os, ov::test::utils::InputLayerType type);

std::ostream& operator<<(std::ostream& os, ov::test::utils::ComparisonTypes type);

std::ostream& operator<<(std::ostream& os, ov::test::utils::LogicalTypes type);

std::ostream& operator<<(std::ostream& os, ov::op::v4::Interpolate::InterpolateMode type);

std::ostream& operator<<(std::ostream& os, ov::op::v4::Interpolate::CoordinateTransformMode type);

std::ostream& operator<<(std::ostream& os, ov::op::v4::Interpolate::NearestMode type);

std::ostream& operator<<(std::ostream& os, ov::op::v4::Interpolate::ShapeCalcMode type);

std::ostream& operator<<(std::ostream& os, SequenceTestsMode type);

std::ostream& operator<<(std::ostream& os, ov::op::util::MulticlassNmsBase::SortResultType type);

std::ostream& operator<<(std::ostream& os, ov::op::v8::MatrixNms::SortResultType type);

std::ostream& operator<<(std::ostream& os, ov::op::v8::MatrixNms::DecayFunction type);

std::ostream& operator<<(std::ostream& os, TensorIteratorBody type);

std::ostream& operator<<(std::ostream& os, QuantizationGranularity type);

std::ostream& operator<<(std::ostream& os, MemoryTransformation type);

}  // namespace utils
}  // namespace test
}  // namespace ov
