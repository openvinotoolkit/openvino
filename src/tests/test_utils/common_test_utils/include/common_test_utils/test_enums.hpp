// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
    ERF
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
    RoundHalfToEven,
    RoundHalfAwayFromZero,
    GeluErf,
    GeluTanh,
    SoftSign
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

enum class PadMode {
    CONSTANT,
    EDGE,
    REFLECT,
    SYMMETRIC,
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

enum class DFTOpType {
    FORWARD,
    INVERSE
};

// clang-format on

}  // namespace utils
}  // namespace test
}  // namespace ov
