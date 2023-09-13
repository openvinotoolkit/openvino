// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
namespace utils {

std::ostream& operator<<(std::ostream& os, const ComparisonTypes type) {
    switch (type) {
    case ComparisonTypes::EQUAL:
        os << "Equal";
        break;
    case ComparisonTypes::NOT_EQUAL:
        os << "NotEqual";
        break;
    case ComparisonTypes::GREATER:
        os << "Greater";
        break;
    case ComparisonTypes::GREATER_EQUAL:
        os << "GreaterEqual";
        break;
    case ComparisonTypes::IS_FINITE:
        os << "IsFinite";
        break;
    case ComparisonTypes::IS_INF:
        os << "IsInf";
        break;
    case ComparisonTypes::IS_NAN:
        os << "IsNaN";
        break;
    case ComparisonTypes::LESS:
        os << "Less";
        break;
    case ComparisonTypes::LESS_EQUAL:
        os << "LessEqual";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_COMPARISON_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const ConversionTypes type) {
    switch (type) {
    case ConversionTypes::CONVERT:
        os << "Convert";
        break;
    case ConversionTypes::CONVERT_LIKE:
        os << "ConvertLike";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_CONVERSION_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const ReductionType type) {
    switch (type) {
    case ReductionType::Mean:
        os << "Mean";
        break;
    case ReductionType::Max:
        os << "Max";
        break;
    case ReductionType::Min:
        os << "Min";
        break;
    case ReductionType::Prod:
        os << "Prod";
        break;
    case ReductionType::Sum:
        os << "Sum";
        break;
    case ReductionType::LogicalOr:
        os << "LogicalOr";
        break;
    case ReductionType::LogicalAnd:
        os << "LogicalAnd";
        break;
    case ReductionType::L1:
        os << "ReduceL1";
        break;
    case ReductionType::L2:
        os << "ReduceL2";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_REDUCTION_TYPE");
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, const InputLayerType type) {
    switch (type) {
    case InputLayerType::CONSTANT:
        os << "CONSTANT";
        break;
    case InputLayerType::PARAMETER:
        os << "PARAMETER";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_INPUT_LAYER_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const SequenceTestsMode type) {
    switch (type) {
    case SequenceTestsMode::PURE_SEQ:
        os << "PURE_SEQ";
        break;
    case SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST:
        os << "PURE_SEQ_RAND_SEQ_LEN_CONST";
        break;
    case SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM:
        os << "PURE_SEQ_RAND_SEQ_LEN_PARAM";
        break;
    case SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM:
        os << "CONVERT_TO_TI_RAND_SEQ_LEN_PARAM";
        break;
    case SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST:
        os << "CONVERT_TO_TI_RAND_SEQ_LEN_CONST";
        break;
    case SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM:
        os << "CONVERT_TO_TI_MAX_SEQ_LEN_PARAM";
        break;
    case SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_CONST:
        os << "CONVERT_TO_TI_MAX_SEQ_LEN_CONST";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
