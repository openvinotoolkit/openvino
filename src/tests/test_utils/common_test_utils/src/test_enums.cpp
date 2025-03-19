// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
namespace utils {

std::ostream& operator<<(std::ostream& os, const ov::test::utils::ReductionType& m) {
    switch (m) {
    case ov::test::utils::Mean:
        os << "Mean";
        break;
    case ov::test::utils::Max:
        os << "Max";
        break;
    case ov::test::utils::Min:
        os << "Min";
        break;
    case ov::test::utils::Prod:
        os << "Prod";
        break;
    case ov::test::utils::Sum:
        os << "Sum";
        break;
    case ov::test::utils::LogicalOr:
        os << "LogicalOr";
        break;
    case ov::test::utils::LogicalAnd:
        os << "LogicalAnd";
        break;
    case ov::test::utils::L1:
        os << "ReduceL1";
        break;
    case ov::test::utils::L2:
        os << "ReduceL2";
        break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const ov::test::utils::EltwiseTypes type) {
    switch (type) {
    case ov::test::utils::EltwiseTypes::SUBTRACT:
        os << "Sub";
        break;
    case ov::test::utils::EltwiseTypes::MULTIPLY:
        os << "Prod";
        break;
    case ov::test::utils::EltwiseTypes::ADD:
        os << "Sum";
        break;
    case ov::test::utils::EltwiseTypes::DIVIDE:
        os << "Div";
        break;
    case ov::test::utils::EltwiseTypes::SQUARED_DIFF:
        os << "SqDiff";
        break;
    case ov::test::utils::EltwiseTypes::POWER:
        os << "Pow";
        break;
    case ov::test::utils::EltwiseTypes::FLOOR_MOD:
        os << "FloorMod";
        break;
    case ov::test::utils::EltwiseTypes::MOD:
        os << "Mod";
        break;
    case ov::test::utils::EltwiseTypes::ERF:
        os << "Erf";
        break;
    case ov::test::utils::EltwiseTypes::BITWISE_AND:
        os << "BitwiseAnd";
        break;
    case ov::test::utils::EltwiseTypes::BITWISE_NOT:
        os << "BitwiseNot";
        break;
    case ov::test::utils::EltwiseTypes::BITWISE_OR:
        os << "BitwiseOr";
        break;
    case ov::test::utils::EltwiseTypes::BITWISE_XOR:
        os << "BitwiseXor";
        break;
    case ov::test::utils::EltwiseTypes::RIGHT_SHIFT:
        os << "BitwiseRightShift";
        break;
    case ov::test::utils::EltwiseTypes::LEFT_SHIFT:
        os << "BitwiseLeftShift";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, ov::test::utils::SqueezeOpType type) {
    switch (type) {
    case ov::test::utils::SqueezeOpType::SQUEEZE:
        os << "Squeeze";
        break;
    case ov::test::utils::SqueezeOpType::UNSQUEEZE:
        os << "Unsqueeze";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, ov::test::utils::InputLayerType type) {
    switch (type) {
    case ov::test::utils::InputLayerType::CONSTANT:
        os << "CONSTANT";
        break;
    case ov::test::utils::InputLayerType::PARAMETER:
        os << "PARAMETER";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_INPUT_LAYER_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, ov::test::utils::ComparisonTypes type) {
    switch (type) {
    case ov::test::utils::ComparisonTypes::EQUAL:
        os << "Equal";
        break;
    case ov::test::utils::ComparisonTypes::NOT_EQUAL:
        os << "NotEqual";
        break;
    case ov::test::utils::ComparisonTypes::GREATER:
        os << "Greater";
        break;
    case ov::test::utils::ComparisonTypes::GREATER_EQUAL:
        os << "GreaterEqual";
        break;
    case ov::test::utils::ComparisonTypes::IS_FINITE:
        os << "IsFinite";
        break;
    case ov::test::utils::ComparisonTypes::IS_INF:
        os << "IsInf";
        break;
    case ov::test::utils::ComparisonTypes::IS_NAN:
        os << "IsNaN";
        break;
    case ov::test::utils::ComparisonTypes::LESS:
        os << "Less";
        break;
    case ov::test::utils::ComparisonTypes::LESS_EQUAL:
        os << "LessEqual";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, ov::test::utils::LogicalTypes type) {
    switch (type) {
    case ov::test::utils::LogicalTypes::LOGICAL_AND:
        os << "LogicalAnd";
        break;
    case ov::test::utils::LogicalTypes::LOGICAL_OR:
        os << "LogicalOr";
        break;
    case ov::test::utils::LogicalTypes::LOGICAL_NOT:
        os << "LogicalNot";
        break;
    case ov::test::utils::LogicalTypes::LOGICAL_XOR:
        os << "LogicalXor";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, ov::op::v11::Interpolate::InterpolateMode type) {
    switch (type) {
    case ov::op::v11::Interpolate::InterpolateMode::CUBIC:
        os << "cubic";
        break;
    case ov::op::v11::Interpolate::InterpolateMode::LINEAR:
        os << "linear";
        break;
    case ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX:
        os << "linear_onnx";
        break;
    case ov::op::v11::Interpolate::InterpolateMode::NEAREST:
        os << "nearest";
        break;
    case ov::op::v11::Interpolate::InterpolateMode::BILINEAR_PILLOW:
        os << "bilinear_pillow";
        break;
    case ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW:
        os << "bicubic_pillow";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, ov::op::v11::Interpolate::CoordinateTransformMode type) {
    switch (type) {
    case ov::op::v11::Interpolate::CoordinateTransformMode::ALIGN_CORNERS:
        os << "align_corners";
        break;
    case ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC:
        os << "asymmetric";
        break;
    case ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL:
        os << "half_pixel";
        break;
    case ov::op::v11::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL:
        os << "pytorch_half_pixel";
        break;
    case ov::op::v11::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN:
        os << "tf_half_pixel_for_nn";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, ov::op::v11::Interpolate::NearestMode type) {
    switch (type) {
    case ov::op::v11::Interpolate::NearestMode::CEIL:
        os << "ceil";
        break;
    case ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_CEIL:
        os << "round_prefer_ceil";
        break;
    case ov::op::v11::Interpolate::NearestMode::FLOOR:
        os << "floor";
        break;
    case ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR:
        os << "round_prefer_floor";
        break;
    case ov::op::v11::Interpolate::NearestMode::SIMPLE:
        os << "simple";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, ov::op::v11::Interpolate::ShapeCalcMode type) {
    switch (type) {
    case ov::op::v11::Interpolate::ShapeCalcMode::SCALES:
        os << "scales";
        break;
    case ov::op::v11::Interpolate::ShapeCalcMode::SIZES:
        os << "sizes";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, SequenceTestsMode type) {
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

std::ostream& operator<<(std::ostream& os, op::v8::MatrixNms::SortResultType type) {
    switch (type) {
    case op::v8::MatrixNms::SortResultType::CLASSID:
        os << "CLASSID";
        break;
    case op::v8::MatrixNms::SortResultType::SCORE:
        os << "SCORE";
        break;
    case op::v8::MatrixNms::SortResultType::NONE:
        os << "NONE";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, ov::op::util::MulticlassNmsBase::SortResultType type) {
    switch (type) {
    case op::util::MulticlassNmsBase::SortResultType::CLASSID:
        os << "CLASSID";
        break;
    case op::util::MulticlassNmsBase::SortResultType::SCORE:
        os << "SCORE";
        break;
    case op::util::MulticlassNmsBase::SortResultType::NONE:
        os << "NONE";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, op::v8::MatrixNms::DecayFunction type) {
    switch (type) {
    case op::v8::MatrixNms::DecayFunction::GAUSSIAN:
        os << "GAUSSIAN";
        break;
    case op::v8::MatrixNms::DecayFunction::LINEAR:
        os << "LINEAR";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, TensorIteratorBody type) {
    switch (type) {
    case TensorIteratorBody::LSTM:
        os << "LSTM";
        break;
    case TensorIteratorBody::RNN:
        os << "RNN";
        break;
    case TensorIteratorBody::GRU:
        os << "GRU";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, QuantizationGranularity type) {
    switch (type) {
    case QuantizationGranularity::Pertensor:
        os << "Pertensor";
        break;
    case QuantizationGranularity::Perchannel:
        os << "Perchannel";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_OP_TYPE");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, MemoryTransformation type) {
    switch (type) {
    case MemoryTransformation::NONE:
        os << "NONE";
        break;
    case MemoryTransformation::LOW_LATENCY_V2:
        os << "LOW_LATENCY_V2";
        break;
    case MemoryTransformation::LOW_LATENCY_V2_REGULAR_API:
        os << "LOW_LATENCY_V2_REGULAR_API";
        break;
    case MemoryTransformation::LOW_LATENCY_V2_ORIGINAL_INIT:
        os << "LOW_LATENCY_V2_ORIGINAL_INIT";
        break;
    default:
        throw std::runtime_error("NOT_SUPPORTED_TYPE");
    }
    return os;
}

}  // namespace utils
}  // namespace test
}  // namespace ov
