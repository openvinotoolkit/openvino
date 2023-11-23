// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#endif

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/runtime/tensor.hpp>
#include <vector>

#include "common_test_utils/test_enums.hpp"

namespace ngraph {
namespace helpers {

// clang-format off
using ov::test::utils::PoolingTypes;
using ov::test::utils::ROIPoolingTypes;
using ov::test::utils::ActivationTypes;
using ov::test::utils::ActivationTypes::None;
using ov::test::utils::ActivationTypes::Sigmoid;
using ov::test::utils::ActivationTypes::Tanh;
using ov::test::utils::ActivationTypes::Relu;
using ov::test::utils::ActivationTypes::LeakyRelu;
using ov::test::utils::ActivationTypes::Exp;
using ov::test::utils::ActivationTypes::Log;
using ov::test::utils::ActivationTypes::Sign;
using ov::test::utils::ActivationTypes::Abs;
using ov::test::utils::ActivationTypes::Gelu;
using ov::test::utils::ActivationTypes::Clamp;
using ov::test::utils::ActivationTypes::Negative;
using ov::test::utils::ActivationTypes::Acos;
using ov::test::utils::ActivationTypes::Acosh;
using ov::test::utils::ActivationTypes::Asin;
using ov::test::utils::ActivationTypes::Asinh;
using ov::test::utils::ActivationTypes::Atan;
using ov::test::utils::ActivationTypes::Atanh;
using ov::test::utils::ActivationTypes::Cos;
using ov::test::utils::ActivationTypes::Cosh;
using ov::test::utils::ActivationTypes::Floor;
using ov::test::utils::ActivationTypes::Sin;
using ov::test::utils::ActivationTypes::Sinh;
using ov::test::utils::ActivationTypes::Sqrt;
using ov::test::utils::ActivationTypes::Tan;
using ov::test::utils::ActivationTypes::Elu;
using ov::test::utils::ActivationTypes::Erf;
using ov::test::utils::ActivationTypes::HardSigmoid;
using ov::test::utils::ActivationTypes::Selu;
using ov::test::utils::ActivationTypes::Ceiling;
using ov::test::utils::ActivationTypes::PReLu;
using ov::test::utils::ActivationTypes::Mish;
using ov::test::utils::ActivationTypes::HSwish;
using ov::test::utils::ActivationTypes::SoftPlus;
using ov::test::utils::ActivationTypes::Swish;
using ov::test::utils::ActivationTypes::HSigmoid;
using ov::test::utils::ActivationTypes::RoundHalfToEven;
using ov::test::utils::ActivationTypes::RoundHalfAwayFromZero;
using ov::test::utils::ActivationTypes::GeluErf;
using ov::test::utils::ActivationTypes::GeluTanh;
using ov::test::utils::ActivationTypes::SoftSign;
using ov::test::utils::EltwiseTypes;
using ov::test::utils::EltwiseTypes::ADD;
using ov::test::utils::EltwiseTypes::MULTIPLY;
using ov::test::utils::EltwiseTypes::SUBTRACT;
using ov::test::utils::EltwiseTypes::DIVIDE;
using ov::test::utils::EltwiseTypes::SQUARED_DIFF;
using ov::test::utils::EltwiseTypes::POWER;
using ov::test::utils::EltwiseTypes::FLOOR_MOD;
using ov::test::utils::EltwiseTypes::MOD;
using ov::test::utils::EltwiseTypes::ERF;
using ov::test::utils::ComparisonTypes;
using ov::test::utils::ConversionTypes;
using ov::test::utils::LogicalTypes;
using ov::test::utils::SqueezeOpType;
using ov::test::utils::MinMaxOpType;

using ov::test::utils::TensorIteratorBody;
using ov::test::utils::ReductionType;
using ov::test::utils::DFTOpType;
using ov::test::utils::InputLayerType;
using ov::test::utils::PadMode;
using ov::test::utils::SequenceTestsMode;
using ov::test::utils::MemoryTransformation;
using ov::test::utils::QuantizationGranularity;
// clang-format on

bool is_tensor_iterator_exist(const std::shared_ptr<ov::Model>& func);

ov::OutputVector convert2OutputVector(const std::vector<std::shared_ptr<ov::Node>>& nodes);

template <class opType>
inline ov::NodeVector castOps2Nodes(const std::vector<std::shared_ptr<opType>>& ops) {
    ov::NodeVector nodes;
    for (const auto& op : ops) {
        nodes.push_back(std::dynamic_pointer_cast<ov::Node>(op));
    }
    return nodes;
}

std::vector<std::pair<ov::element::Type, std::vector<std::uint8_t>>> interpreterFunction(
    const std::shared_ptr<ov::Model>& function,
    const std::vector<std::vector<std::uint8_t>>& inputs,
    const std::vector<ov::element::Type>& inputTypes = {});

std::vector<ov::Tensor> interpretFunction(const std::shared_ptr<ov::Model>& function,
                                          const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs);

std::shared_ptr<ov::Model> foldFunction(const std::shared_ptr<ov::Model>& function,
                                       const std::vector<std::vector<std::uint8_t>>& inputs,
                                       const std::vector<ov::element::Type>& inputTypes = {});

std::vector<std::uint8_t> convertOutputPrecision(const std::vector<std::uint8_t>& output,
                                                 const element::Type_t& fromPrecision,
                                                 const element::Type_t& toPrecision,
                                                 const size_t elementsCount);

// todo: remove the following function from the source code after cleaning up VPU repo
void resize_function(std::shared_ptr<ov::Model> function, const std::vector<ov::Shape>& targetInputStaticShapes);

using ov::test::utils::operator<<;

}  // namespace helpers
}  // namespace ngraph
