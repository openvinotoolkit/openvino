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
#include "common_test_utils/test_enums.hpp"
#include <vector>

namespace ngraph {
namespace helpers {

// clang-format off
using ov::test::utils::PoolingTypes;
using ov::test::utils::ROIPoolingTypes;
using ov::test::utils::ActivationTypes;
using ov::test::utils::EltwiseTypes;
using ov::test::utils::ComparisonTypes;
using ov::test::utils::ConversionTypes;
using ov::test::utils::LogicalTypes;
using ov::test::utils::SqueezeOpType;
using ov::test::utils::MinMaxOpType;

enum QuantizationGranularity {
    Pertensor,
    Perchannel
};

using ov::test::utils::ReductionType;
using ov::test::utils::DFTOpType;
using ov::test::utils::InputLayerType;
using ov::test::utils::PadMode;


enum class TensorIteratorBody {
    RNN,
    GRU,
    LSTM,
    // CNN todo: implement
};

using ov::test::utils::SequenceTestsMode;

enum class MemoryTransformation {
    NONE,
    LOW_LATENCY_V2,
    LOW_LATENCY_V2_REGULAR_API,
    LOW_LATENCY_V2_ORIGINAL_INIT
};
// clang-format on

bool is_tensor_iterator_exist(const std::shared_ptr<ngraph::Function>& func);

inline std::string quantizationGranularityToString(const QuantizationGranularity& granularity) {
    static std::map<QuantizationGranularity, std::string> names = {
        {Pertensor, "Pertensor"},
        {Perchannel, "Perchannel"},
    };

    auto i = names.find(granularity);
    if (i != names.end())
        return i->second;
    else
        throw std::runtime_error("Unsupported QuantizationGranularity type");
}

inline std::ostream& operator<<(std::ostream& out, const QuantizationGranularity& granularity) {
    return out << quantizationGranularityToString(granularity);
}

ngraph::OutputVector convert2OutputVector(const std::vector<std::shared_ptr<ngraph::Node>>& nodes);

template <class opType>
inline ngraph::NodeVector castOps2Nodes(const std::vector<std::shared_ptr<opType>>& ops) {
    ngraph::NodeVector nodes;
    for (const auto& op : ops) {
        nodes.push_back(std::dynamic_pointer_cast<ngraph::Node>(op));
    }
    return nodes;
}

std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> interpreterFunction(
    const std::shared_ptr<Function>& function,
    const std::vector<std::vector<std::uint8_t>>& inputs,
    const std::vector<ngraph::element::Type>& inputTypes = {});

std::vector<ov::Tensor> interpretFunction(const std::shared_ptr<Function>& function,
                                          const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs);

//
// This function compares two nGraph functions and requires them to have exactly one output
// Check nodes types
// Check number of inputs
// Check shapes of each Node
//
void CompareFunctions(const Function& actual, const Function& expected);

std::shared_ptr<Function> foldFunction(const std::shared_ptr<Function>& function,
                                       const std::vector<std::vector<std::uint8_t>>& inputs,
                                       const std::vector<ngraph::element::Type>& inputTypes = {});

std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> getConstData(
    const std::shared_ptr<Function>& function);

std::shared_ptr<ngraph::Node> getNodeSharedPtr(const ngraph::NodeTypeInfo& type_info,
                                               const ngraph::OutputVector& outputVector);

std::vector<std::uint8_t> convertOutputPrecision(const std::vector<std::uint8_t>& output,
                                                 const element::Type_t& fromPrecision,
                                                 const element::Type_t& toPrecision,
                                                 const size_t elementsCount);

std::ostream& operator<<(std::ostream& os, TensorIteratorBody type);

std::ostream& operator<<(std::ostream& os, MemoryTransformation type);

void resize_function(std::shared_ptr<ov::Model> function, const std::vector<ov::Shape>& targetInputStaticShapes);

using ov::test::utils::operator<<;

}  // namespace helpers
}  // namespace ngraph
