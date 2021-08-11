// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ngraph/check.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "low_precision/lpt_visibility.hpp"
#include "transformations/rt_info/dequantization_attribute.hpp"

namespace ov {
namespace pass {
namespace low_precision {

// template<typename BaseOp2>
// class LP_TRANSFORMATIONS_API DequantizationOp : public BaseOp2 {
// public:
//    template <typename ... Args>
//    DequantizationOp(Args&&... args) : BaseOp2(std::forward<Args>(args)...) {
//        init();
//    }
//
//    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
//        std::shared_ptr<Node> cloned = BaseOp2::clone_with_new_inputs(inputs);
//        auto& rtInfo = cloned->get_rt_info();
//        rtInfo = get_rt_info();
//
//        return cloned;
//    }
//
// protected:
//    void init() {
//        auto& rtInfo = get_rt_info();
//        rtInfo["DEQUANTIZATION"] = std::make_shared<ov::VariantWrapper<std::string>>("");
//    }
// };
//
// using DequantizationConvert = DequantizationOp<ov::opset1::Convert>;
// using DequantizationSubtract = DequantizationOp<ov::opset1::Subtract>;
// using DequantizationMultiply = DequantizationOp<ov::opset1::Multiply>;

namespace {
void initRuntimeInfo(ov::Node& operation) {
    auto& rtInfo = operation.get_rt_info();
    rtInfo["DEQUANTIZATION"] = std::make_shared<VariantWrapper<DequantizationAttr>>(DequantizationAttr());
}

// #include <ngraph/rt_info.hpp>
// ov::copy_runtime_info(from, to);
void copyRuntimeInfo(const ov::Node& from, ov::Node& to) {
    const auto& rtInfoFrom = from.get_rt_info();
    auto& rtInfoTo = to.get_rt_info();
    rtInfoTo = rtInfoFrom;
}

} // namespace

class LP_TRANSFORMATIONS_API DequantizationConvert : public ov::opset1::Convert {
public:
    DequantizationConvert(const ov::Output<Node>& arg, const ov::element::Type& destination_type) :
        ov::opset1::Convert(arg, destination_type) {
        initRuntimeInfo(*this);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        std::shared_ptr<Node> cloned = ov::opset1::Convert::clone_with_new_inputs(inputs);
        copyRuntimeInfo(*this, *cloned);
        return cloned;
    }
};

class LP_TRANSFORMATIONS_API DequantizationSubtract : public ov::opset1::Subtract {
public:
    DequantizationSubtract(
        const ov::Output<Node>& arg0,
        const ov::Output<Node>& arg1,
        const ov::op::AutoBroadcastSpec& auto_broadcast = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY)) :
        ov::opset1::Subtract(arg0, arg1, auto_broadcast) {
        initRuntimeInfo(*this);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        std::shared_ptr<Node> cloned = ov::opset1::Subtract::clone_with_new_inputs(inputs);
        copyRuntimeInfo(*this, *cloned);
        return cloned;
    }
};

class LP_TRANSFORMATIONS_API DequantizationMultiply : public ov::opset1::Multiply {
public:
    DequantizationMultiply(
        const Output<Node>& arg0,
        const Output<Node>& arg1,
        const ov::op::AutoBroadcastSpec& auto_broadcast = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY)) :
        ov::opset1::Multiply(arg0, arg1, auto_broadcast) {
        initRuntimeInfo(*this);
    }

    DequantizationMultiply(const ov::opset1::Multiply& multiply) :
        ov::opset1::Multiply(multiply) {
        initRuntimeInfo(*this);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        std::shared_ptr<Node> cloned = ov::opset1::Multiply::clone_with_new_inputs(inputs);
        copyRuntimeInfo(*this, *cloned);
        return cloned;
    }
};

class LP_TRANSFORMATIONS_API DequantizationAdd : public ov::opset1::Add {
public:
    DequantizationAdd(
        const ov::Output<Node>& arg0,
        const ov::Output<Node>& arg1,
        const ov::op::AutoBroadcastSpec& auto_broadcast = ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY)) :
        ov::opset1::Add(arg0, arg1, auto_broadcast) {
        initRuntimeInfo(*this);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        std::shared_ptr<Node> cloned = ov::opset1::Add::clone_with_new_inputs(inputs);
        copyRuntimeInfo(*this, *cloned);
        return cloned;
    }
};

} // namespace low_precision
} // namespace pass
} // namespace ov
