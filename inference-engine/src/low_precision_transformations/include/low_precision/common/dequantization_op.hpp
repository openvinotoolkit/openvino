// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "ngraph/op/multiply.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/add.hpp"

#include "low_precision/lpt_visibility.hpp"
#include "transformations/rt_info/dequantization_attribute.hpp"

namespace ngraph {
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
//        rtInfo["DEQUANTIZATION"] = std::make_shared<ngraph::VariantWrapper<std::string>>("");
//    }
// };
//
// using DequantizationConvert = DequantizationOp<ngraph::op::v0::Convert>;
// using DequantizationSubtract = DequantizationOp<ngraph::op::v1::Subtract>;
// using DequantizationMultiply = DequantizationOp<ngraph::op::v1::Multiply>;

namespace {
void initRuntimeInfo(ngraph::Node& operation) {
    auto& rtInfo = operation.get_rt_info();
    rtInfo["DEQUANTIZATION"] = std::make_shared<VariantWrapper<DequantizationAttr>>(DequantizationAttr());
}

// #include <ngraph/rt_info.hpp>
// ngraph::copy_runtime_info(from, to);
void copyRuntimeInfo(const ngraph::Node& from, ngraph::Node& to) {
    const auto& rtInfoFrom = from.get_rt_info();
    auto& rtInfoTo = to.get_rt_info();
    rtInfoTo = rtInfoFrom;
}

} // namespace

class LP_TRANSFORMATIONS_API DequantizationConvert : public ngraph::op::v0::Convert {
public:
    DequantizationConvert(const ngraph::Output<Node>& arg, const ngraph::element::Type& destination_type) :
        ngraph::op::v0::Convert(arg, destination_type) {
        initRuntimeInfo(*this);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        std::shared_ptr<Node> cloned = ngraph::op::v0::Convert::clone_with_new_inputs(inputs);
        copyRuntimeInfo(*this, *cloned);
        return cloned;
    }
};

class LP_TRANSFORMATIONS_API DequantizationSubtract : public ngraph::op::v1::Subtract {
public:
    DequantizationSubtract(
        const ngraph::Output<Node>& arg0,
        const ngraph::Output<Node>& arg1,
        const ngraph::op::AutoBroadcastSpec& auto_broadcast = ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY)) :
        ngraph::op::v1::Subtract(arg0, arg1, auto_broadcast) {
        initRuntimeInfo(*this);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        std::shared_ptr<Node> cloned = ngraph::op::v1::Subtract::clone_with_new_inputs(inputs);
        copyRuntimeInfo(*this, *cloned);
        return cloned;
    }
};

class LP_TRANSFORMATIONS_API DequantizationMultiply : public ngraph::op::v1::Multiply {
public:
    DequantizationMultiply(
        const Output<Node>& arg0,
        const Output<Node>& arg1,
        const ngraph::op::AutoBroadcastSpec& auto_broadcast = ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY)) :
        ngraph::op::v1::Multiply(arg0, arg1, auto_broadcast) {
        initRuntimeInfo(*this);
    }

    DequantizationMultiply(const ngraph::op::v1::Multiply& multiply) :
        ngraph::op::v1::Multiply(multiply) {
        initRuntimeInfo(*this);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        std::shared_ptr<Node> cloned = ngraph::op::v1::Multiply::clone_with_new_inputs(inputs);
        copyRuntimeInfo(*this, *cloned);
        return cloned;
    }
};

class LP_TRANSFORMATIONS_API DequantizationAdd : public ngraph::op::v1::Add {
public:
    DequantizationAdd(
        const ngraph::Output<Node>& arg0,
        const ngraph::Output<Node>& arg1,
        const ngraph::op::AutoBroadcastSpec& auto_broadcast = ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY)) :
        ngraph::op::v1::Add(arg0, arg1, auto_broadcast) {
        initRuntimeInfo(*this);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        std::shared_ptr<Node> cloned = ngraph::op::v1::Add::clone_with_new_inputs(inputs);
        copyRuntimeInfo(*this, *cloned);
        return cloned;
    }
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
