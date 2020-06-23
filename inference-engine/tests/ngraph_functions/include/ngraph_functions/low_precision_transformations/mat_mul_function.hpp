// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

class MatMulFunctionConvert {
public:
    MatMulFunctionConvert() : precision(ngraph::element::undefined) {}
    MatMulFunctionConvert(ngraph::element::Type precision) : precision(precision) {}
    bool empty() const { return precision == ngraph::element::undefined; }
    ngraph::element::Type precision;
};

class MatMulFunctionSubConst {
public:
    MatMulFunctionSubConst() {}
    MatMulFunctionSubConst(ngraph::Shape shape, std::vector<float> values) : shape(shape), values(values) {}
    bool empty() const { return values.empty(); }
    ngraph::Shape shape;
    std::vector<float> values;
};

class MatMulFunctionMultiplyConst {
public:
    MatMulFunctionMultiplyConst() {}
    MatMulFunctionMultiplyConst(ngraph::Shape shape, std::vector<float> values) : shape(shape), values(values) {}
    bool empty() const { return shape.empty() && values.empty(); }
    ngraph::Shape shape;
    std::vector<float> values;
};

class MatMulFunctionBranch {
public:
    // MatMulFunctionBranch() {}
    ngraph::Shape shape;
    MatMulFunctionConvert convert1;
    MatMulFunctionConvert convert2;
    MatMulFunctionSubConst subConst;
    MatMulFunctionMultiplyConst multiplyConst;
};

typedef std::pair<ngraph::builder::subgraph::MatMulFunctionBranch, ngraph::builder::subgraph::MatMulFunctionBranch> MatMulFunctionBranches;

class MatMulFunction {
public:
    // TODO: move to base class
    static std::vector<std::shared_ptr<ngraph::op::Parameter>> getInputs(const std::vector<std::shared_ptr<ngraph::Node>>& nodes);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type ngPrecision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::MatMulFunctionBranches& branches);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type ngPrecision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::MatMulFunctionBranches& branches);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
