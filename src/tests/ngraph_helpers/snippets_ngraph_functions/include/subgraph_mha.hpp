// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "./snippets_helpers.hpp"


namespace ov {
namespace test {
namespace snippets {

// TODO: Write Graph
class MHAFunction : public SnippetsFunctionBase {
public:
    explicit MHAFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 4, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

// TODO: Write Graph
class MHAMatMul0TransposeFunction : public MHAFunction {
public:
    explicit MHAMatMul0TransposeFunction(const std::vector<PartialShape>& inputShapes) : MHAFunction(inputShapes) {}
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

// TODO: Write Graph
class MHASelectFunction : public SnippetsFunctionBase {
public:
    explicit MHASelectFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 5, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

class TransposeSoftmaxFunction : public SnippetsFunctionBase {
public:
    explicit TransposeSoftmaxFunction(const std::vector<PartialShape>& inputShapes, const std::vector<int64_t>& order, const int64_t axis)
        : SnippetsFunctionBase(inputShapes), m_order(order), m_axis(axis) {
        NGRAPH_CHECK(input_shapes.size() > 0, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    std::vector<int64_t> m_order;
    int64_t m_axis;
};


class TransposeSoftmaxEltwiseFunction : public TransposeSoftmaxFunction {
public:
    explicit TransposeSoftmaxEltwiseFunction(const std::vector<PartialShape>& inputShapes, const std::vector<int64_t>& order, const int64_t axis)
        : TransposeSoftmaxFunction(inputShapes, order, axis) {}
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};


}  // namespace snippets
}  // namespace test
}  // namespace ov
