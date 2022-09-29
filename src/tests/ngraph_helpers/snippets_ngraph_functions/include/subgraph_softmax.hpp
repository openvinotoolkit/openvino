// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"
#include "./snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

class SoftmaxFunction : public SnippetsFunctionBase {
public:
    explicit SoftmaxFunction(const std::vector<PartialShape>& inputShapes, int axis) : SnippetsFunctionBase(inputShapes), axis(axis) {
        NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    int axis;
};

class SinhSoftmaxFunction : public SnippetsFunctionBase {
public:
    explicit SinhSoftmaxFunction(const std::vector<PartialShape>& inputShapes, int axis) : SnippetsFunctionBase(inputShapes), axis(axis) {
        NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    int axis;
};

class AddSoftmaxFunction : public SnippetsFunctionBase {
public:
    explicit AddSoftmaxFunction(const std::vector<PartialShape>& inputShapes, int axis) : SnippetsFunctionBase(inputShapes), axis(axis) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    int axis;
};

class SinhAddSoftmaxFunction : public SnippetsFunctionBase {
public:
    explicit SinhAddSoftmaxFunction(const std::vector<PartialShape>& inputShapes, int axis) : SnippetsFunctionBase(inputShapes), axis(axis) {
        NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    int axis;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
