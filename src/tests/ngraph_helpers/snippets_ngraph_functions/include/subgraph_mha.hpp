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
class MHASinhFunction : public SnippetsFunctionBase {
public:
    explicit MHASinhFunction(const std::vector<PartialShape>& inputShapes, bool with_mul = true)
        : SnippetsFunctionBase(inputShapes), with_mul(with_mul) {
        NGRAPH_CHECK(input_shapes.size() == 4, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    bool with_mul = true;
};

// TODO: Write Graph
class MHAMatMul0TransposeFunction : public MHAFunction {
public:
    explicit MHAMatMul0TransposeFunction(const std::vector<PartialShape>& inputShapes) : MHAFunction(inputShapes) {}
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

// TODO: Write Graph
class MHASelectSinhFunction : public SnippetsFunctionBase {
public:
    explicit MHASelectSinhFunction(const std::vector<PartialShape>& inputShapes) : SnippetsFunctionBase(inputShapes) {
        NGRAPH_CHECK(input_shapes.size() == 6, "Got invalid number of input shapes");
    }
protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
