// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/model.hpp"
#include "common_test_utils/ov_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {
using ov::Model;

/// \brief Base class for snippets-related subgraphs. Note that inputShapes size is model-specific
/// and expected to be checked inside a child class constructor.
/// \param inputShapes vector of input shapes accepted by the underlying model
class SnippetsFunctionBase {
public:
    SnippetsFunctionBase() = delete;
    virtual ~SnippetsFunctionBase() = default;

    explicit SnippetsFunctionBase(const std::vector<PartialShape>& inputShapes, ov::element::Type_t precision = element::f32)
                : precision{precision}, input_shapes{inputShapes} {}

    std::shared_ptr<Model> getReference() const {
        std::shared_ptr<Model> function_ref = initReference();
        validate_function(function_ref);
        return function_ref;
    }

    std::shared_ptr<Model> getOriginal() const {
        std::shared_ptr<Model> function = initOriginal();
        validate_function(function);
        return function;
    }

    std::shared_ptr<Model> getLowered() const {
        std::shared_ptr<Model> function_low = initLowered();
        validate_function(function_low);
        return function_low;
    }

    size_t getNumInputs() const { return input_shapes.size(); }

protected:
    virtual std::shared_ptr<Model> initOriginal() const = 0;

    virtual std::shared_ptr<Model> initReference() const {
        throw std::runtime_error("initReference() for this class is not implemented");
    }

    virtual std::shared_ptr<Model> initLowered() const {
        throw std::runtime_error("initLowered() for this class is not implemented");
    }

    const ov::element::Type_t precision;
    const std::vector<PartialShape> input_shapes;

    virtual void validate_function(const std::shared_ptr<Model> &f) const;
    static void validate_params_shape(const std::vector<PartialShape>& input_shapes, const ov::ParameterVector& params);
};

/// \brief Base class for snippets subgraphs with customizable embedded op sequences. Note that the custom_ops allowed types are
/// model-specific and expected to be checked inside a child class constructor.
/// \param  custom_ops  vector of ops to be inserted in the graph. Required vector size and acceptable op types are subgraph-specific.
/// The ops are expected to be default-constructed to facilitate test development, the class will take care of the ops inputs for you.
/// \param  customOpsNumInputs  size_t vector that specifies the number of inputs for each of the custom_ops. Not that an rvalue is expected,
/// since it should be hard-coded along with the Original and Reference functions.
class SnippetsFunctionCustomizable : public SnippetsFunctionBase {
public:
    SnippetsFunctionCustomizable() = delete;
    SnippetsFunctionCustomizable(const std::vector<PartialShape>& inputShapes,
                                 const std::vector<std::shared_ptr<Node>>& customOps,
                                 const std::vector<size_t>&& customOpsNumInputs);

protected:
    std::vector<std::shared_ptr<Node>> custom_ops;
    std::vector<size_t> custom_ops_num_inputs;
    void ResetCustomOpsInputs();
};
}  // namespace snippets
}  // namespace test
}  // namespace ov
