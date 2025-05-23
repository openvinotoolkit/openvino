// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/ov_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/pass/manager.hpp"

namespace transpose_sinking {
namespace testing {

struct Preprocessing {
    std::vector<std::function<ov::OutputVector(std::vector<size_t>, ov::OutputVector)>> preprocessing;
    std::vector<std::vector<size_t>> indices;

    ov::OutputVector apply(const ov::OutputVector& inputs) const {
        ov::OutputVector new_inputs = inputs;
        for (size_t i = 0; i < preprocessing.size(); ++i) {
            new_inputs = preprocessing[i](indices[i], new_inputs);
        }
        return new_inputs;
    }
};

class IFactory {
public:
    explicit IFactory(const std::string& type_name) : type_name_(type_name) {}
    virtual ~IFactory() = default;
    virtual std::shared_ptr<ov::Node> create(const ov::OutputVector& parent_nodes) const = 0;

    const std::string& getTypeName() const {
        return type_name_;
    }

private:
    const std::string type_name_;
};
using FactoryPtr = std::shared_ptr<IFactory>;

class IPassFactory {
public:
    explicit IPassFactory(const std::string& type_name) : type_name_(type_name) {}
    virtual ~IPassFactory() = default;
    virtual void registerPass(ov::pass::Manager& pass_manager) const = 0;
    const std::string& getTypeName() const {
        return type_name_;
    }

private:
    const std::string type_name_;
};

template <typename PassT>
class PassFactory : public IPassFactory {
public:
    explicit PassFactory(const std::string& type_name) : IPassFactory(type_name) {}
    void registerPass(ov::pass::Manager& pass_manager) const override {
        pass_manager.register_pass<PassT>();
    }
};
using PassFactoryPtr = std::shared_ptr<IPassFactory>;
#define CREATE_PASS_FACTORY(pass_name) std::make_shared<PassFactory<ov::pass::transpose_sinking::pass_name>>(#pass_name)

struct TestCase;
struct ModelDescription;
using TestParams = std::tuple<size_t /* idx num_main_ops */, size_t /* idx main_op */, TestCase>;
using CreateGraphF = std::function<
    std::shared_ptr<ov::Model>(size_t main_op_idx, const ModelDescription&, size_t, const ov::OutputVector&)>;

// Describes a model to test.
// Expects to be used in such a scenario:
// 1st Preprocessing inserts Transpose/Gather to the inputs
// of the main node.
// Factory contains the rules how to create the main testing node.
// 2nd Preprocessing inserts Transpose/Gather to the outputs
// of the main node.
// model_template is a function which uses the arguments above.
// Examples of the scenarios:
// ModelDescription model: Param -> (Transpose inserted by 1st Preprocessing) -> Abs (main_node) -> Result
// ModelDescription reference: Param -> Abs (main_node) -> (Transpose inserted by 2nd Preprocessing) -> Result
struct ModelDescription {
    Preprocessing preprocess_inputs_to_main;
    // @parameterized with multiple values
    std::vector<FactoryPtr> main_op;
    Preprocessing preprocess_outputs_of_main;
    CreateGraphF model_template;
};

struct TestCase {
    ov::OutputVector inputs_to_main;
    // @parameterized with multiple values
    std::vector<size_t> num_main_ops;

    ModelDescription model;
    ModelDescription model_ref;
    PassFactoryPtr transformation;
};

class TSTestFixture : public ::testing::WithParamInterface<TestParams>, public TransformationTestsF {
public:
    static std::string get_test_name(const ::testing::TestParamInfo<TestParams>& obj);
};

}  // namespace testing
}  // namespace transpose_sinking
