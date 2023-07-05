// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <common_test_utils/ngraph_test_utils.hpp>
#include <openvino/op/pad.hpp>

#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"

class IPadFactory {
public:
    explicit IPadFactory(const std::string& type_name) : type_name_(type_name) {}
    virtual ~IPadFactory() = default;
    virtual std::shared_ptr<ov::Node> create(const ov::Output<ov::Node>& arg,
                                             const ov::Output<ov::Node>& pads_begin,
                                             const ov::Output<ov::Node>& pads_end,
                                             ov::op::PadMode pad_mode) const = 0;
    virtual std::shared_ptr<ov::Node> create(const ov::Output<ov::Node>& arg,
                                             const ov::Output<ov::Node>& pads_begin,
                                             const ov::Output<ov::Node>& pads_end,
                                             const ov::Output<ov::Node>& arg_pad_value,
                                             ov::op::PadMode pad_mode) const = 0;

    const std::string& getTypeName() const {
        return type_name_;
    }

private:
    const std::string type_name_;
};

template <typename PadT>
class PadFactory : public IPadFactory {
public:
    explicit PadFactory(const std::string& type_name) : IPadFactory(type_name) {}
    std::shared_ptr<ov::Node> create(const ov::Output<ov::Node>& arg,
                                     const ov::Output<ov::Node>& pads_begin,
                                     const ov::Output<ov::Node>& pads_end,
                                     ov::op::PadMode pad_mode) const override {
        return std::make_shared<PadT>(arg, pads_begin, pads_end, pad_mode);
    }
    std::shared_ptr<ov::Node> create(const ov::Output<ov::Node>& arg,
                                     const ov::Output<ov::Node>& pads_begin,
                                     const ov::Output<ov::Node>& pads_end,
                                     const ov::Output<ov::Node>& arg_pad_value,
                                     ov::op::PadMode pad_mode) const override {
        return std::make_shared<PadT>(arg, pads_begin, pads_end, arg_pad_value, pad_mode);
    }
};

template <typename PadT>
std::shared_ptr<IPadFactory> CreatePadFactory(const std::string& type_name) {
    return std::make_shared<PadFactory<PadT>>(type_name);
}

struct ITestModelFactory {
    explicit ITestModelFactory(const std::string& a_test_name) : test_name(a_test_name) {}
    virtual ~ITestModelFactory() = default;
    virtual void setup(std::shared_ptr<IPadFactory> pad_factory, ov::pass::Manager& manager) = 0;
    std::string test_name;
    std::shared_ptr<ov::Model> function;
    std::shared_ptr<ov::Model> function_ref;
};

class PadTestFixture : public ::testing::WithParamInterface<
                           std::tuple<std::shared_ptr<IPadFactory>, std::shared_ptr<ITestModelFactory>>>,
                       public TransformationTestsF {
public:
    static std::string get_test_name(
        const ::testing::TestParamInfo<std::tuple<std::shared_ptr<IPadFactory>, std::shared_ptr<ITestModelFactory>>>&
            obj);
};

#define PAD_TEST_BODY(TestName)                                                                    \
    struct TestName : public ITestModelFactory {                                                   \
        TestName() : ITestModelFactory(#TestName) {}                                               \
        void setup(std::shared_ptr<IPadFactory> pad_factory, ov::pass::Manager& manager) override; \
    };                                                                                             \
    void TestName::setup(std::shared_ptr<IPadFactory> pad_factory, ov::pass::Manager& manager)
