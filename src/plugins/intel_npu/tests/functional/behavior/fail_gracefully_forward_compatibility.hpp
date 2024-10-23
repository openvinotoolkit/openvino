// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <exception>
#include <openvino/core/any.hpp>
#include <openvino/core/node_vector.hpp>
#include <openvino/op/op.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>

#include "base/ov_behavior_test_utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "intel_npu/config/common.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

using ::testing::AllOf;
using ::testing::HasSubstr;

namespace ov {
namespace test {
namespace behavior {

class UnsupportedTestOperation : public ov::op::Op {
public:
    OPENVINO_OP("UnsupportedTestOperation");

    UnsupportedTestOperation() = default;
    explicit UnsupportedTestOperation(const ov::Output<ov::Node>& arg) : Op({arg}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto input_pshape = get_input_partial_shape(0);
        auto input_shape = input_pshape.to_shape();
        ov::Shape output_shape(input_shape);
        set_output_type(0, get_input_element_type(0), ov::PartialShape(output_shape));
    }

    bool visit_attributes(AttributeVisitor& /*visitor*/) override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");
        return std::make_shared<UnsupportedTestOperation>(new_args.at(0));
    }
};

class FailGracefullyTest : public ov::test::behavior::OVPluginTestBase,
                           public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> ov_model;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
        ov_model = createModelWithUnknownNode();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

private:
    std::shared_ptr<ov::Model> createModelWithUnknownNode() {
        const ov::Shape input_shape = {1, 4096};
        const ov::element::Type precision = ov::element::f32;

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{input_shape})};
        auto constant = ov::test::utils::make_constant(precision, ov::Shape{4096, 1024});
        auto custom_op = std::make_shared<UnsupportedTestOperation>(constant);

        ov::NodeVector results{custom_op};
        return std::make_shared<ov::Model>(results, ov::ParameterVector{params}, "CustomOpModel");
    }
};

TEST_P(FailGracefullyTest, OnUnsupprotedOperator) {
    auto compilerType = configuration[ov::intel_npu::compiler_type.name()].as<std::string>();
    try {
        core->compile_model(ov_model, target_device, configuration);
    } catch (std::exception& ex) {
        // TODO: the below error messages will be improved in E#64716
        if (compilerType == "MLIR") {
            EXPECT_THAT(ex.what(),
                        AllOf(HasSubstr("Unsupported operation"), HasSubstr("with type UnsupportedTestOperation")));
        } else if (compilerType == "DRIVER") {
            EXPECT_THAT(ex.what(), AllOf(HasSubstr("Failed to compile network")));
        }
        return;
    }

    ASSERT_FALSE(true) << "Oops, compilation of unsupported op happened";
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
