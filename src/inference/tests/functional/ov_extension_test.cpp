// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_extension_test.hpp"

#include "common_test_utils/file_utils.hpp"
#include "openvino/util/file_util.hpp"

using namespace testing;
using namespace ov::test::utils;

#if defined(ENABLE_OV_IR_FRONTEND)
namespace {

std::string getOVExtensionPath() {
    return ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                              std::string("openvino_template_extension") + OV_BUILD_POSTFIX);
}

std::string getIncorrectExtensionPath() {
    return ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                              std::string("incorrect") + OV_BUILD_POSTFIX);
}

std::string getRelativeOVExtensionPath() {
    std::string absolutePath =
        ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                           std::string("openvino_template_extension") + OV_BUILD_POSTFIX);
    return ov::test::utils::getRelativePath(ov::test::utils::getCurrentWorkingDir(), absolutePath);
}

}  // namespace
#endif

class CustomNewIdentity : public ov::op::Op {
public:
    OPENVINO_OP("Identity")

    CustomNewIdentity() = default;
    CustomNewIdentity(const ov::Output<ov::Node>& arg) : Op({arg}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() != 1, "Incorrect number of new arguments");

        return std::make_shared<CustomNewIdentity>(new_args.at(0));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }
};

class CustomReLU : public ov::op::Op {
public:
    OPENVINO_OP("CustomReLU")

    CustomReLU() = default;
    CustomReLU(const ov::Output<ov::Node>& arg) : Op({arg}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        OPENVINO_ASSERT(new_args.size() != 1, "Incorrect number of new arguments");

        return std::make_shared<CustomReLU>(new_args.at(0));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }
};

#if defined(ENABLE_OV_IR_FRONTEND)
TEST_F(OVExtensionTests, ReshapeIRWithNewExtensionsPathLib) {
    core.add_extension(std::filesystem::path(getOVExtensionPath()));
    test();
}

TEST_F(OVExtensionTests, ReshapeIRWithNewExtensionsLib) {
    core.add_extension(getOVExtensionPath());
    test();
}

TEST_F(OVExtensionTests, ReshapeIRWithNewExtensionPtr) {
    core.add_extension(std::make_shared<ov::OpExtension<CustomNewIdentity>>());
    test();
}

TEST_F(OVExtensionTests, ReshapeIRWithNewExtension) {
    core.add_extension(ov::OpExtension<CustomNewIdentity>());
    test();
}

TEST_F(OVExtensionTests, ReshapeIRWithNewOp) {
    core.add_extension<CustomNewIdentity>();
    test();
}

TEST_F(OVExtensionTests, IncorrectReshapeIRWithNewExtensionPtr) {
    core.add_extension(std::make_shared<ov::OpExtension<CustomNewIdentity>>());
    EXPECT_ANY_THROW(test_two_op());
}

TEST_F(OVExtensionTests, IncorrectReshapeIRWithNewExtension) {
    core.add_extension(ov::OpExtension<CustomNewIdentity>());
    EXPECT_ANY_THROW(test_two_op());
}

TEST_F(OVExtensionTests, IncorrectReshapeIRWithNewOp) {
    core.add_extension<CustomNewIdentity>();
    EXPECT_ANY_THROW(test_two_op());
}

TEST_F(OVExtensionTests, ReshapeIRWithSeveralNewExtensionPtrs) {
    core.add_extension(
        {std::make_shared<ov::OpExtension<CustomNewIdentity>>(), std::make_shared<ov::OpExtension<CustomReLU>>()});
    test_two_op();
}

TEST_F(OVExtensionTests, ReshapeIRWithSeveralNewExtensions) {
    core.add_extension(ov::OpExtension<CustomNewIdentity>(), ov::OpExtension<CustomReLU>());
    test_two_op();
}

TEST_F(OVExtensionTests, ReshapeIRWithSeveralNewOps) {
    core.add_extension<CustomNewIdentity, CustomReLU>();
    test_two_op();
}

TEST_F(OVExtensionTests, load_new_extension) {
    EXPECT_NO_THROW(core.add_extension(getOVExtensionPath()));
}

TEST_F(OVExtensionTests, load_incorrect_extension) {
    EXPECT_THROW(core.add_extension(getIncorrectExtensionPath()), ov::Exception);
}

TEST_F(OVExtensionTests, load_relative) {
    EXPECT_NO_THROW(core.add_extension(getRelativeOVExtensionPath()));
}

#endif  // defined(ENABLE_OV_IR_FRONTEND)
