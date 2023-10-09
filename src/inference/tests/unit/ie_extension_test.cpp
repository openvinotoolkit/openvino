// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include <gtest/gtest.h>
#include <ie_extension.h>

#include <memory>
#include <ngraph/opsets/opset.hpp>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_common.hpp"

using namespace InferenceEngine;

using ExtensionTests = ::testing::Test;

#ifndef OPENVINO_STATIC_LIBRARY

OPENVINO_SUPPRESS_DEPRECATED_START

static std::string getExtensionPath() {
    return FileUtils::makePluginLibraryName<char>(ov::test::utils::getExecutableDirectory(),
                                                  std::string("template_extension") + OV_BUILD_POSTFIX);
}

TEST(ExtensionTests, testGetOpSets) {
    IExtensionPtr extension = std::make_shared<Extension>(getExtensionPath());
    auto opsets = extension->getOpSets();
    ASSERT_FALSE(opsets.empty());
    opsets.clear();
}

TEST(ExtensionTests, testGetImplTypes) {
    IExtensionPtr extension = std::make_shared<Extension>(getExtensionPath());
    auto opset = extension->getOpSets().begin()->second;
    std::shared_ptr<ngraph::Node> op(opset.create(opset.get_types_info().begin()->name));
    ASSERT_FALSE(extension->getImplTypes(op).empty());
}

TEST(ExtensionTests, testGetImplTypesThrowsIfNgraphNodeIsNullPtr) {
    IExtensionPtr extension = std::make_shared<Extension>(getExtensionPath());
    ASSERT_THROW(extension->getImplTypes(std::shared_ptr<ngraph::Node>()), InferenceEngine::Exception);
}

TEST(ExtensionTests, testGetImplementation) {
    IExtensionPtr extension = std::make_shared<Extension>(getExtensionPath());
    auto opset = extension->getOpSets().begin()->second;
    std::shared_ptr<ngraph::Node> op(opset.create("Template"));
    ASSERT_NE(nullptr, extension->getImplementation(op, extension->getImplTypes(op)[0]));
}

TEST(ExtensionTests, testGetImplementationThrowsIfNgraphNodeIsNullPtr) {
    IExtensionPtr extension = std::make_shared<Extension>(getExtensionPath());
    ASSERT_THROW(extension->getImplementation(std::shared_ptr<ngraph::Node>(), ""), InferenceEngine::Exception);
}

OPENVINO_SUPPRESS_DEPRECATED_END

#endif  // OPENVINO_STATIC_LIBRARY
