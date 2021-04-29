// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string>
#include <memory>

#include <ie_extension.h>
#include <file_utils.h>

#include <ngraph/opsets/opset.hpp>

#include "common_test_utils/test_common.hpp"

IE_SUPPRESS_DEPRECATED_START

using namespace InferenceEngine;

using ExtensionTests = ::testing::Test;

std::string getExtensionPath() {
    return FileUtils::makePluginLibraryName<char>({},
            std::string("template_extension") + IE_BUILD_POSTFIX);
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
    ASSERT_THROW(extension->getImplTypes(std::shared_ptr<ngraph::Node> ()),
            InferenceEngine::Exception);
}

TEST(ExtensionTests, testGetImplementation) {
    IExtensionPtr extension = std::make_shared<Extension>(getExtensionPath());
    auto opset = extension->getOpSets().begin()->second;
    std::shared_ptr<ngraph::Node> op(opset.create("Template"));
    ASSERT_NE(nullptr, extension->getImplementation(op, extension->getImplTypes(op)[0]));
}

TEST(ExtensionTests, testGetImplementationThrowsIfNgraphNodeIsNullPtr) {
    IExtensionPtr extension = std::make_shared<Extension>(getExtensionPath());
    ASSERT_THROW(extension->getImplementation(std::shared_ptr<ngraph::Node> (), ""),
            InferenceEngine::Exception);
}
