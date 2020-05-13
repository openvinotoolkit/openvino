// Copyright (C) 2020 Intel Corporation
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

class ExtensionLibTests : public CommonTestUtils::TestsCommon {
public:
    std::string getExtensionPath() {
        return FileUtils::makeSharedLibraryName<char>({},
            std::string("extension_tests") + IE_BUILD_POSTFIX);
    }
};

TEST_F(ExtensionLibTests, testGetFactoryFor) {
    IExtensionPtr extension = make_so_pointer<IExtension>(getExtensionPath());
    CNNLayer testLayer({"test1", "test", Precision::FP32});
    ILayerImplFactory* factory = nullptr;
    ResponseDesc resp;
    ASSERT_EQ(OK, extension->getFactoryFor(factory, &testLayer, &resp));
}

TEST_F(ExtensionLibTests, testGetIncorrectFactoryFor) {
    IExtensionPtr extension = make_so_pointer<IExtension>(getExtensionPath());
    CNNLayer testLayer({"test1", "test_incorrect", Precision::FP32});
    ILayerImplFactory* factory = nullptr;
    ResponseDesc resp;
    ASSERT_NE(OK, extension->getFactoryFor(factory, &testLayer, &resp));
}

TEST_F(ExtensionLibTests, testGetPrimitiveTypes) {
    IExtensionPtr extension = make_so_pointer<IExtension>(getExtensionPath());
    ResponseDesc resp;
    char **types;
    unsigned int size(0);
    ASSERT_EQ(OK, extension->getPrimitiveTypes(types, size, &resp));
    ASSERT_EQ(1, size);
}

TEST_F(ExtensionLibTests, testGetShapeInferTypes) {
    IShapeInferExtensionPtr extension = make_so_pointer<IShapeInferExtension>(getExtensionPath());
    ResponseDesc resp;
    char **types;
    unsigned int size(0);
    ASSERT_EQ(OK, extension->getShapeInferTypes(types, size, &resp));
    ASSERT_EQ(1, size);
}

TEST_F(ExtensionLibTests, testGetShapeInferImpl) {
    IShapeInferExtensionPtr extension = make_so_pointer<IShapeInferExtension>(getExtensionPath());
    IShapeInferImpl::Ptr impl;
    ResponseDesc resp;
    ASSERT_EQ(OK, extension->getShapeInferImpl(impl, "test", &resp));
}

TEST_F(ExtensionLibTests, testGetIncorrectShapeInferImpl) {
    IShapeInferExtensionPtr extension = make_so_pointer<IShapeInferExtension>(getExtensionPath());
    CNNLayer testLayer({"test1", "test", Precision::FP32});
    IShapeInferImpl::Ptr impl;
    ResponseDesc resp;
    ASSERT_NE(OK, extension->getShapeInferImpl(impl, "test_incorrect", &resp));
}

TEST_F(ExtensionLibTests, testGetOpSets) {
    IExtensionPtr extension = make_so_pointer<IExtension>(getExtensionPath());
    auto opsets = extension->getOpSets();
    ASSERT_FALSE(opsets.empty());
    opsets.clear();
}

TEST_F(ExtensionLibTests, testGetImplTypes) {
    IExtensionPtr extension = make_so_pointer<IExtension>(getExtensionPath());
    auto opset = extension->getOpSets().begin()->second;
    std::shared_ptr<ngraph::Node> op(opset.create(opset.get_types_info().begin()->name));
    ASSERT_FALSE(extension->getImplTypes(op).empty());
}

TEST_F(ExtensionLibTests, testGetImplementation) {
    IExtensionPtr extension = make_so_pointer<IExtension>(getExtensionPath());
    auto opset = extension->getOpSets().begin()->second;
    std::shared_ptr<ngraph::Node> op(opset.create(opset.get_types_info().begin()->name));
    ASSERT_NE(nullptr, extension->getImplementation(op, extension->getImplTypes(op)[0]));
}
