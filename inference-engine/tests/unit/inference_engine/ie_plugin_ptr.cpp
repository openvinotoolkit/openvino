// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "so_ptr.hpp"
#include "openvino/util/shared_object.hpp"

#include "unit_test_utils/mocks/mock_engine/mock_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"


using namespace InferenceEngine;
using namespace ::testing;
using namespace InferenceEngine::details;

class PluginTest: public ::testing::Test {
protected:
    std::shared_ptr<void> sharedObjectLoader;
    std::function<IInferencePlugin*(IInferencePlugin*)> createPluginEngineProxy;
    ov::runtime::SoPtr<InferenceEngine::IInferencePlugin> getPtr();

    std::string get_mock_engine_name() {
        std::string mockEngineName("mock_engine");
        return CommonTestUtils::pre + mockEngineName + IE_BUILD_POSTFIX + CommonTestUtils::ext;
    }

    void SetUp() override {
        std::string libraryName = get_mock_engine_name();
        sharedObjectLoader = ov::util::load_shared_object(libraryName.c_str());
        createPluginEngineProxy = make_std_function<IInferencePlugin*(IInferencePlugin*)>("CreatePluginEngineProxy");
    }
    template <class T>
    std::function<T> make_std_function(const std::string& functionName) {
        std::function <T> ptr(reinterpret_cast<T*>(
            ov::util::get_symbol(sharedObjectLoader, functionName.c_str())));
        return ptr;
    }

    MockInferencePluginInternal2 engine;
};

#ifndef OPENVINO_STATIC_LIBRARY

TEST_F(PluginTest, canCreatePluginUsingSmartPtr) {
    ASSERT_NO_THROW(ov::runtime::SoPtr<InferenceEngine::IInferencePlugin> ptr(get_mock_engine_name()));
}

TEST_F(PluginTest, shouldThrowExceptionIfPluginNotExist) {
    EXPECT_THROW(ov::runtime::SoPtr<InferenceEngine::IInferencePlugin>(std::string{"unknown_plugin"}), Exception);
}

ov::runtime::SoPtr<InferenceEngine::IInferencePlugin> PluginTest::getPtr() {
    ov::runtime::SoPtr<InferenceEngine::IInferencePlugin> smart_ptr(get_mock_engine_name());
    return smart_ptr;
}

TEST_F(PluginTest, canSetConfiguration) {
    ov::runtime::SoPtr<InferenceEngine::IInferencePlugin> ptr = getPtr();
    // TODO: dynamic->reinterpret because of clang/gcc cannot
    // dynamically cast this MOCK object
    ASSERT_TRUE(dynamic_cast<MockPlugin*>(ptr.operator->())->config.empty());

    std::map<std::string, std::string> config = { { "key", "value" } };
    ASSERT_NO_THROW(ptr->SetConfig(config));
    config.clear();

    ASSERT_STREQ(dynamic_cast<MockPlugin*>(ptr.operator->())->config["key"].c_str(), "value");
}

#endif // OPENVINO_STATIC_LIBRARY
