// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "details/ie_so_loader.h"

#include "unit_test_utils/mocks/mock_engine/mock_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"


using namespace std;
using namespace InferenceEngine;
using namespace ::testing;
using namespace InferenceEngine::details;

class PluginTest: public ::testing::Test {
protected:
    unique_ptr<SharedObjectLoader> sharedObjectLoader;
    std::function<IInferencePlugin*(IInferencePlugin*)> createPluginEngineProxy;
    InferenceEngine::details::SOPointer<InferenceEngine::IInferencePlugin> getPtr();

    std::string get_mock_engine_name() {
        std::string mockEngineName("mock_engine");
        return CommonTestUtils::pre + mockEngineName + IE_BUILD_POSTFIX + CommonTestUtils::ext;
    }

    virtual void SetUp() {
        std::string libraryName = get_mock_engine_name();
        sharedObjectLoader.reset(new SharedObjectLoader(libraryName.c_str()));
        createPluginEngineProxy = make_std_function<IInferencePlugin*(IInferencePlugin*)>("CreatePluginEngineProxy");
    }
    template <class T>
    std::function<T> make_std_function(const std::string& functionName) {
        std::function <T> ptr(reinterpret_cast<T*>(sharedObjectLoader->get_symbol(functionName.c_str())));
        return ptr;
    }

    MockInferencePluginInternal2 engine;
};

TEST_F(PluginTest, canCreatePluginUsingSmartPtr) {
    ASSERT_NO_THROW(InferenceEngine::details::SOPointer<InferenceEngine::IInferencePlugin> ptr(get_mock_engine_name()));
}

TEST_F(PluginTest, shouldThrowExceptionIfPluginNotExist) {
    EXPECT_THROW(InferenceEngine::details::SOPointer<InferenceEngine::IInferencePlugin>(std::string{"unknown_plugin"}), Exception);
}

InferenceEngine::details::SOPointer<InferenceEngine::IInferencePlugin> PluginTest::getPtr() {
    InferenceEngine::details::SOPointer<InferenceEngine::IInferencePlugin> smart_ptr(get_mock_engine_name());
    return smart_ptr;
}

TEST_F(PluginTest, canSetConfiguration) {
    InferenceEngine::details::SOPointer<InferenceEngine::IInferencePlugin> ptr = getPtr();
    // TODO: dynamic->reinterpret because of clang/gcc cannot
    // dynamically cast this MOCK object
    ASSERT_TRUE(dynamic_cast<MockPlugin*>(ptr.operator->())->config.empty());

    std::map<std::string, std::string> config = { { "key", "value" } };
    ASSERT_NO_THROW(ptr->SetConfig(config));
    config.clear();

    ASSERT_STREQ(dynamic_cast<MockPlugin*>(ptr.operator->())->config["key"].c_str(), "value");
}
