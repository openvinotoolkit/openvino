// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <file_utils.h>
#include "openvino/util/shared_object.hpp"
#include <cpp/ie_plugin.hpp>

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class SharedObjectLoaderTests: public ::testing::Test {
protected:
    std::string get_mock_engine_name() {
        return FileUtils::makePluginLibraryName<char>({},
            std::string("mock_engine") + IE_BUILD_POSTFIX);
    }

    void loadDll(const string &libraryName) {
        sharedObjectLoader = ov::util::load_shared_object(libraryName.c_str());
    }
    std::shared_ptr<void> sharedObjectLoader;

    using CreateF = void(std::shared_ptr<IInferencePlugin>&);

    std::function<CreateF> make_std_function(const std::string& functionName) {
        std::function<CreateF> ptr(reinterpret_cast<CreateF*>(
            ov::util::get_symbol(sharedObjectLoader, functionName.c_str())));
        return ptr;
    }
};

typedef void*(*PluginEngineCreateFunc)(void);
typedef void(*PluginEngineDestoryFunc)(void *);

TEST_F(SharedObjectLoaderTests, canLoadExistedPlugin) {
    loadDll(get_mock_engine_name());
    EXPECT_NE(nullptr, sharedObjectLoader.get());
}

TEST_F(SharedObjectLoaderTests, loaderThrowsIfNoPlugin) {
    EXPECT_THROW(loadDll("wrong_name"), std::runtime_error);
}

TEST_F(SharedObjectLoaderTests, canFindExistedMethod) {
    loadDll(get_mock_engine_name());

    auto factory = make_std_function("CreatePluginEngine");
    EXPECT_NE(nullptr, factory);
}

TEST_F(SharedObjectLoaderTests, throwIfMethodNofFoundInLibrary) {
    loadDll(get_mock_engine_name());
    EXPECT_THROW(make_std_function("wrong_function"), std::runtime_error);
}

TEST_F(SharedObjectLoaderTests, canCallExistedMethod) {
    loadDll(get_mock_engine_name());

    auto factory = make_std_function("CreatePluginEngine");
    std::shared_ptr<IInferencePlugin> ptr;
    EXPECT_NO_THROW(factory(ptr));
}
