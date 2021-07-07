// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <file_utils.h>
#include "details/ie_so_loader.h"
#include <cpp/ie_plugin.hpp>

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class SharedObjectLoaderTests: public ::testing::Test {
protected:
    std::string get_mock_engine_name() {
        return FileUtils::makePluginLibraryName<char>(getIELibraryPath(),
            std::string("mock_engine") + IE_BUILD_POSTFIX);
    }

    void loadDll(const string &libraryName) {
        sharedObjectLoader.reset(new details::SharedObjectLoader(libraryName.c_str()));
    }
    unique_ptr<SharedObjectLoader> sharedObjectLoader;

    using CreateF = void(std::shared_ptr<IInferencePlugin>&);

    std::function<CreateF> make_std_function(const std::string& functionName) {
        std::function<CreateF> ptr(reinterpret_cast<CreateF*>(sharedObjectLoader->get_symbol(functionName.c_str())));
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
    EXPECT_THROW(loadDll("wrong_name"), InferenceEngine::Exception);
}

TEST_F(SharedObjectLoaderTests, canFindExistedMethod) {
    loadDll(get_mock_engine_name());

    auto factory = make_std_function("CreatePluginEngine");
    EXPECT_NE(nullptr, factory);
}

TEST_F(SharedObjectLoaderTests, throwIfMethodNofFoundInLibrary) {
    loadDll(get_mock_engine_name());
    EXPECT_THROW(make_std_function("wrong_function"), InferenceEngine::Exception);
}

TEST_F(SharedObjectLoaderTests, canCallExistedMethod) {
    loadDll(get_mock_engine_name());

    auto factory = make_std_function("CreatePluginEngine");
    std::shared_ptr<IInferencePlugin> ptr;
    EXPECT_NO_THROW(factory(ptr));
}
