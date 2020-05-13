// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_plugin_ptr.hpp>
#include <file_utils.h>
#include "details/ie_so_loader.h"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

IE_SUPPRESS_DEPRECATED_START

class SharedObjectLoaderTests: public ::testing::Test {
protected:
    std::string get_mock_engine_name() {
        return FileUtils::makeSharedLibraryName<char>(getIELibraryPath(),
            std::string("mock_engine") + IE_BUILD_POSTFIX);
    }

    void loadDll(const string &libraryName) {
        sharedObjectLoader.reset(new details::SharedObjectLoader(libraryName.c_str()));
    }
    unique_ptr<SharedObjectLoader> sharedObjectLoader;

    template <class T>
    std::function<T> make_std_function(const std::string& functionName) {
        std::function<T> ptr(reinterpret_cast<T*>(sharedObjectLoader->get_symbol(functionName.c_str())));
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
    EXPECT_THROW(loadDll("wrong_name"), InferenceEngine::details::InferenceEngineException);
}

TEST_F(SharedObjectLoaderTests, canFindExistedMethod) {
    loadDll(get_mock_engine_name());

    auto factory = make_std_function<StatusCode(IInferencePlugin*&, ResponseDesc*)>("CreatePluginEngine");
    EXPECT_NE(nullptr, factory);
}

TEST_F(SharedObjectLoaderTests, throwIfMethodNofFoundInLibrary) {
    loadDll(get_mock_engine_name());

    EXPECT_THROW(make_std_function<IInferencePlugin*()>("wrong_function"), InferenceEngine::details::InferenceEngineException);
}

TEST_F(SharedObjectLoaderTests, canCallExistedMethod) {
    loadDll(get_mock_engine_name());

    auto factory = make_std_function<StatusCode(IInferencePlugin*&, ResponseDesc*)>("CreatePluginEngine");
    IInferencePlugin* ptr = nullptr;
    ResponseDesc resp;
    EXPECT_NO_THROW(factory(ptr, &resp));
}
