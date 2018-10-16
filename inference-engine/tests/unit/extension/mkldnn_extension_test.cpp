// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_common.hpp"

#if !defined(ENABLE_MKL_DNN)
# include "disable_tests.hpp"
#endif

#include <ie_plugin_ptr.hpp>
#include <mkldnn/mkldnn_extension.hpp>
#include <mkldnn/mkldnn_extension_ptr.hpp>
#include "details/ie_so_loader.h"
#include "inference_engine.hpp"
#include "mock_inference_engine.hpp"
#include "mock_error_listener.hpp"
#include "mock_mkldnn_extension.hpp"

using namespace std;
using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

class ExtensionTest: public TestsCommon {
 protected:
    unique_ptr<SharedObjectLoader> sharedObjectLoader;
    //std::function<IInferencePlugin*(IInferencePlugin*)> createPluginEngineProxy;
    std::function<void(MKLDNExtension*)> injectExtension ;
    InferenceEnginePluginPtr getPtr() ;
    virtual void SetUp() {

        std::string libraryName = get_mock_extension_name();
        sharedObjectLoader.reset(new SharedObjectLoader(libraryName.c_str()));

        //createPluginEngineProxy = make_std_function<IInferencePlugin*(IInferencePlugin*)>("CreatePluginEngineProxy");

        injectExtension = make_std_function<void(MKLDNExtension*)>("InjectProxyMKLDNNExtension");
    }
    template <class T>
    std::function<T> make_std_function(const std::string& functionName) {
        std::function <T> ptr (reinterpret_cast<T*>(sharedObjectLoader->get_symbol(functionName.c_str())));
        return ptr;
    }

    MockMKLDNNExtension mock_extension;
    Listener error;
};

#ifndef _WIN32
TEST_F(ExtensionTest, canForwardCreateExtension) {

    //expect that setlogcallback is forwarder
    GenericPrimitive  *gprim;
    InferenceEngine::CNNLayerPtr layer;
    EXPECT_CALL(mock_extension, CreateGenericPrimitive(gprim, _, nullptr)).Times(1);

    injectExtension(&mock_extension);

    InferenceEngine::MKLDNNPlugin::MKLDNNExtension extension(get_mock_extension_name());

    extension.CreateGenericPrimitive(gprim, layer, nullptr);
}

TEST_F(ExtensionTest, canLeaveAsSharedPointer) {

    //expect that setlogcallback is forwarder
    GenericPrimitive  *gprim;
    InferenceEngine::CNNLayerPtr layer;
    EXPECT_CALL(mock_extension, CreateGenericPrimitive(gprim, _, nullptr)).Times(1);

    injectExtension(&mock_extension);

    std::shared_ptr<InferenceEngine::MKLDNNPlugin::IMKLDNNExtension> pointer;
    std::shared_ptr<InferenceEngine::IExtension> base_pointer;

    {
        InferenceEngine::MKLDNNPlugin::MKLDNNExtension extension(get_mock_extension_name());
        base_pointer = make_so_pointer<InferenceEngine::MKLDNNPlugin::IMKLDNNExtension>(get_mock_extension_name());
        //here we are deleting original sharedObject loader, means reference to it should remain in
        //extension pointer
        sharedObjectLoader.reset(nullptr);
    }

    pointer = dynamic_pointer_cast<InferenceEngine::MKLDNNPlugin::IMKLDNNExtension>(base_pointer);
    pointer->CreateGenericPrimitive(gprim, layer, nullptr);
}
#endif