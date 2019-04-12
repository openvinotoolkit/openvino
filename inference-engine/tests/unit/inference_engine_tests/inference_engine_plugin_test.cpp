// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_common.hpp"

#include <ie_plugin_ptr.hpp>
#include "details/ie_so_loader.h"
#include "inference_engine.hpp"
#include "mock_inference_engine.hpp"
#include "mock_error_listener.hpp"
#include "../tests/mock_engine/mock_plugin.hpp"


using namespace std;
using namespace InferenceEngine;
using namespace ::testing;
using namespace InferenceEngine::details;

class PluginTest: public TestsCommon {
protected:
    unique_ptr<SharedObjectLoader> sharedObjectLoader;
    std::function<IInferencePlugin*(IInferencePlugin*)> createPluginEngineProxy;
    std::function<void(IInferencePlugin*)> injectPluginEngineProxy ;
    InferenceEnginePluginPtr getPtr() ;
    virtual void SetUp() {

        std::string libraryName = get_mock_engine_name();
        sharedObjectLoader.reset(new SharedObjectLoader(libraryName.c_str()));
        createPluginEngineProxy = make_std_function<IInferencePlugin*(IInferencePlugin*)>("CreatePluginEngineProxy");
        injectPluginEngineProxy = make_std_function<void(IInferencePlugin*)>("InjectProxyEngine");

    }
    template <class T>
    std::function<T> make_std_function(const std::string& functionName) {
        std::function <T> ptr (reinterpret_cast<T*>(sharedObjectLoader->get_symbol(functionName.c_str())));
        return ptr;
    }

    MockInferenceEngine engine;
    Listener error;
};

TEST_F(PluginTest, canCreatePlugin) {
    auto ptr = make_std_function<IInferencePlugin*(IInferencePlugin*)>("CreatePluginEngineProxy");

    unique_ptr<IInferencePlugin, std::function<void (IInferencePlugin*)>> smart_ptr(ptr(nullptr), [](IInferencePlugin *p) {
        p->Release();
    });

    //expect that no error handler has been called
    smart_ptr->SetLogCallback(error);
    EXPECT_CALL(error, onError(_)).Times(0);
}

TEST_F(PluginTest, canCreatePluginUsingSmartPtr) {
    ASSERT_NO_THROW(InferenceEnginePluginPtr ptr(get_mock_engine_name()));
}

TEST_F(PluginTest, shouldThrowExceptionIfPluginNotExist) {
    EXPECT_THROW(InferenceEnginePluginPtr("unknown_plugin"), InferenceEngineException);
}

ACTION_TEMPLATE(CallListenerWithErrorMessage,
                HAS_1_TEMPLATE_PARAMS(int, k),
                AND_1_VALUE_PARAMS(pointer))
{
    InferenceEngine::IErrorListener & data = ::std::get<k>(args);
    data.onError(pointer);
}

TEST_F(PluginTest, canCallErrorHandlerIfNecessary) {

    unique_ptr<IInferencePlugin, std::function<void(IInferencePlugin*)>> smart_ptr(createPluginEngineProxy(&engine), [](IInferencePlugin *p) {
        p->Release();
    });

    const char * err = "my error forward";

    EXPECT_CALL(error, onError(err)).Times(1);
    EXPECT_CALL(engine, SetLogCallback(_)).WillOnce(CallListenerWithErrorMessage<0>(err));
    EXPECT_CALL(engine, Release()).Times(1);

    smart_ptr->SetLogCallback(error);
}

InferenceEnginePluginPtr PluginTest::getPtr() {
    InferenceEnginePluginPtr smart_ptr(get_mock_engine_name());
    return smart_ptr;
};

TEST_F(PluginTest, canForwardPluginEnginePtr) {

    injectPluginEngineProxy(&engine);
    InferenceEnginePluginPtr ptr3 = getPtr();

    EXPECT_CALL(engine, Infer(_, A<Blob&>(), _)).WillOnce(Return(OK));
    EXPECT_CALL(engine, Release()).Times(1);

    TBlob <float> b1(Precision::FP32, NCHW);
    TBlob <float> b2(Precision::FP32, NCHW);
    ptr3->Infer(b1, b2, nullptr);
}


TEST_F(PluginTest, canSetConfiguration) {
    InferenceEnginePluginPtr ptr = getPtr();
    // TODO: dynamic->reinterpret because of calng/gcc cannot
    // dynamically cast this MOCK object
    ASSERT_TRUE(reinterpret_cast<MockPlugin*>(*ptr)->config.empty());

    ResponseDesc resp;
    std::map<std::string, std::string> config = { { "key", "value" } };
    ASSERT_EQ(ptr->SetConfig(config, &resp), OK);
    config.clear();

    ASSERT_STREQ(reinterpret_cast<MockPlugin*>(*ptr)->config["key"].c_str(), "value");
}
