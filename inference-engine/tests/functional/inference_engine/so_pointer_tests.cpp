// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <gmock/gmock-spec-builders.h>

#include <file_utils.h>

#include <memory>
#include <common_test_utils/test_assertions.hpp>
#include <details/ie_so_pointer.hpp>
#include <details/ie_irelease.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <ie_plugin_ptr.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace ::testing;
using ::testing::InSequence;

class PointedObjHelper {
public:
    MOCK_METHOD0(obj_dtor, void());

    virtual ~PointedObjHelper() {
        obj_dtor();
    }
};

class SharedObjectLoaderHelper {
public:
    MOCK_METHOD0(loader_dtor, void());

    virtual ~SharedObjectLoaderHelper() {
        loader_dtor();
    }
};

template <class T = PointedObjHelper, class L = SharedObjectLoaderHelper>
class SoPointerHelper : public SOPointer<T, L> {
public:
    SoPointerHelper(std::shared_ptr<L>&& loader, std::shared_ptr<T>&& object)
            : SOPointer<T, L>() {
        SOPointer<T, L>::_so_loader = std::move(loader);
        SOPointer<T, L>::_pointedObj = std::move(object);
    }
};

class SoPointerTests : public ::testing::Test {
protected:
    SharedObjectLoaderHelper *loader1 = nullptr, *loader2 = nullptr;
    PointedObjHelper *obj1 = nullptr, *obj2 = nullptr;
};

TEST_F(SoPointerTests, assignObjThenLoader) {
    auto loaderPtr1 = std::make_shared<SharedObjectLoaderHelper>();
    auto objPtr1 = std::make_shared<PointedObjHelper>();
    auto loaderPtr2 = std::make_shared<SharedObjectLoaderHelper>();
    auto objPtr2 = std::make_shared<PointedObjHelper>();
    loader1 = loaderPtr1.get();
    obj1 = objPtr1.get();
    loader2 = loaderPtr2.get();
    obj2 = objPtr2.get();
    SoPointerHelper<> soPointer1(std::move(loaderPtr1), std::move(objPtr1));
    SoPointerHelper<> soPointer2(std::move(loaderPtr2), std::move(objPtr2));

    {
        InSequence dummy;
        EXPECT_CALL(*obj1, obj_dtor());
        EXPECT_CALL(*loader1, loader_dtor());
        EXPECT_CALL(*obj2, obj_dtor());
        EXPECT_CALL(*loader2, loader_dtor());
    }

    soPointer1 = soPointer2;
}

namespace InferenceEngine {

namespace details {

template<>
class SOCreatorTrait<InferenceEngine::details::IRelease> {
public:
    static constexpr auto name = "CreateIRelease";
};

}  // namespace details

}  // namespace InferenceEngine

TEST_F(SoPointerTests, UnknownPlugin) {
    ASSERT_THROW(SOPointer<InferenceEngine::details::IRelease>("UnknownPlugin"), InferenceEngineException);
}

TEST_F(SoPointerTests, UnknownPluginExceptionStr) {
    try {
        SOPointer<InferenceEngine::details::IRelease>("UnknownPlugin");
    }
    catch (InferenceEngineException &e) {
        ASSERT_STR_CONTAINS(e.what(), "Cannot load library 'UnknownPlugin':");
        ASSERT_STR_DOES_NOT_CONTAIN(e.what(), "path:");
        ASSERT_STR_DOES_NOT_CONTAIN(e.what(), "from CWD:");
    }
}

using SymbolLoaderTests = ::testing::Test;

TEST_F(SymbolLoaderTests, throwCreateNullPtr) {
    ASSERT_THROW(SymbolLoader<SharedObjectLoader>(nullptr), InferenceEngineException);
}

TEST_F(SymbolLoaderTests, instantiateSymbol) {
    std::string name = FileUtils::makePluginLibraryName<char>(getIELibraryPath(),
        std::string("mock_engine") + IE_BUILD_POSTFIX);
    std::shared_ptr<SharedObjectLoader> sharedLoader(new SharedObjectLoader(name.c_str()));
    SymbolLoader<SharedObjectLoader> loader(sharedLoader);
    IInferencePlugin * value = nullptr;
    ASSERT_NE(nullptr, value = loader.instantiateSymbol<IInferencePlugin>(
        SOCreatorTrait<IInferencePlugin>::name));
    value->Release();
}
