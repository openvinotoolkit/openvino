// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <gmock/gmock-spec-builders.h>

#include <memory>
#include <details/ie_so_pointer.hpp>

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

template<class T=PointedObjHelper, class L = SharedObjectLoaderHelper>
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
