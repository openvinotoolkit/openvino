// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MULTIDEVICEPLUGIN_SINGLETON_H
#define MULTIDEVICEPLUGIN_SINGLETON_H
#include <cassert>
#include <list>
#include <memory>
#include <mutex>

#include "non_copyable.hpp"

template <typename Type>
class Singleton : public NonCopyable {
public:
    static std::shared_ptr<Type>& instance() {
        static std::shared_ptr<Type> obj;
        std::call_once(m_onceFlag, [&]() {
            auto* objPtr = new Type();
            assert(objPtr!= nullptr);
            obj.reset(objPtr);
        });
        return obj;
    }

protected:
    static std::once_flag m_onceFlag;
    Singleton() = default;
    virtual ~Singleton() = default;
};

template <typename Type>
std::once_flag Singleton<Type>::m_onceFlag;

#endif //MULTIDEVICEPLUGIN_SINGLETON_H
