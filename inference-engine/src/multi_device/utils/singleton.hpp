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

#define MAX_UTILS_SUPPORTED_PRIORITY 4

class SingleMem {
public:
    SingleMem() = default;
    virtual ~SingleMem() = default;
    virtual void setPriority(int priority = 3) {
        std::lock_guard<std::mutex> lockGuard(m_mutex);
        if (m_hasSetPriority) {
            return;
        }

        m_pointers[priority].push_front(std::shared_ptr<SingleMem>(this, [](SingleMem* p) { delete p; }));
        m_hasSetPriority = true;
    }

    static void releaseSingltons();

private:
    static std::mutex m_mutex;
    static std::list<std::shared_ptr<SingleMem>> m_pointers[MAX_UTILS_SUPPORTED_PRIORITY];
    bool m_hasSetPriority { false };
};

template <typename Type>
class Singleton : public NonCopyable, public SingleMem {
public:
    static Type* instance() {
        static Type* obj = nullptr;
        std::call_once(m_onceFlag, [&]() {
            obj = new Type();
            assert(obj!= nullptr);
            obj->setPriority(); });
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
