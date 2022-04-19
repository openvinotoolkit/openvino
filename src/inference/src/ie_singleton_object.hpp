// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <list>
#include <memory>
#include <mutex>

namespace InferenceEngine {

#define MAX_SUPPORTED_SINGLETON_PRIORITY 4

class NonCopyableObject {
public:
    NonCopyableObject(const NonCopyableObject&) = delete;
    NonCopyableObject(NonCopyableObject&&) = delete;

    NonCopyableObject& operator=(const NonCopyableObject&) = delete;
    NonCopyableObject& operator=(NonCopyableObject&&) = delete;

protected:
    NonCopyableObject() = default;
    virtual ~NonCopyableObject() = default;
};

class SingletonMem {
public:
    SingletonMem() = default;
    virtual ~SingletonMem() = default;
    virtual void setPriority(int priority = 3) {
        std::lock_guard<std::mutex> lockGuard(m_mutex);
        if (m_hasSetPriority) {
            return;
        }

        m_pointers[priority].push_front(std::shared_ptr<SingletonMem>(this, [](SingletonMem* p) {
            delete p;
        }));
        m_hasSetPriority = true;
    }

    static void releaseSingltons();

private:
    static std::mutex m_mutex;
    static std::list<std::shared_ptr<SingletonMem>> m_pointers[MAX_SUPPORTED_SINGLETON_PRIORITY];
    bool m_hasSetPriority{false};
};

template <typename Type>
class SingletonObject : public NonCopyableObject, public SingletonMem {
public:
    static Type* instance() {
        static Type* obj = nullptr;
        std::call_once(m_onceFlag, [&]() {
            obj = new Type();
            assert(obj != nullptr);
            obj->setPriority();
        });
        return obj;
    }

protected:
    static std::once_flag m_onceFlag;
    SingletonObject() = default;
    virtual ~SingletonObject() = default;
};

template <typename Type>
std::once_flag SingletonObject<Type>::m_onceFlag;
}  // namespace InferenceEngine