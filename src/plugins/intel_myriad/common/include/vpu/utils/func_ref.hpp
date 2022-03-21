// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <type_traits>

namespace vpu {

//
// Non-owning alternative for std::function
//

template <typename> class FuncRef;

template <typename R, typename... Args>
class FuncRef<R(Args...)> {
public:
    template <class Func>
    FuncRef(const Func& func) :
            _realFuncPtr(&func),
            _impl(&caller<typename std::remove_reference<Func>::type>) {
        using actual_result_type = typename std::result_of<Func(Args...)>::type;
        static_assert(
            !std::is_reference<R>::value || std::is_reference<actual_result_type>::value,
            "Mismatch between Func and FuncRef prototype");
    }

    FuncRef(const FuncRef&) = delete;
    FuncRef& operator=(const FuncRef&) = delete;

    FuncRef(FuncRef&&) = delete;
    FuncRef& operator=(FuncRef&&) = delete;

    R operator()(Args... args) const {
        return _impl(_realFuncPtr, std::forward<Args>(args)...);
    }

private:
    template <class Func>
    static R caller(const void* realFuncPtr, Args... args) {
        const auto& realFunc = *static_cast<const Func*>(realFuncPtr);
        return realFunc(std::forward<Args>(args)...);
    }

private:
    using ImplFunc = R(*)(const void*, Args...);

    const void* _realFuncPtr = nullptr;
    ImplFunc _impl = nullptr;
};

}  // namespace vpu
