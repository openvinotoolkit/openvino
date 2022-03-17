// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MULTIDEVICEPLUGIN_NONCOPYABLE_H
#define MULTIDEVICEPLUGIN_NONCOPYABLE_H

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
class NonCopyable {
public:
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable(NonCopyable&&) = delete;

    NonCopyable& operator=(const NonCopyable&) = delete;
    NonCopyable& operator=(NonCopyable&&) = delete;

protected:
    NonCopyable() = default;
    virtual ~NonCopyable() = default;
};
} // namespace MultiDevicePlugin

#endif //MULTIDEVICEPLUGIN_NONCOPYABLE_H
