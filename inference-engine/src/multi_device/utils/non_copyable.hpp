// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MULTIDEVICEPLUGIN_NONCOPYABLE_H
#define MULTIDEVICEPLUGIN_NONCOPYABLE_H

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

#endif //MULTIDEVICEPLUGIN_NONCOPYABLE_H
