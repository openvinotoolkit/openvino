// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef EXECUTABLE_HPP
#define EXECUTABLE_HPP

namespace util
{
    class any;
}

namespace ade
{
class Executable
{
public:
    virtual ~Executable() = default;
    virtual void run() = 0;
    virtual void run(util::any &opaque) = 0;      // WARNING: opaque may be accessed from various threads.

    virtual void runAsync() = 0;
    virtual void runAsync(util::any &opaque) = 0; // WARNING: opaque may be accessed from various threads.

    virtual void wait() = 0;
};
}

#endif // EXECUTABLE_HPP
