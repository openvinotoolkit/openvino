// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef BACKEND_HPP
#define BACKEND_HPP

#include <memory>

namespace util
{
    class any;
}

namespace ade
{

class Graph;
class Executable;
class ExecutionEngineSetupContext;

class BackendExecutable;

class ExecutionBackend
{
public:
    virtual ~ExecutionBackend() = default;

    virtual void setupExecutionEngine(ExecutionEngineSetupContext& engine) = 0;
    virtual std::unique_ptr<BackendExecutable> createExecutable(const Graph& graph) = 0;
};

class BackendExecutable
{
protected:
    // Backward-compatibility stubs
    virtual void run()      {};                    // called by run(any)
    virtual void runAsync() {};                    // called by runAsync(any)

public:
    virtual ~BackendExecutable() = default;

    virtual void run(util::any &opaque);           // Triggered by ADE engine
    virtual void runAsync(util::any &opaque);      // Triggered by ADE engine

    virtual void wait() = 0;
    virtual void cancel() = 0;
};
}

#endif // BACKEND_HPP
