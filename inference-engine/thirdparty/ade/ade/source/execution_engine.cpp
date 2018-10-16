// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_engine/execution_engine.hpp"
#include "execution_engine/backend.hpp"
#include "execution_engine/executable.hpp"

#include "graph.hpp"

#include "util/assert.hpp"
#include "util/range.hpp"
#include "util/checked_cast.hpp"

namespace ade
{

class ExecutableImpl final : public Executable
{
    util::any m_dummy_opaque;

public:
    void addExec(std::unique_ptr<BackendExecutable>&& exec);

    virtual void run() override;
    virtual void run(util::any &opaque) override;

    virtual void runAsync() override;
    virtual void runAsync(util::any &opaque) override;
    virtual void wait() override;

private:
    std::unique_ptr<BackendExecutable> mainExec;
    std::vector<std::unique_ptr<BackendExecutable>> execs;
};

void BackendExecutable::run(util::any &)
{
    // Default implementation calls run() (backward compatibility)
    run();
}

void BackendExecutable::runAsync(util::any &)
{
    // Default implementation calls runAsync() (backward compatibility)
    runAsync();
}

ExecutionEngine::ExecutionEngine()
{

}

ExecutionEngine::~ExecutionEngine()
{

}

void ExecutionEngine::addPrePassCallback(ExecutionEngine::PassCallback callback)
{
    ASSERT(nullptr != callback);
    m_prePassCallbacks.callbacks.emplace_back(std::move(callback));
}

void ExecutionEngine::addPostPassCallback(ExecutionEngine::PassCallback callback)
{
    ASSERT(nullptr != callback);
    m_postPassCallbacks.callbacks.emplace_back(std::move(callback));
}

void ExecutionEngine::addBackend(std::unique_ptr<ExecutionBackend>&& backend)
{
    ASSERT(nullptr != backend);
    ASSERT(m_backends.end() == std::find(m_backends.begin(), m_backends.end(), backend));
    m_backends.emplace_back(std::move(backend));
}

void ExecutionEngine::setupBackends()
{
    ExecutionEngineSetupContext context(*this);
    for (auto& b: m_backends)
    {
        b->setupExecutionEngine(context);
    }
}

namespace
{
struct GraphListenerSetter final
{
    Graph& graph;
    IGraphListener* listener = nullptr;
    GraphListenerSetter(Graph& gr, IGraphListener* l):
        graph(gr), listener(l)
    {
        ASSERT(nullptr == graph.getListener());
        graph.setListener(listener);
    }
    ~GraphListenerSetter()
    {
        ASSERT(listener == graph.getListener());
        graph.setListener(nullptr);
    }

    GraphListenerSetter(const GraphListenerSetter&) = delete;
    GraphListenerSetter& operator=(const GraphListenerSetter&) = delete;
};
}

void ExecutionEngine::runPasses(Graph& graph)
{
    m_lazyPasses.reset();
    GraphListenerSetter setter(graph, m_lazyPasses.getListener());
    passes::PassContext context{graph};
    m_passManager.run(context);
    for (auto& str: m_executableDependencies)
    {
        ASSERT(!str.empty());
        auto pass = m_lazyPasses.getPass(str);
        ASSERT(nullptr != pass);
        pass->process(context);
    }
}

std::unique_ptr<Executable> ExecutionEngine::createExecutable(const Graph& graph)
{
    std::unique_ptr<ExecutableImpl> ret;
    for (auto& b : m_backends)
    {
        std::unique_ptr<BackendExecutable> bexec(b->createExecutable(graph));
        if (nullptr != bexec)
        {
            if (nullptr == ret)
            {
                ret.reset(new ExecutableImpl);
            }
            ret->addExec(std::move(bexec));
        }
    }

    return std::move(ret);
}

void ExecutionEngine::addExecutableDependency(const std::string& lazyPassName)
{
    ASSERT(!lazyPassName.empty());
    ASSERT(m_lazyPasses.getPass(lazyPassName) != nullptr);
    m_executableDependencies.emplace(lazyPassName);
}

void ExecutionEngine::addPassStage(const std::string& stageName)
{
    ASSERT(!stageName.empty());
    m_passManager.addStage(stageName);
}

void ExecutionEngine::addPassStage(const std::string& stageName, const std::string& prevStage)
{
    ASSERT(!stageName.empty());
    ASSERT(!prevStage.empty());
    m_passManager.addStage(stageName, prevStage);
}

ExecutionEngine::StagesRange ExecutionEngine::passStages() const
{
    return util::map<PassMapper>(m_passManager.stages());
}

void ExecutionEngine::prePass(const PassDesc& desc,
                              const passes::PassContext& context)
{
    m_prePassCallbacks.call(desc, context);
}

void ExecutionEngine::postPass(const PassDesc& desc,
                               const passes::PassContext& context)
{
    m_postPassCallbacks.call(desc, context);
}

void ExecutableImpl::addExec(std::unique_ptr<BackendExecutable>&& exec)
{
    ASSERT(nullptr != exec);
    if (nullptr == mainExec)
    {
        mainExec = std::move(exec);
    }
    else
    {
        execs.emplace_back(std::move(exec));
    }
}

struct ExecExceptionHandler
{
    size_t passedCount = 0;
    std::vector<std::unique_ptr<BackendExecutable>> &handledVector;
    ExecExceptionHandler(std::vector<std::unique_ptr<BackendExecutable>> &execs) : handledVector(execs) {}
    ~ExecExceptionHandler()
    {
        ASSERT(handledVector.size() >= passedCount);
        auto count = util::checked_cast<int>(handledVector.size() - passedCount);
        for (auto i = util::checked_cast<int>(handledVector.size()) - 1;
             i >= count;
             i--)
        {
            handledVector[i]->cancel();
        }
    }
};

void ExecutableImpl::run()
{
    // Since run() takes a modifiable `any`, reset it before the run
    m_dummy_opaque = util::any();
    run(m_dummy_opaque);
}

void ExecutableImpl::run(util::any &opaque)
{
    ASSERT(nullptr != mainExec);
    ExecExceptionHandler handler(execs);
    for (auto& e: util::toRangeReverse(execs))
    {
        e->runAsync(opaque);
        handler.passedCount++;
    }

    mainExec->run(opaque);

    for (auto& e: util::toRange(execs))
    {
        handler.passedCount--;
        e->wait();
    }
}

void ExecutableImpl::runAsync()
{
    // Since runAsync() takes a modifiable `any`, reset it before the run
    m_dummy_opaque = util::any();
    runAsync(m_dummy_opaque);
}

void ExecutableImpl::runAsync(util::any &opaque)
{
    ASSERT(nullptr != mainExec);
    for (auto& e: util::toRangeReverse(execs))
    {
        e->runAsync(opaque);
    }
    mainExec->runAsync(opaque);
}

void ExecutableImpl::wait()
{
    ASSERT(nullptr != mainExec);
    ExecExceptionHandler handler(execs);
    handler.passedCount = util::checked_cast<decltype(handler.passedCount)>(execs.size());
    mainExec->wait();
    for (auto& e: util::toRange(execs))
    {
        handler.passedCount--;
        e->wait();
    }
}

ExecutionEngineSetupContext::ExecutionEngineSetupContext(ExecutionEngine& e):
    m_engine(e)
{

}

void ExecutionEngineSetupContext::addPrePassCallback(PassCallback callback)
{
    m_engine.addPrePassCallback(std::move(callback));
}

void ExecutionEngineSetupContext::addPostPassCallback(PassCallback callback)
{
    m_engine.addPostPassCallback(std::move(callback));
}

void ExecutionEngineSetupContext::addExecutableDependency(const std::string& lazyPassName)
{
    m_engine.addExecutableDependency(lazyPassName);
}

void ExecutionEngineSetupContext::addPassStage(const std::string& stageName)
{
    m_engine.addPassStage(stageName);
}

void ExecutionEngineSetupContext::addPassStage(const std::string& stageName, const std::string& prevStage)
{
    m_engine.addPassStage(stageName, prevStage);
}

ExecutionEngineSetupContext::StagesRange ExecutionEngineSetupContext::passStages() const
{
    return m_engine.passStages();
}

void ExecutionEngine::CallbackList::call(const ExecutionEngine::PassDesc& desc, const passes::PassContext& context) const
{
    for (auto& callback: callbacks)
    {
        ASSERT(nullptr != callback);
        callback(desc, context);
    }
}

bool ExecutionEngine::CallbackList::empty() const
{
    return callbacks.empty();
}

IGraphListener* ExecutionEngine::LazyPasses::getListener() const
{
    ASSERT((nullptr == last) == passes.empty());
    return last;
}

ExecutionEngine::LazyPassWrapper* ExecutionEngine::LazyPasses::getPass(const std::string& name) const
{
    ASSERT(!name.empty());
    ASSERT(util::contains(passes, name));
    auto it = passes.find(name);
    auto ret = it->second.get();
    ASSERT(nullptr != ret);
    return ret;
}

void ExecutionEngine::LazyPasses::reset()
{
    if (nullptr != last)
    {
        last->reset();
    }
}

ExecutionEngine::LazyPassWrapper::~LazyPassWrapper()
{

}

bool ExecutionEngine::LazyPassWrapper::isValid() const
{
    return m_valid;
}

bool ExecutionEngine::LazyPassWrapper::isFirst() const
{
    return nullptr == m_prev;
}

ExecutionEngine::LazyPassWrapper::LazyPassWrapper(ExecutionEngine::LazyPassWrapper* prev):
    m_prev(prev)
{

}

void ExecutionEngine::LazyPassWrapper::reset()
{
    m_valid = false;
    if (!isFirst())
    {
        m_prev->reset();
    }
}

}
