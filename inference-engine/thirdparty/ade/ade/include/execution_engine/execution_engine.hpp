// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef EXECUTION_ENGINE_HPP
#define EXECUTION_ENGINE_HPP

#include <memory>
#include <vector>
#include <utility>
#include <functional>
#include <initializer_list>
#include <unordered_set>

#include "util/assert.hpp"
#include "util/map_range.hpp"
#include "util/algorithm.hpp"
#include "util/type_traits.hpp"

#include "graph_listener.hpp"

#include "passmanager.hpp"
#include "passes/pass_base.hpp"

namespace ade
{

class Graph;
class Executable;
class ExecutionBackend;

class ExecutionEngine final
{
public:
    struct PassMapper final
    {
        const std::string& operator()(const std::pair<std::string, PassList<passes::PassContext>>& p) const
        {
            return p.first;
        }
    };

    using PassMan = PassManager<passes::PassContext>;
    struct PassDesc final
    {
        std::string stage;
        std::string pass;
    };

    using PassCallback = std::function<void(const PassDesc&, const passes::PassContext&)>;

    using StagesRange = util::MapRange<PassMan::StagesCRange, PassMapper>;


    ExecutionEngine();
    ExecutionEngine(const ExecutionEngine&) = delete;
    ExecutionEngine& operator=(const ExecutionEngine&) = delete;
    ~ExecutionEngine();

    /// Add callback called before each pass execution
    void addPrePassCallback(PassCallback callback);

    /// Add callback called after each pass execution
    void addPostPassCallback(PassCallback callback);

    /// Add backend to engine
    /// Engine takes ownership of backend object
    void addBackend(std::unique_ptr<ExecutionBackend>&& backend);

    /// Call backends initialization function
    void setupBackends();

    /// Run all registered passes on graph
    void runPasses(Graph& graph);

    /// Create executable from graph
    /// Caller must take ownership of returned executable
    /// Can return null if nothing to execute
    std::unique_ptr<Executable> createExecutable(const Graph& graph);

    /// Add lazy pass to engine
    /// Other passes can add lazy passes  as dependencies
    /// Lazy pass listen to graph events via checker object (for list of events see IGraphListener)
    /// Checker can return false from event handler to invalidate corresponding pass
    /// Before execution of pass for any invalid lazy passes will be called operator() to make them valid again
    ///
    /// @param passName     Name of the pass, must be unique ant not to be empty
    /// @param pass         Pass object
    /// @param checker      Checker object
    template<typename P, typename C>
    void addLazyPass(const std::string& passName, P&& pass, C&& checker)
    {
        m_lazyPasses.addPass(passName, std::forward<P>(pass), std::forward<C>(checker));
    }

    /// Adds lazy pass as final executable dependency
    /// It will be executed even if there are no passes depending on it
    /// @param lazyPassName Name of the pass, must be one the passes added via addLazyPass
    void addExecutableDependency(const std::string& lazyPassName);

    /// Add pass stage
    /// @param stageName    Name of stage to be added, must not be empty
    void addPassStage(const std::string& stageName);

    /// Add pass stage before specified stage
    /// @param stageName    Name of stage to be added, must not be empty
    /// @param prevStage    Name of the previous stage, must not be empty
    void addPassStage(const std::string& stageName, const std::string& prevStage);

    /// Returns range with stages names
    StagesRange passStages() const;

    /// Add pass to specified stage
    /// @param stageName    Name of the stage pass to be added, must not be empty
    /// @param passName     Name of the pass, must not be empty
    /// @param pass         Pass object
    template<typename T>
    void addPass(const std::string& stageName, const std::string& passName, T&& pass)
    {
        ASSERT(!stageName.empty());
        ASSERT(!passName.empty());
        addPass(stageName, passName, std::forward<T>(pass), std::initializer_list<const char*>{});
    }

    /// Add pass to specified stage
    /// @param stageName       Name of the stage pass to be added, must not be empty
    /// @param passName        Name of the pass, must not be empty
    /// @param pass            Pass object
    /// @param lazyPassesNames List of lasy passes names from which this pass is depends
    template<typename T, typename Range>
    void addPass(const std::string& stageName, const std::string& passName, T&& pass, Range&& lazyPassesNames)
    {
        ASSERT(!stageName.empty());
        ASSERT(!passName.empty());
        m_passManager.addPass(stageName, PassWrapper<T>{{stageName, passName},
                                                        *this,
                                                        std::move(getLazyPasses(std::forward<Range>(lazyPassesNames))),
                                                        std::forward<T>(pass)});
    }

    /// Add pass to specified stage
    /// @param stageName       Name of the stage pass to be added, must not be empty
    /// @param passName        Name of the pass, must not be empty
    /// @param pass            Pass object
    /// @param lazyPassesNames List of lasy passes names from which this pass is depends
    template<typename T, typename NameT>
    void addPass(const std::string& stageName, const std::string& passName, T&& pass, std::initializer_list<NameT> lazyPassesNames)
    {
        ASSERT(!stageName.empty());
        ASSERT(!passName.empty());
        m_passManager.addPass(stageName, PassWrapper<T>{{stageName, passName},
                                                        *this,
                                                        std::move(getLazyPasses(lazyPassesNames)),
                                                        std::forward<T>(pass)});
    }

private:
    void prePass(const PassDesc& desc,
                 const passes::PassContext& context);
    void postPass(const PassDesc& desc,
                  const passes::PassContext& context);

    class LazyPassWrapper;

    template<typename Func>
    struct PassWrapper final
    {
        PassDesc desc;
        ExecutionEngine& engine;
        std::vector<LazyPassWrapper*> lazy_passes;
        Func func;
        void operator()(passes::PassContext& context) const
        {
            for (auto pass: lazy_passes)
            {
                ASSERT(nullptr != pass);
                pass->process(context);
            }
            engine.prePass(desc, context);
            func(context);
            engine.postPass(desc, context);
        }
    };

    struct CallbackList final
    {
        std::vector<PassCallback> callbacks;

        void call(const PassDesc& desc,
                  const passes::PassContext& context) const;

        bool empty() const;
    };

    class LazyPassWrapper : public IGraphListener
    {
    protected:
        LazyPassWrapper* m_prev = nullptr;
        bool m_valid = false;

        bool isValid() const;
        bool isFirst() const;
    public:
        LazyPassWrapper(LazyPassWrapper* prev);
        virtual ~LazyPassWrapper();

        void reset();

        virtual void process(passes::PassContext& context) = 0;
    };

    template<typename BasePass, typename Checker>
    class LazyPassImpl final : public LazyPassWrapper
    {
        BasePass m_pass;
        Checker m_checker;
    public:
        template<typename P, typename C>
        LazyPassImpl(LazyPassWrapper* prev, P&& pass, C&& checker):
            LazyPassWrapper(prev),
            m_pass(std::forward<P>(pass)),
            m_checker(std::forward<C>(checker))
        {}

        LazyPassImpl(const LazyPassImpl&) = delete;
        LazyPassImpl& operator=(const LazyPassImpl&) = delete;
        LazyPassImpl(LazyPassImpl&&) = default;
        LazyPassImpl& operator=(LazyPassImpl&&) = default;

        virtual void process(passes::PassContext& context) override
        {
            if (!isValid())
            {
                m_pass(context);
                m_valid = true;
            }
        }

        virtual void nodeCreated(const Graph& graph, const NodeHandle& node) override
        {
            if (isValid())
            {
                m_valid = m_checker.nodeCreated(graph, node);
            }

            if (!isFirst())
            {
                m_prev->nodeCreated(graph, node);
            }
        }

        virtual void nodeAboutToBeDestroyed(const Graph& graph, const NodeHandle& node) override
        {
            if (isValid())
            {
                m_valid = m_checker.nodeAboutToBeDestroyed(graph, node);
            }

            if (!isFirst())
            {
                m_prev->nodeAboutToBeDestroyed(graph, node);
            }
        }

        virtual void edgeCreated(const Graph& graph, const EdgeHandle& edge) override
        {
            if (isValid())
            {
                m_valid = m_checker.edgeCreated(graph, edge);
            }

            if (!isFirst())
            {
                m_prev->edgeCreated(graph, edge);
            }
        }

        virtual void edgeAboutToBeDestroyed(const Graph& graph, const EdgeHandle& edge) override
        {
            if (isValid())
            {
                m_valid = m_checker.edgeAboutToBeDestroyed(graph, edge);
            }

            if (!isFirst())
            {
                m_prev->edgeAboutToBeDestroyed(graph, edge);
            }
        }

        virtual void edgeAboutToBeRelinked(const Graph& graph,
                                   const EdgeHandle& edge,
                                   const NodeHandle& newSrcNode,
                                   const NodeHandle& newDstNode) override
        {
            if (isValid())
            {
                m_valid = m_checker.edgeAboutToBeRelinked(graph, edge, newSrcNode, newDstNode);
            }

            if (!isFirst())
            {
                m_prev->edgeAboutToBeRelinked(graph, edge, newSrcNode, newDstNode);
            }
        }
    };

    struct LazyPasses final
    {
        LazyPassWrapper* last = nullptr;
        std::unordered_map<std::string, std::unique_ptr<LazyPassWrapper>> passes;

        template<typename P, typename C>
        void addPass(const std::string& name, P&& pass, C&& checker)
        {
            ASSERT(!name.empty());
            ASSERT(!util::contains(passes, name));
            std::unique_ptr<LazyPassWrapper> wrapper;
            ASSERT(passes.empty() == (nullptr == last));
            using PassT = LazyPassImpl<util::remove_reference_t<P>,util::remove_reference_t<C> >;
            wrapper.reset(new PassT(last, std::forward<P>(pass), std::forward<C>(checker)));
            auto it = passes.emplace(std::make_pair(name, std::move(wrapper))).first;
            last = it->second.get();
            ASSERT(nullptr != last);
        }

        IGraphListener* getListener() const;
        LazyPassWrapper* getPass(const std::string& name) const;
        void reset();
    };

    template<typename Range>
    std::vector<LazyPassWrapper*> getLazyPasses(const Range& names) const
    {
        std::vector<LazyPassWrapper*> ret;
        for (auto&& name: names)
        {
            ASSERT(!std::string(name).empty());
            ret.emplace_back(m_lazyPasses.getPass(name));
        }
        return ret;
    }

    std::vector<std::unique_ptr<ExecutionBackend>> m_backends;

    CallbackList m_prePassCallbacks;
    CallbackList m_postPassCallbacks;

    LazyPasses m_lazyPasses;
    std::unordered_set<std::string> m_executableDependencies;

    PassMan m_passManager;
};

class ExecutionEngineSetupContext final
{
public:
    using PassCallback = ExecutionEngine::PassCallback;
    using StagesRange = ExecutionEngine::StagesRange;

    ExecutionEngineSetupContext(ExecutionEngine& e);
    ExecutionEngineSetupContext(const ExecutionEngineSetupContext&) = delete;
    ExecutionEngineSetupContext& operator=(const ExecutionEngineSetupContext&) = delete;

    /// Add pre pass callback to the list
    void addPrePassCallback(PassCallback callback);

    /// Add post pass callback to the list
    void addPostPassCallback(PassCallback callback);

    /// Add lazy pass to engine
    /// @param passName     Name of the pass, must be unique ant not to be empty
    /// @param pass         Pass object
    template<typename T>
    void addLazyPass(const std::string& passName, T&& pass)
    {
        m_engine.addLazyPass(passName, std::forward<T>(pass));
    }

    /// Adds lazy pass as final executable dependency
    /// It will be executed even if there are no passes depending on it
    /// @param lazyPassName Name of the pass, must be one the passes added via addLazyPass
    void addExecutableDependency(const std::string& lazyPassName);

    /// Add pass stage
    /// @param stageName    Name of stage to be added, must not be empty
    void addPassStage(const std::string& stageName);

    /// Add pass stage before specified stage
    /// @param stageName    Name of stage to be added, must not be empty
    /// @param prevStage    Name of the previous stage, must not be empty
    void addPassStage(const std::string& stageName, const std::string& prevStage);

    /// Returns range with stages names
    StagesRange passStages() const;

    /// Add pass to specified stage
    /// @param stageName    Name of the stage pass to be added, must not be empty
    /// @param passName     Name of the pass, must not be empty
    /// @param pass         Pass object
    template<typename T>
    void addPass(const std::string& stageName, const std::string& passName, T&& pass)
    {
        m_engine.addPass(stageName, passName, std::forward<T>(pass));
    }

    /// Add pass to specified stage
    /// @param stageName       Name of the stage pass to be added, must not be empty
    /// @param passName        Name of the pass, must not be empty
    /// @param pass            Pass object
    /// @param lazyPassesNames List of lasy passes names from which this pass is depends
    template<typename T, typename Range>
    void addPass(const std::string& stageName, const std::string& passName, T&& pass, Range&& lazyPassesNames)
    {
        m_engine.addPass(stageName, passName, std::forward<T>(pass), std::forward<Range>(lazyPassesNames));
    }

private:
    ExecutionEngine& m_engine;
};

}

#endif // EXECUTION_ENGINE_HPP
