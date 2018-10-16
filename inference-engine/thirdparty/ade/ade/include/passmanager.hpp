// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef PASSMANAGER_HPP
#define PASSMANAGER_HPP

#include <vector>
#include <list>
#include <unordered_map>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>

#include "util/assert.hpp"
#include "util/range.hpp"

namespace ade
{

template<typename Context>
class PassList;

template<typename Context>
class PassManager final
{
public:
    using StageList    = std::list<std::pair<std::string, PassList<Context>>>;
    using StagesRange  = util::IterRange<typename StageList::iterator>;
    using StagesCRange = util::IterRange<typename StageList::const_iterator>;

    PassManager() = default;
    PassManager(PassManager&&) = default;
    PassManager& operator=(PassManager&&) = default;

    void addStage(const std::string& stageName, const std::string& prevStage)
    {
        ASSERT(!stageName.empty());
        ASSERT(!prevStage.empty());
        ASSERT(!hasStage(stageName));
        ASSERT(hasStage(prevStage));
        auto it = m_stagesMap.find(prevStage)->second;

        ++it;
        m_stages.insert(it, std::make_pair(stageName, PassList<Context>{}));
        --it;
        m_stagesMap.insert(std::make_pair(stageName, it)); //TODO: exception safety
    }
    void addStage(const std::string& stageName)
    {
        ASSERT(!stageName.empty());
        ASSERT(!hasStage(stageName));
        m_stages.emplace_back(std::make_pair(stageName, PassList<Context>{}));
        auto it = m_stages.end();
        --it;
        m_stagesMap.insert(std::make_pair(stageName, it)); //TODO: exception safety
    }

    PassList<Context>& getStage(const std::string& stageName)
    {
        auto it = m_stagesMap.find(stageName);
        ASSERT(m_stagesMap.end() != it);
        return (*it->second).second;
    }


    template<typename T>
    void addPass(const std::string& stageName, T&& pass)
    {
        auto& stage = getStage(stageName);
        stage.addPass(std::forward<T>(pass));
    }

    void run(Context& context)
    {
        for (auto& pass: m_stages)
        {
            pass.second.run(context);
        }
    }

    StagesRange stages()
    {
        return util::toRange(m_stages);
    }

    StagesCRange stages() const
    {
        return util::toRange(m_stages);
    }

private:
    PassManager(const PassManager&) = delete;
    PassManager& operator=(const PassManager&) = delete;

    bool hasStage(const std::string& name) const
    {
        return m_stagesMap.end() != m_stagesMap.find(name);
    }

    StageList m_stages;
    using StageMap = std::unordered_map<std::string, typename StageList::iterator>;
    StageMap m_stagesMap;
};

namespace detail
{
template<typename Context>
class PassConceptBase
{
public:
    virtual ~PassConceptBase() {}
    virtual void run(Context& context) = 0;
};

template<typename Context, typename PassT>
class PassConceptImpl : public PassConceptBase<Context>
{
    PassT m_pass;
public:
    template<typename T>
    PassConceptImpl(T&& pass): m_pass(std::forward<T>(pass)) {}

    virtual void run(Context& context) override
    {
        m_pass(context);
    }
};
}

template<typename Context>
class PassList final
{
public:
    PassList() = default;
    PassList(PassList&&) = default;
    PassList& operator=(PassList&&) = default;

    template<typename PassT>
    void addPass(PassT&& pass)
    {
        m_passes.emplace_back(new detail::PassConceptImpl<Context, PassT>(std::forward<PassT>(pass)));
    }

    void run(Context& context)
    {
        for (auto& pass: m_passes)
        {
            pass->run(context);
        }
    }

private:
    PassList(const PassList&) = delete;
    PassList& operator=(const PassList&) = delete;

    std::vector<std::unique_ptr<detail::PassConceptBase<Context>>> m_passes;
};

}

#endif // PASSMANAGER_HPP
