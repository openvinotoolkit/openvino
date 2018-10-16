// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef CALLBACK_CONNECTOR_HPP
#define CALLBACK_CONNECTOR_HPP

#include <vector>
#include <functional>
#include <atomic>
#include <memory>

#include "util/assert.hpp"

namespace ade
{

/// This is helper class to connect one or more consumers callabaks
/// with one or more producer callbacks
///
/// All consumer callbacks will be notified when after each producer callbacks
/// was called
template<typename... Args>
class CallbackConnector final
{
public:
    using CallbackT = std::function<void(Args...)>;

    CallbackConnector(int producersCount, int consumersCount);
    CallbackConnector(const CallbackConnector&) = delete;
    CallbackConnector& operator=(const CallbackConnector&) = delete;

    /// Add consumer callback to connector
    /// All consumer callbacks must be added before finalization
    ///
    /// @param callback Callback to be added. Must not be null.
    void addConsumerCallback(CallbackT callback);

    /// Get producer callback from connector
    /// This method must be called only after finalization
    ///
    /// @returns Callback to be called by producers. Will never be null.
    CallbackT getProducerCallback();

    /// Prepares producer callbacks
    /// No more consumers callback can be added after call to this method
    /// Producer callbacks can be retrieved only after call to this method
    ///
    /// @returns Resetter which will be used to reset internal reference counters
    /// or null
    std::function<void()> finalize();
private:
    const int m_producersCount = 0;
    const int m_consumersCount = 0;
    std::vector<CallbackT> m_consumerCallbacks;
    CallbackT m_producerCallback;

    bool isFinalized() const;
};

template<typename... Args>
CallbackConnector<Args...>::CallbackConnector(int producersCount, int consumersCount):
    m_producersCount(producersCount), m_consumersCount(consumersCount)
{
    ASSERT(m_producersCount > 0);
    ASSERT(m_consumersCount > 0);
}

template<typename... Args>
void CallbackConnector<Args...>::addConsumerCallback(typename CallbackConnector::CallbackT callback)
{
    ASSERT(nullptr != callback);
    ASSERT(!isFinalized() && "Can't add callbacks to finalized comm node");
    m_consumerCallbacks.emplace_back(std::move(callback));
}

template<typename... Args>
typename CallbackConnector<Args...>::CallbackT CallbackConnector<Args...>::getProducerCallback()
{
    ASSERT(isFinalized() && "Can't get callbacks from unfinalized comm node");
    return m_producerCallback;
}

template<typename... Args>
std::function<void()> CallbackConnector<Args...>::finalize()
{
    ASSERT(!isFinalized() && "Finalized must called only once");
    ASSERT(!m_consumerCallbacks.empty());
    ASSERT(m_producersCount > 0);
    std::function<void()> resetter;
    if (1 == m_producersCount)
    {
        if (1 == m_consumerCallbacks.size())
        {
            // 1 producer and 1 consumer - connect them directly
            m_producerCallback = std::move(m_consumerCallbacks[0]);
        }
        else
        {
            // 1 producer and multiple consumers - wrap all consumers into functor
            // and assign it to producer
            struct Connector final
            {
                std::vector<CallbackT> consumerCallbacks;

                void operator()(Args... args)
                {
                    for (auto& func: consumerCallbacks)
                    {
                        func(args...);
                    }
                }
            };
            m_producerCallback = Connector{std::move(m_consumerCallbacks)};
        }
    }
    else
    {
        if (1 == m_consumerCallbacks.size())
        {
            // multiple producers and 1 consumer - wrap consumer into thread-safe reference-counted functor
            struct Connector final
            {
                const int initial;
                std::atomic<int> counter;
                CallbackT consumerCallback;

                Connector(int cnt, CallbackT&& func):
                    initial(cnt), counter(cnt), consumerCallback(std::move(func)) {}

                void reset()
                {
                    counter = initial;
                }

                void operator()(Args... args)
                {
                    auto res = --counter;
                    ASSERT(res >= 0);
                    if (0 == res)
                    {
                        consumerCallback(args...);
                    }
                }
            };
            auto connector = std::make_shared<Connector>(m_producersCount, std::move(m_consumerCallbacks[0]));
            m_producerCallback = [connector](Args... args)
            {
                (*connector)(args...);
            };
            resetter = [connector]()
            {
                connector->reset();
            };
        }
        else
        {
            // multiple producers and multiple consumers - wrap all consumers into thread-safe reference-counted functor
            struct Connector final
            {
                const int initial;
                std::atomic<int> counter;
                std::vector<CallbackT> consumerCallbacks;

                Connector(int cnt, std::vector<CallbackT>&& callbacks):
                    initial(cnt), counter(cnt), consumerCallbacks(std::move(callbacks)) {}

                void reset()
                {
                    counter = initial;
                }

                void operator()(Args... args)
                {
                    auto res = --counter;
                    ASSERT(res >= 0);
                    if (0 == res)
                    {
                        for (auto& func: consumerCallbacks)
                        {
                            func(args...);
                        }
                    }
                }
            };
            auto connector = std::make_shared<Connector>(m_producersCount, std::move(m_consumerCallbacks));
            m_producerCallback = [connector](Args... args)
            {
                (*connector)(args...);
            };
            resetter = [connector]()
            {
                connector->reset();
            };
        }
    }
    ASSERT(nullptr != m_producerCallback);
    return std::move(resetter);
}

template<typename... Args>
bool CallbackConnector<Args...>::isFinalized() const
{
    return (nullptr != m_producerCallback);
}

}

#endif // CALLBACK_CONNECTOR_HPP
