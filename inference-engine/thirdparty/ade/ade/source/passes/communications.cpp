// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <passes/communications.hpp>

#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <stdexcept>

#include <typed_graph.hpp>

#include <communication/comm_buffer.hpp>
#include <communication/comm_interface.hpp>
#include <communication/callback_connector.hpp>

#include <memory/memory_descriptor.hpp>
#include <memory/memory_descriptor_view.hpp>

#include <util/algorithm.hpp>
#include <util/chain_range.hpp>

#include <memory/alloc.hpp>

namespace
{

using NodeHasher = ade::HandleHasher<ade::Node>;

struct CacheEntry final
{
    std::unordered_set<ade::NodeHandle, NodeHasher> commNodes;
    std::unordered_set<ade::NodeHandle, NodeHasher> producers;
    std::unordered_set<ade::NodeHandle, NodeHasher> consumers;
};

using Cache = std::unordered_map<ade::MemoryDescriptorView*, CacheEntry>;

struct CallbackCacheEntry final
{
    std::unordered_set<ade::NodeHandle, NodeHasher> producers;
    std::unordered_set<ade::NodeHandle, NodeHasher> consumers;
};

using CallbackCache = std::unordered_map<ade::NodeHandle, CallbackCacheEntry, NodeHasher>;


ade::MemoryDescriptorView* findParentView(ade::MemoryDescriptorView* view)
{
    ASSERT(nullptr != view);

    auto parent = view->getParentView();
    if (nullptr != parent)
    {
        return findParentView(parent);
    }
    return view;
}

void visitProducer(Cache& cache,
                   CallbackCache& callbackCache,
                   const ade::NodeHandle& commNode,
                   const ade::NodeHandle& node,
                   ade::passes::ConnectCommChannels::Context& ctx)
{
    ASSERT(nullptr != node);
    ASSERT(ctx.graph.metadata(node).contains<ade::meta::DataObject>());
    auto memDesc = findParentView(ctx.graph.metadata(node).get<ade::meta::DataObject>().dataRef.getView());
    ASSERT(nullptr != memDesc);
    bool connectedToNode = false;
    for (auto edge: node->inEdges())
    {
        auto srcNode = edge->srcNode();
        if (ctx.graph.metadata(srcNode).contains<ade::meta::NodeInfo>())
        {
            connectedToNode = true;
            callbackCache[commNode].producers.insert(srcNode);
        }
        else if (ctx.graph.metadata(srcNode).contains<ade::meta::DataObject>())
        {
            visitProducer(cache, callbackCache, commNode, srcNode, ctx);
        }
    }

    if (connectedToNode)
    {
        cache[memDesc].producers.insert(node);
        cache[memDesc].commNodes.insert(commNode);
    }
}

void visitConsumer(Cache& cache,
                   CallbackCache& callbackCache,
                   const ade::NodeHandle& commNode,
                   const ade::NodeHandle& node,
                   ade::passes::ConnectCommChannels::Context& ctx)
{
    ASSERT(nullptr != node);
    ASSERT(ctx.graph.metadata(node).contains<ade::meta::DataObject>());
    auto memDesc = findParentView(ctx.graph.metadata(node).get<ade::meta::DataObject>().dataRef.getView());
    ASSERT(nullptr != memDesc);
    bool connectedToNode = false;
    for (auto edge: node->outEdges())
    {
        auto dstNode = edge->dstNode();
        if (ctx.graph.metadata(dstNode).contains<ade::meta::NodeInfo>())
        {
            connectedToNode = true;
            callbackCache[commNode].consumers.insert(dstNode);
        }
        else if (ctx.graph.metadata(dstNode).contains<ade::meta::DataObject>())
        {
            visitConsumer(cache, callbackCache, commNode, dstNode, ctx);
        }
    }

    if (connectedToNode)
    {
        cache[memDesc].consumers.insert(node);
        cache[memDesc].commNodes.insert(commNode);
    }
}

struct DataObject final
{
    ade::MemoryDescriptorRef memory_ref;
    std::vector<ade::NodeHandle> commNodes;
    std::vector<ade::NodeHandle> producers;
    std::vector<ade::NodeHandle> consumers;
};

struct CallbackObject final
{
    ade::NodeHandle commNode;
    std::vector<ade::NodeHandle> producers;
    std::vector<ade::NodeHandle> consumers;
};

struct CommObjects
{
    std::vector<DataObject> dataObjects;
    std::vector<CallbackObject> callbackObjects;
};

CommObjects collectDataObjects(ade::passes::ConnectCommChannels::Context& ctx)
{
    Cache cache;
    CallbackCache callbackCache;
    for (auto node: ctx.graph.nodes())
    {
        auto meta = ctx.graph.metadata(node);
        if (meta.contains<ade::meta::CommNode>())
        {
            for (auto edge: node->inEdges())
            {
                auto srcNode = edge->srcNode();
                visitProducer(cache, callbackCache, node, srcNode, ctx);
            }

            for (auto edge: node->outEdges())
            {
                auto dstNode = edge->dstNode();
                visitConsumer(cache, callbackCache, node, dstNode, ctx);
            }
        }
    }

    CommObjects ret;
    for (auto& obj: cache)
    {
        DataObject newObj;
        newObj.memory_ref = *obj.first;
        newObj.commNodes.reserve(obj.second.commNodes.size());
        newObj.producers.reserve(obj.second.producers.size());
        newObj.consumers.reserve(obj.second.consumers.size());
        util::copy(obj.second.commNodes, std::back_inserter(newObj.commNodes));
        util::copy(obj.second.producers, std::back_inserter(newObj.producers));
        util::copy(obj.second.consumers, std::back_inserter(newObj.consumers));
        ASSERT(!newObj.commNodes.empty());
        ASSERT(!newObj.producers.empty());
        ASSERT(!newObj.consumers.empty());
        ret.dataObjects.emplace_back(std::move(newObj));
    }

    for (auto& obj: callbackCache)
    {
        CallbackObject newObj;
        newObj.commNode = obj.first;
        newObj.producers.reserve(obj.second.producers.size());
        newObj.consumers.reserve(obj.second.consumers.size());
        util::copy(obj.second.producers, std::back_inserter(newObj.producers));
        util::copy(obj.second.consumers, std::back_inserter(newObj.consumers));
        ASSERT(!newObj.producers.empty());
        ASSERT(!newObj.consumers.empty());
        ret.callbackObjects.emplace_back(std::move(newObj));
    }
    return ret;
}

/// Fill common part of the BufferDesc
template<typename T>
ade::ICommChannel::BufferDesc fillBufferDesc(T& elem)
{
    auto memRef = elem.memory_ref;
    ASSERT(nullptr != memRef);
    ade::ICommChannel::BufferDesc bufferDesc;

    // Fill common part of the BufferDesc
    bufferDesc.writersCount = util::checked_cast<int>(elem.producers.size());
    bufferDesc.readersCount = util::checked_cast<int>(elem.consumers.size());
    bufferDesc.memoryRef    = memRef;
    return bufferDesc;
}

class HostBufferImpl final : public ade::IDataBuffer
{
public:
    HostBufferImpl(std::size_t elementSize,
                   const ade::memory::DynMdSize& size,
                   const ade::memory::DynMdSize& alignment);

    HostBufferImpl(const ade::MemoryDescriptorRef& memRef);

    ~HostBufferImpl();

    // IDataBuffer interface
    virtual MapId map(const Span& span, Access access) override;
    virtual void unmap(const MapId& id) override;
    virtual void finalizeWrite(const ade::IDataBuffer::Span& span) override;
    virtual void finalizeRead(const ade::IDataBuffer::Span& span) override;
    virtual Size alignment(const Span& span) override;

private:
    struct Deleter
    {
        void operator()(void* ptr) const
        {
            ASSERT(nullptr != ptr);
            ade::aligned_free(ptr);
        }
    };

    std::atomic<int> m_accessCount = {0};
    ade::memory::DynMdSize m_size;
    ade::memory::DynMdSize m_alignment;
    ade::memory::DynMdView<void> m_view;
    std::unique_ptr<void, Deleter> m_memory;
    ade::MemoryDescriptorRef m_memRef;
};

HostBufferImpl::HostBufferImpl(std::size_t elementSize,
                               const ade::memory::DynMdSize& size,
                               const ade::memory::DynMdSize& alignment):
    m_size(size),
    m_alignment(alignment),
    m_view(util::alloc_view<ade::memory::MaxDimensions>
           (elementSize,
            util::memory_range(size.data(),      size.dims_count()),
            util::memory_range(alignment.data(), alignment.dims_count()),
            [](std::size_t size, std::size_t align)
            {
                auto ptr = ade::aligned_alloc(size, align);
                if (nullptr == ptr)
                {
                    throw_error(std::bad_alloc());
                }
                return ptr;
            })),
    m_memory(m_view.mem.data)
{

}

HostBufferImpl::HostBufferImpl(const ade::MemoryDescriptorRef& memRef):
    m_size(memRef.span().size()),
    m_memRef(memRef)
{

}

HostBufferImpl::~HostBufferImpl()
{
    ASSERT(0 == m_accessCount);
}

ade::IDataBuffer::MapId HostBufferImpl::map(const Span& span, Access /*access*/)
{
    auto view = (nullptr != m_view ? m_view : m_memRef.getExternalView());
    ASSERT(nullptr != view);
    ASSERT(span.dims_count() == m_size.dims_count());
    auto accessCount = ++m_accessCount;
    ASSERT(accessCount > 0);
    return MapId{view.slice(span), 0};
}

void HostBufferImpl::unmap(const MapId& /*id*/)
{
    auto accessCount = --m_accessCount;
    ASSERT(accessCount >= 0);
}

void HostBufferImpl::finalizeWrite(const ade::IDataBuffer::Span& /*span*/)
{
    //Nothing
}

void HostBufferImpl::finalizeRead(const ade::IDataBuffer::Span& /*span*/)
{
    //Nothing
}

ade::IDataBuffer::Size HostBufferImpl::alignment(const Span& span)
{
    ASSERT(span.dims_count() == m_size.dims_count());
    // TODO: report actual alignment
    Size ret;
    ret.redim(span.dims_count());
    util::fill(ret, 1);
    return ret;
}

}

void ade::passes::ConnectCommChannels::operator()(ade::passes::ConnectCommChannels::Context ctx) const
{
    // Step 1:
    // Collect all data objects directly or indirectly connected to comm nodes
    // group them by MemoryDescriptor and by a commnode
    const auto commObjects = collectDataObjects(ctx);

    // Step 2:
    // Check comm channels and callbacks validity
    {
        // Step 2.1
        // Check comm channels validity
        for (auto& elem: commObjects.dataObjects)
        {
            for (auto node: util::chain(util::toRange(elem.producers),
                                        util::toRange(elem.consumers)))
            {
                auto meta = ctx.graph.metadata(node);
                if (!meta.contains<ade::meta::DataObject>() ||
                    !meta.contains<ade::meta::CommChannel>() ||
                    nullptr == meta.get<ade::meta::CommChannel>().channel)
                {
                    throw_error(std::runtime_error("Comm channel wasn't setup properly"));
                }
            }
        }

        // Step 2.2
        // Check comm callbacks validity
        for (auto& elem: commObjects.callbackObjects)
        {
            for (auto node: elem.consumers)
            {
                auto meta = ctx.graph.metadata(node);
                if (!meta.contains<ade::meta::CommConsumerCallback>() ||
                    nullptr == meta.get<ade::meta::CommConsumerCallback>().callback)
                {
                    throw_error(std::runtime_error("Consumer callback metadata error"));
                }
            }
        }
    }

    // Step 3:
    // Connect comm channels
    for (auto& elem: commObjects.dataObjects)
    {
        ade::ICommChannel::BufferDesc bufferDesc = fillBufferDesc(elem);

        // Step 3.1:
        // Collect buffer preferences
        ade::ICommChannel::BufferPrefs summary;
        summary.preferredAlignment.redim(bufferDesc.memoryRef.span().dims_count());
        util::fill(summary.preferredAlignment, 1);
        for (auto node: util::chain(util::toRange(elem.producers),
                                    util::toRange(elem.consumers)))
        {
            auto meta = ctx.graph.metadata(node);
            auto channel = meta.get<ade::meta::CommChannel>().channel;
            ASSERT(nullptr != channel);
            ade::ICommChannel::BufferPrefs prefs = channel->getBufferPrefs(bufferDesc);
            ASSERT(prefs.preferredAlignment.dims_count() == summary.preferredAlignment.dims_count());
            for (auto i: util::iota(summary.preferredAlignment.dims_count()))
            {
                ASSERT(prefs.preferredAlignment[i] > 0);
                // TODO: assert alignment is power of 2
                summary.preferredAlignment[i] =
                        std::max(summary.preferredAlignment[i],
                                 prefs.preferredAlignment[i]);
            }
        }

        // Step 3.2:
        // Try to get buffer from channels
        std::unique_ptr<ade::IDataBuffer> buffer;
        for (auto node: util::chain(util::toRange(elem.producers),
                                    util::toRange(elem.consumers)))
        {
            ASSERT(nullptr == buffer);
            auto meta = ctx.graph.metadata(node);
            auto channel = meta.get<ade::meta::CommChannel>().channel;
            ASSERT(nullptr != channel);
            buffer = channel->getBuffer(bufferDesc, summary);
            if (nullptr != buffer)
            {
                break;
            }
        }

        if (nullptr == buffer)
        {
            // Step 3.3:
            // Buffer wasn't allocated by plugins, allocate it by framework
            if (nullptr == bufferDesc.memoryRef.getExternalView())
            {
                buffer.reset(new HostBufferImpl(bufferDesc.memoryRef.elementSize(),
                                                bufferDesc.memoryRef.size(),
                                                summary.preferredAlignment));
            }
            else
            {
                // Use existing buffer (e.g. from non-virtual object)
                buffer.reset(new HostBufferImpl(bufferDesc.memoryRef));
            }
        }

        // Step 3.4:
        // Notify plugins about buffer object
        ASSERT(nullptr != buffer);
        for (auto node: util::chain(util::toRange(elem.producers),
                                    util::toRange(elem.consumers)))
        {
            auto meta = ctx.graph.metadata(node);
            auto channel = meta.get<ade::meta::CommChannel>().channel;
            channel->setBuffer(ade::DataBufferView(buffer.get(), bufferDesc.memoryRef.span()), bufferDesc);
        }
        std::shared_ptr<ade::IDataBuffer> sharedBuffer(std::move(buffer));
        for (auto commNode: elem.commNodes)
        {
            auto meta = ctx.graph.metadata(commNode);
            meta.get<ade::meta::CommNode>().addDataBuffer(sharedBuffer);
        }
    }

    // Step 4
    // Connect comm objects callbacks
    {
        // Multiple comm nodes can be attached to single producer data object
        // so we need to collect and merge them
        std::unordered_map<ade::NodeHandle, std::vector<std::function<void()>>, NodeHasher> producerCallbacks;
        for (auto& elem: commObjects.callbackObjects)
        {
            ASSERT(nullptr != elem.commNode);
            ASSERT(!elem.producers.empty() && !elem.consumers.empty());

            ade::CallbackConnector<> connector(util::checked_cast<int>(elem.producers.size()),
                                               util::checked_cast<int>(elem.consumers.size()));

            // Step 4.1
            // Collect callbacks from consumers
            for (auto& consumer: elem.consumers)
            {
                auto meta = ctx.graph.metadata(consumer);
                auto callback = std::move(meta.get<ade::meta::CommConsumerCallback>().callback);
                ASSERT(nullptr != callback);
                connector.addConsumerCallback(std::move(callback));
            }

            // Step 4.2
            // Create producer callbacks
            auto resetter = connector.finalize();
            if (nullptr != resetter)
            {
                auto meta = ctx.graph.metadata();
                if (!meta.contains<ade::meta::Finalizers>())
                {
                    meta.set(ade::meta::Finalizers());
                }
                meta.get<ade::meta::Finalizers>().finalizers.emplace_back(std::move(resetter));
            }

            // Step 4.3
            // Collect producer callbacks
            for (auto& producer: elem.producers)
            {
                auto callback = connector.getProducerCallback();
                ASSERT(nullptr != callback);
                producerCallbacks[producer].emplace_back(std::move(callback));
            }
        }

        // Step 4.4
        // Assign producer callbacks
        for (auto& elem: producerCallbacks)
        {
            auto producer = elem.first;

            auto callbacks = std::move(elem.second);
            ASSERT(!callbacks.empty());

            auto meta = ctx.graph.metadata(producer);
            if (!meta.contains<ade::meta::CommProducerCallback>())
            {
                meta.set(ade::meta::CommProducerCallback());
            }

            if (1 == callbacks.size())
            {
                // Assign directly
                meta.get<ade::meta::CommProducerCallback>().callback = callbacks[0];
            }
            else
            {
                // Create wrapper to call all callbacks
                struct Connector final
                {
                    std::vector<std::function<void()>> callbacks;

                    void operator()() const
                    {
                        ASSERT(!callbacks.empty());
                        for (auto& callback: callbacks)
                        {
                            ASSERT(nullptr != callback);
                            callback();
                        }
                    }
                };

                meta.get<ade::meta::CommProducerCallback>().callback = Connector{std::move(callbacks)};
            }
        }
    }
}

const char* ade::passes::ConnectCommChannels::name()
{
    return "ade::passes::ConnectCommChannels";
}
