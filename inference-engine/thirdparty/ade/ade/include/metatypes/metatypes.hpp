// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef METATYPES_HPP
#define METATYPES_HPP

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <memory/memory_descriptor_ref.hpp>

namespace ade
{
class ICommChannel;
class IDataBuffer;

namespace meta
{
struct NodeInfo final
{
    std::string kernel_name;
    std::string target_name;

    using NodeId = void*;

    NodeInfo() = default;
    NodeInfo(const std::string& kernel,
             const std::string& target);
    NodeInfo(const NodeInfo&) = default;
    NodeInfo& operator=(const NodeInfo&) = default;
    NodeInfo(NodeInfo&&) = default;
    NodeInfo& operator=(NodeInfo&&) = default;

    NodeId getId() const;

    void link(NodeInfo& node);

    static const char* name();
private:
    struct IdStruct final
    {
        // Nothing
    };

    using IdPtr = std::shared_ptr<IdStruct>;

    // mutable to allocate id lazily
    mutable IdPtr m_id;

    IdPtr getIdInternal() const;
};

struct DataObject final
{
    ade::MemoryDescriptorRef dataRef;

    std::string originalFormat; /// Original format of composite object

    static const char* name();
};

struct CommNode final
{
    CommNode(int producersCount);
    CommNode(const CommNode&) = default;
    CommNode& operator=(const CommNode&) = default;

    int producersCount() const { return m_producersCount; }

    void addDataBuffer(const std::shared_ptr<ade::IDataBuffer>& buff);

    static const char* name();
private:
    int m_producersCount = 0;
    std::vector<std::shared_ptr<ade::IDataBuffer>> m_buffers;
};

struct CommChannel final
{
    std::shared_ptr<ade::ICommChannel> channel;

    static const char* name();
};

/// Backends must set this metadata to their consumer nodes in heterogeneous case.
struct CommConsumerCallback final
{
    using Callback = std::function<void(void)>;

    Callback callback;

    static const char* name();
};

/// The framework will set this metadata to producers nodes in heterogeneous case.
/// Backends must use it in their executable representation.
struct CommProducerCallback final
{
    using Callback = std::function<void(void)>;

    Callback callback;

    static const char* name();
};

/// Store various finalizers which should run after graph execution completion
struct Finalizers final
{
    std::vector<std::function<void()>> finalizers;

    static const char* name();
};

std::ostream& operator<<(std::ostream& os, const ade::meta::NodeInfo& obj);

std::ostream& operator<<(std::ostream& os, const ade::meta::CommNode& obj);

std::ostream& operator<<(std::ostream& os, const ade::meta::CommConsumerCallback& obj);

std::ostream& operator<<(std::ostream& os, const ade::meta::CommProducerCallback& obj);

std::ostream& operator<<(std::ostream& os, const ade::meta::DataObject& obj);

std::ostream& operator<<(std::ostream& os, const ade::meta::CommChannel& obj);

std::ostream& operator<<(std::ostream& os, const ade::meta::Finalizers& obj);

}
}

#endif // METATYPES_HPP
