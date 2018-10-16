// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <metatypes/metatypes.hpp>
#include <sstream>

namespace ade
{
namespace meta
{

NodeInfo::NodeInfo(const std::string& kernel,
                   const std::string& target):
    kernel_name(kernel),
    target_name(target)
{

}

NodeInfo::NodeId NodeInfo::getId() const
{
    return getIdInternal().get();
}

void NodeInfo::link(NodeInfo& node)
{
    m_id = node.getIdInternal();
}

const char* NodeInfo::name()
{
    return "VxNodeInfo";
}

NodeInfo::IdPtr NodeInfo::getIdInternal() const
{
    if (nullptr == m_id)
    {
        m_id = std::make_shared<IdStruct>();
    }
    return m_id;
}

const char* DataObject::name()
{
    return "DataObject";
}

CommNode::CommNode(int producersCount):
    m_producersCount(producersCount)
{
    ASSERT(m_producersCount > 0);
}

void CommNode::addDataBuffer(const std::shared_ptr<ade::IDataBuffer>& buff)
{
    ASSERT(buff != nullptr);
    m_buffers.emplace_back(buff);
}

const char* CommNode::name()
{
    return "CommNode";
}

const char* CommChannel::name()
{
    return "CommChannel";
}

const char* CommConsumerCallback::name()
{
    return "CommConsumerCallback";
}

const char* CommProducerCallback::name()
{
    return "CommProducerCallback";
}

const char* Finalizers::name()
{
    return "Finalizers";
}

std::ostream& operator<<(std::ostream& os, const ade::meta::NodeInfo& obj)
{
    os << obj.kernel_name << " " << obj.target_name;
    return os;
}

std::ostream& operator<<(std::ostream& os, const ade::meta::CommNode& obj)
{
    os << "producer_count : " << obj.producersCount();
    return os;
}

std::ostream& operator<<(std::ostream& os, const ade::meta::CommConsumerCallback& /*obj*/)
{
    return os;
}

std::ostream& operator<<(std::ostream& os, const ade::meta::CommProducerCallback& /*obj*/)
{
    return os;
}

std::ostream& operator<<(std::ostream& os, const ade::meta::DataObject& obj)
{
    os << "mem descriptor ref: " << obj.dataRef << ", "
       << "originalFormat: " << obj.originalFormat;
    return os;
}

std::ostream& operator<<(std::ostream& os, const ade::meta::CommChannel& obj)
{
    os << obj.channel;
    return os;
}

std::ostream& operator<<(std::ostream& os, const ade::meta::Finalizers& /*obj*/)
{
    return os;
}

}
}
