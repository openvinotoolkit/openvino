//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <cstring>
#include <memory>

#include "host_tensor.hpp"
#include "ngraph/chrome_trace.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

static const size_t alignment = 64;

runtime::HostTensor::HostTensor(const ngraph::element::Type& element_type,
                                const Shape& shape,
                                void* memory_pointer,
                                const string& name)
    : runtime::Tensor(std::make_shared<ngraph::descriptor::Tensor>(element_type, shape, name))
    , m_memory_pointer(memory_pointer)
{
    if (get_partial_shape().is_static() && get_element_type().is_static())
    {
        allocate_buffer();
    }
}

runtime::HostTensor::HostTensor(const element::Type& element_type,
                                const Shape& shape,
                                const std::string& name)
    : HostTensor(element_type, shape, nullptr, name)
{
}

runtime::HostTensor::HostTensor(const element::Type& element_type,
                                const PartialShape& partial_shape,
                                const std::string& name)
    : runtime::Tensor(
          std::make_shared<ngraph::descriptor::Tensor>(element_type, partial_shape, name))
{
    // Defer allocation until ptr is requested
}

runtime::HostTensor::HostTensor(const std::string& name)
    : HostTensor(element::dynamic, PartialShape::dynamic())
{
}

runtime::HostTensor::HostTensor(const Output<Node>& value)
    : HostTensor(value.get_element_type(), value.get_partial_shape(), value.get_tensor().get_name())
{
}

void runtime::HostTensor::allocate_buffer()
{
    NGRAPH_CHECK(get_partial_shape().is_static(),
                 "Attempt to allocate buffer for tensor with partial shape: ",
                 get_partial_shape());
    NGRAPH_CHECK(get_element_type().is_static(),
                 "Attempt to allocate buffer for tensor with dynamic type: ",
                 get_element_type());
    m_descriptor->set_tensor_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorLayout>(*m_descriptor));
    m_buffer_size = m_descriptor->get_tensor_layout()->get_size() * get_element_type().size();
    if (m_memory_pointer != nullptr)
    {
        m_aligned_buffer_pool = m_memory_pointer;
    }
    else
    {
        // Add 1 so that even for zero-sized tensor we get at least 1 byte
        size_t allocation_size = m_buffer_size + alignment + 1;
        uint8_t* allocated_buffer_pool = static_cast<uint8_t*>(ngraph_malloc(allocation_size));
        m_allocated_buffer_pool = allocated_buffer_pool;
        size_t mod = size_t(allocated_buffer_pool) % alignment;
        if (mod == 0)
        {
            m_aligned_buffer_pool = allocated_buffer_pool;
        }
        else
        {
            m_aligned_buffer_pool = (allocated_buffer_pool + alignment - mod);
        }
    }
}

runtime::HostTensor::HostTensor(const std::shared_ptr<op::v0::Constant>& constant)
    : HostTensor(constant->output(0).get_tensor().get_name())
{
    initialize(constant);
}

void runtime::HostTensor::initialize(const std::shared_ptr<op::v0::Constant>& constant)
{
    set_element_type(constant->get_output_element_type(0));
    set_shape(constant->get_output_shape(0));
    memcpy(get_data_ptr(), constant->get_data_ptr(), get_size_in_bytes());
}

runtime::HostTensor::~HostTensor()
{
    if (m_allocated_buffer_pool != nullptr)
    {
        ngraph_free(m_allocated_buffer_pool);
    }
}

void* runtime::HostTensor::get_data_ptr()
{
    if (!m_aligned_buffer_pool)
    {
        allocate_buffer();
    }
    return m_aligned_buffer_pool;
}

const void* runtime::HostTensor::get_data_ptr() const
{
    NGRAPH_CHECK(m_aligned_buffer_pool, "Buffer not initialized");
    return m_aligned_buffer_pool;
}

void runtime::HostTensor::write(const void* source, size_t n)
{
    event::Duration d1("write", "HostTensor");
    void* target = get_data_ptr();
    if (n != m_buffer_size)
    {
        throw out_of_range("partial tensor write not supported");
    }
    if (n > 0)
    {
        if (!source)
        {
            throw runtime_error("nullptr passed to HostTensor::write");
        }
        memcpy(target, source, n);
    }
}

void runtime::HostTensor::read(void* target, size_t n) const
{
    event::Duration d1("read", "HostTensor");
    const void* source = get_data_ptr();
    if (n != m_buffer_size)
    {
        throw out_of_range("partial tensor read access not supported");
    }
    if (n > 0)
    {
        if (!target)
        {
            throw runtime_error("nullptr passed to HostTensor::read");
        }
        memcpy(target, source, n);
    }
}

bool runtime::HostTensor::get_is_allocated() const
{
    return m_aligned_buffer_pool != nullptr;
}

void runtime::HostTensor::set_element_type(const element::Type& element_type)
{
    NGRAPH_CHECK(get_element_type().is_dynamic() || get_element_type() == element_type,
                 "Can not change a static element type");
    m_descriptor->set_element_type(element_type);
}

void runtime::HostTensor::set_shape(const Shape& shape)
{
    NGRAPH_CHECK(PartialShape(shape).refines(get_partial_shape()),
                 "Allocation shape ",
                 shape,
                 " must be compatible with the partial shape: ",
                 get_partial_shape());
    m_descriptor->set_partial_shape(shape);
}

void runtime::HostTensor::set_unary(const HostTensorPtr& arg)
{
    set_element_type(arg->get_element_type());
    set_shape(arg->get_partial_shape().get_shape());
}

void runtime::HostTensor::set_broadcast(const op::AutoBroadcastSpec& autob,
                                        const HostTensorPtr& arg0,
                                        const HostTensorPtr& arg1)
{
    element::Type element_type = arg0->get_element_type();
    NGRAPH_CHECK(element::Type::merge(element_type, element_type, arg1->get_element_type()),
                 "Argument element types are inconsistent.");
    set_broadcast(autob, arg0, arg1, element_type);
}

void runtime::HostTensor::set_broadcast(const op::AutoBroadcastSpec& autob,
                                        const HostTensorPtr& arg0,
                                        const HostTensorPtr& arg1,
                                        const element::Type& element_type)
{
    set_element_type(element_type);

    PartialShape pshape = arg0->get_partial_shape();
    if (autob.m_type == op::AutoBroadcastType::NONE)
    {
        NGRAPH_CHECK(PartialShape::merge_into(pshape, arg1->get_partial_shape()),
                     "Argument shapes are inconsistent.");
    }
    else if (autob.m_type == op::AutoBroadcastType::NUMPY ||
             autob.m_type == op::AutoBroadcastType::PDPD)
    {
        NGRAPH_CHECK(PartialShape::broadcast_merge_into(pshape, arg1->get_partial_shape(), autob),
                     "Argument shapes are inconsistent.");
    }
    else
    {
        NGRAPH_CHECK(false, "Unsupported auto broadcast specification");
    }
    set_shape(pshape.get_shape());
}
