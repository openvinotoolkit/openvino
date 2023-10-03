// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <memory>

#include "ngraph/descriptor/output.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START
namespace ov {
class Node;
namespace op {
namespace v0 {
class Constant;
}
}  // namespace op
}  // namespace ov
namespace ngraph {

namespace runtime {
class HostTensor;
}

using HostTensorPtr = std::shared_ptr<runtime::HostTensor>;
namespace op {
namespace v0 {
using ov::op::v0::Constant;
}
}  // namespace op
namespace runtime {
class NGRAPH_API NGRAPH_API_DEPRECATED HostTensor : public ngraph::runtime::Tensor {
public:
    HostTensor(const element::Type& element_type, const Shape& shape, void* memory_pointer);
    HostTensor(const element::Type& element_type, const Shape& shape);
    HostTensor(const element::Type& element_type, const PartialShape& partial_shape);
    HostTensor();
    explicit HostTensor(const Output<ov::Node>&);
    explicit HostTensor(const std::shared_ptr<op::v0::Constant>& constant);
    virtual ~HostTensor() override;

    void initialize(const std::shared_ptr<op::v0::Constant>& constant);

    void* get_data_ptr();
    const void* get_data_ptr() const;

    template <typename T>
    T* get_data_ptr() {
        return static_cast<T*>(get_data_ptr());
    }

    template <typename T>
    const T* get_data_ptr() const {
        return static_cast<T*>(get_data_ptr());
    }

    template <element::Type_t ET>
    typename element_type_traits<ET>::value_type* get_data_ptr() {
        NGRAPH_CHECK(ET == get_element_type(), "get_data_ptr() called for incorrect element type.");
        return static_cast<typename element_type_traits<ET>::value_type*>(get_data_ptr());
    }

    template <element::Type_t ET>
    const typename element_type_traits<ET>::value_type* get_data_ptr() const {
        NGRAPH_CHECK(ET == get_element_type(), "get_data_ptr() called for incorrect element type.");
        return static_cast<typename element_type_traits<ET>::value_type>(get_data_ptr());
    }

    /// \brief Write bytes directly into the tensor
    /// \param p Pointer to source of data
    /// \param n Number of bytes to write, must be integral number of elements.
    void write(const void* p, size_t n) override;

    /// \brief Read bytes directly from the tensor
    /// \param p Pointer to destination for data
    /// \param n Number of bytes to read, must be integral number of elements.
    void read(void* p, size_t n) const override;

    bool get_is_allocated() const;
    /// \brief Set the element type. Must be compatible with the current element type.
    /// \param element_type The element type
    void set_element_type(const element::Type& element_type);
    /// \brief Set the actual shape of the tensor compatibly with the partial shape.
    /// \param shape The shape being set
    void set_shape(const Shape& shape);
    /// \brief Set the shape of a node from an input
    /// \param arg The input argument
    void set_unary(const HostTensorPtr& arg);
    /// \brief Set the shape of the tensor using broadcast rules
    /// \param autob The broadcast mode
    /// \param arg0 The first argument
    /// \param arg1 The second argument
    void set_broadcast(const op::AutoBroadcastSpec& autob, const HostTensorPtr& arg0, const HostTensorPtr& arg1);
    /// \brief Set the shape of the tensor using broadcast rules
    /// \param autob The broadcast mode
    /// \param arg0 The first argument
    /// \param arg1 The second argument
    /// \param element_type The output element type
    void set_broadcast(const op::AutoBroadcastSpec& autob,
                       const HostTensorPtr& arg0,
                       const HostTensorPtr& arg1,
                       const element::Type& element_type);

protected:
    virtual void allocate_buffer();
    HostTensor(const HostTensor&) = delete;
    HostTensor(HostTensor&&) = delete;
    HostTensor& operator=(const HostTensor&) = delete;

    void* m_memory_pointer{nullptr};
    void* m_allocated_buffer_pool{nullptr};
    void* m_aligned_buffer_pool{nullptr};
    size_t m_buffer_size;
};
}  // namespace runtime
}  // namespace ngraph
NGRAPH_SUPPRESS_DEPRECATED_END
