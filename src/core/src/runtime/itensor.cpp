// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/itensor.hpp"

#include "dev/make_tensor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {

ITensor::~ITensor() = default;

size_t ITensor::get_size() const {
    return shape_size(get_shape());
}

size_t ITensor::get_byte_size() const {
    return (get_size() * get_element_type().bitwidth() + 8 - 1) / 8;
}

/**
 * @brief View tensor to external memory
 * The tensor doesn't own the external memory
 */
class ViewTensor : public ITensor {
public:
    ViewTensor(const element::Type element_type, const Shape& shape, void* ptr)
        : m_element_type{element_type},
          m_shape{shape},
          m_capacity{shape},
          m_ptr{ptr} {
        OPENVINO_ASSERT(m_ptr != nullptr);
        OPENVINO_ASSERT(m_element_type != element::undefined && m_element_type != element::dynamic);
        update_strides();
    }

    void* data(const element::Type& element_type) const override {
        if (element_type != element::undefined && element_type != element::dynamic) {
            OPENVINO_ASSERT(element_type == get_element_type(),
                            "Tensor data with element type ",
                            get_element_type(),
                            ", is not representable as pointer to ",
                            element_type);
        }
        return m_ptr;
    }

    const element::Type& get_element_type() const override {
        return m_element_type;
    }

    const Shape& get_shape() const override {
        return m_shape;
    }

    void set_shape(ov::Shape new_shape) override {
        OPENVINO_ASSERT(shape_size(new_shape) <= ov::shape_size(m_capacity), "Could set new shape: ", new_shape);
        m_shape = std::move(new_shape);
        update_strides();
    }

    const Strides& get_strides() const override {
        OPENVINO_ASSERT(m_element_type.bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                        m_element_type);
        return m_strides;
    }

protected:
    void update_strides() {
        if (m_element_type.bitwidth() < 8)
            return;
        auto& shape = get_shape();
        m_strides.clear();
        if (!shape.empty()) {
            m_strides.resize(shape.size());
            m_strides.back() = m_element_type.size();
            std::copy(shape.rbegin(), shape.rend() - 1, m_strides.rbegin() + 1);
            std::partial_sum(m_strides.rbegin(), m_strides.rend(), m_strides.rbegin(), std::multiplies<size_t>());
        }
    }

    element::Type m_element_type;
    Shape m_shape;
    Shape m_capacity;
    Strides m_strides;
    void* m_ptr;
};

/**
 * @brief View tensor on external memory with strides
 */
class StridedViewTensor : public ViewTensor {
public:
    StridedViewTensor(const element::Type element_type, const Shape& shape, void* ptr, const Strides& strides)
        : ViewTensor{element_type, shape, ptr} {
        OPENVINO_ASSERT(
            get_element_type().bitwidth() >= 8,
            "Could not create strided access tensor for types with bitwidths less then 8 bit. Tensor type: ",
            get_element_type());
        // Save default strides
        auto shape_strides = m_strides;
        // Change strides
        m_strides = strides;
        OPENVINO_ASSERT(m_shape.size() == m_strides.size());

        for (size_t i = 0; i < m_strides.size(); ++i) {
            OPENVINO_ASSERT(shape_strides[i] <= m_strides[i],
                            "shape stride: ",
                            shape_strides[i],
                            ", stride: ",
                            m_strides[i]);
            OPENVINO_ASSERT((m_strides[i] % get_element_type().size()) == 0,
                            "shape stride: ",
                            shape_strides[i],
                            ", stride: ",
                            m_strides[i]);
            if (i) {
                OPENVINO_ASSERT(m_strides[i - 1] >= m_strides[i] * shape[i],
                                "Strides: ",
                                m_strides,
                                " are incompatible with shapes: ",
                                m_shape);
            }
        }
    }

    void set_shape(ov::Shape new_shape) override {
        OPENVINO_ASSERT(m_capacity.size() == new_shape.size(),
                        "Cannot set new shape: ",
                        new_shape,
                        " for tensor with strides! Shapes are not compatible.");
        for (size_t i = 0; i < new_shape.size(); i++) {
            OPENVINO_ASSERT(m_capacity[i] >= new_shape[i],
                            "Cannot set new shape: ",
                            new_shape,
                            " for tensor with strides! Dimension: ",
                            i,
                            " is not compatible.");
        }
        m_shape = std::move(new_shape);
    }
};

/**
 * @brief Creates view tensor on external memory
 *
 * @param element_type Tensor element type
 * @param shape Tensor shape
 * @param ptr pointer to external memoty
 * @param byte_strides Tensor strides
 *
 * @return Shared pointer to tensor interface
 */
std::shared_ptr<ITensor> make_tensor(const element::Type element_type,
                                     const Shape& shape,
                                     void* ptr,
                                     const Strides& byte_strides) {
    return byte_strides.empty() ? std::make_shared<ViewTensor>(element_type, shape, ptr)
                                : std::make_shared<StridedViewTensor>(element_type, shape, ptr, byte_strides);
}

/**
 * @brief Tensor with allocated memory
 * Tensor owns the memory
 */
class AllocatedTensor : public ViewTensor {
public:
    AllocatedTensor(const element::Type element_type, const Shape& shape, const Allocator& allocator)
        : ViewTensor{element_type,
                     shape,
                     [&] {
                         OPENVINO_ASSERT(allocator, "Allocator was not initialized");
                         return const_cast<Allocator&>(allocator).allocate(element_type.size() * shape_size(shape));
                     }()},
          m_allocator{allocator} {}

    ~AllocatedTensor() {
        m_allocator.deallocate(m_ptr, get_byte_size());
    }

    void set_shape(ov::Shape new_shape) override {
        auto old_byte_size = get_byte_size();
        m_shape = std::move(new_shape);
        if (get_byte_size() > old_byte_size) {
            m_allocator.deallocate(m_ptr, old_byte_size);
            m_ptr = m_allocator.allocate(get_byte_size());
        }
        update_strides();
    }

private:
    Allocator m_allocator;
};

/**
 * @brief Creates allocated tensor
 *
 * @param element_type Tensor element type
 * @param shape Tensor shape
 * @param allocator Tensor allocator
 *
 * @return Shared pointer to tensor interface
 */
std::shared_ptr<ITensor> make_tensor(const element::Type element_type, const Shape& shape, const Allocator& allocator) {
    return std::make_shared<AllocatedTensor>(element_type, shape, allocator);
}

/**
 * @brief ROI tensor on other tensor
 * ROI tensor holds the owner
 */
class RoiTensor : public ITensor {
public:
    RoiTensor(const std::shared_ptr<ITensor>& owner, const Coordinate& begin, const Coordinate& end) : m_owner{owner} {
        OPENVINO_ASSERT(owner->get_element_type().bitwidth() >= 8,
                        "ROI Tensor for types with bitwidths less then 8 bit is not implemented. Tensor type: ",
                        owner->get_element_type());
        auto owner_shape = owner->get_shape();
        OPENVINO_ASSERT(owner_shape.size() == begin.size());
        OPENVINO_ASSERT(begin.size() == end.size());
        m_shape.resize(begin.size());
        for (size_t i = 0; i < begin.size(); ++i) {
            OPENVINO_ASSERT(begin[i] <= owner_shape[i]);
            OPENVINO_ASSERT(end[i] <= owner_shape[i]);
            m_shape[i] = end[i] - begin[i];
            OPENVINO_ASSERT(m_shape[i] <= owner_shape[i]);
        }
        auto& strides = get_strides();
        m_offset = std::inner_product(begin.begin(), begin.end(), strides.begin(), static_cast<size_t>(0));
    }

    const element::Type& get_element_type() const override {
        return m_owner->get_element_type();
    }

    const Strides& get_strides() const override {
        return m_owner->get_strides();
    }

    const Shape& get_shape() const override {
        return m_shape;
    }

    void set_shape(ov::Shape new_shape) override {
        OPENVINO_THROW("Shapes cannot be changed for ROI Tensor");
    }

    void* data(const element::Type& element_type) const override {
        auto owner_data = m_owner->data(element_type);
        return static_cast<uint8_t*>(owner_data) + m_offset;
    }

private:
    std::shared_ptr<ITensor> m_owner;
    size_t m_offset;
    Shape m_shape;
};

/**
 * @brief Creates ROI tensor
 *
 * @param other Tensor what owns the memory
 * @param begin Begin coordinates
 * @param end End coordinates
 *
 * @return Shared pointer to tensor interface
 */
std::shared_ptr<ITensor> make_tensor(const std::shared_ptr<ITensor>& other,
                                     const Coordinate& begin,
                                     const Coordinate& end) {
    return std::make_shared<RoiTensor>(other, begin, end);
}

}  // namespace ov
