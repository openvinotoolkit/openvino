// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/make_tensor.hpp"

#include <memory>
#include <mutex>

#include "openvino/core/type/element_iterator.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#ifdef PROXY_PLUGIN_ENABLED
#    include "openvino/proxy/plugin.hpp"
#endif

namespace ov {

namespace {
Shape make_roi_shape(const Shape& tensor_shape, const Coordinate& begin, const Coordinate& end) {
    OPENVINO_ASSERT(tensor_shape.size() == begin.size());
    OPENVINO_ASSERT(begin.size() == end.size());

    auto roi_shape = Shape(begin.size());

    auto roi_begin = begin.begin();
    auto roi_end = end.begin();
    auto roi_dim = roi_shape.begin();
    auto max_dim = tensor_shape.begin();

    for (; max_dim != tensor_shape.end(); ++max_dim, ++roi_begin, ++roi_end, ++roi_dim) {
        OPENVINO_ASSERT(*roi_begin <= *max_dim);
        OPENVINO_ASSERT(*roi_end <= *max_dim);
        *roi_dim = *roi_end - *roi_begin;
        OPENVINO_ASSERT(*roi_dim <= *max_dim);
    }

    return roi_shape;
}
}  // namespace

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
          m_strides{},
          m_strides_once{},
          m_ptr{ptr} {
        OPENVINO_ASSERT(shape_size(shape) == 0 || m_ptr != nullptr);
        OPENVINO_ASSERT(m_element_type != element::undefined && m_element_type.is_static());
    }

    void* data(const element::Type& element_type) const override {
        if (element_type != element::undefined && element_type != element::dynamic &&
            (element_type.bitwidth() != get_element_type().bitwidth() ||
             element_type.is_real() != get_element_type().is_real() ||
             (element_type == element::string && get_element_type() != element::string) ||
             (element_type != element::string && get_element_type() == element::string))) {
            OPENVINO_THROW("Tensor data with element type ",
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
        m_strides.clear();
        update_strides();
    }

    const Strides& get_strides() const override {
        OPENVINO_ASSERT(m_element_type.bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                        m_element_type);
        std::call_once(m_strides_once, &ViewTensor::update_strides, this);
        return m_strides;
    }

protected:
    void update_strides() const {
        if (m_element_type.bitwidth() < 8)
            return;

        auto& shape = get_shape();
        if (m_strides.empty() && !shape.empty()) {
            m_strides.resize(shape.size());
            m_strides.back() = shape.back() == 0 ? 0 : m_element_type.size();
            std::transform(shape.crbegin(),
                           shape.crend() - 1,
                           m_strides.rbegin(),
                           m_strides.rbegin() + 1,
                           std::multiplies<size_t>());
        }
    }

    element::Type m_element_type;
    Shape m_shape;
    Shape m_capacity;
    mutable Strides m_strides;
    mutable std::once_flag m_strides_once;
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
        auto shape_strides = get_strides();
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
 * @param ptr pointer to external memory
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
                     [&shape, &element_type, &allocator] {
                         OPENVINO_ASSERT(allocator, "Allocator was not initialized");
                         const auto byte_size = element::get_byte_size(element_type, shape_size(shape));
                         auto data = const_cast<Allocator&>(allocator).allocate(byte_size);
                         initialize_elements(data, element_type, shape);
                         return data;
                     }()},
          m_allocator{allocator} {}

    ~AllocatedTensor() {
        destroy_memory();
    }

    void set_shape(ov::Shape new_shape) override {
        if (m_shape == new_shape)
            return;

        m_shape = std::move(new_shape);

        if (get_size() > get_capacity()) {
            destroy_memory();

            // allocate buffer and initialize objects from scratch
            m_capacity = m_shape;
            m_ptr = m_allocator.allocate(get_bytes_capacity());
            initialize_elements(m_ptr, m_element_type, m_shape);
        }

        m_strides.clear();
        update_strides();
    }

private:
    void destroy_elements(size_t begin_ind, size_t end_ind) {
        // it removes elements from tail
        if (get_element_type() == element::Type_t::string) {
            auto strings = static_cast<std::string*>(m_ptr);
            for (size_t ind = begin_ind; ind < end_ind; ++ind) {
                using std::string;
                strings[ind].~string();
            }
        }
    }

    void destroy_memory() {
        destroy_elements(0, get_capacity());
        m_allocator.deallocate(m_ptr, get_bytes_capacity());
        m_ptr = nullptr;
    }

    static void initialize_elements(void* data, const element::Type& element_type, const Shape& shape) {
        if (element_type == element::Type_t::string) {
            auto num_elements = shape_size(shape);
            auto string_ptr = static_cast<std::string*>(data);
            std::uninitialized_fill_n(string_ptr, num_elements, std::string());
        }
    }

    size_t get_capacity() const {
        return shape_size(m_capacity);
    }

    size_t get_bytes_capacity() const {
        return element::get_byte_size(get_element_type(), get_capacity());
    }

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
    RoiTensor(const std::shared_ptr<ITensor>& owner, const Coordinate& begin, const Coordinate& end)
        : m_owner{owner},
          m_shape{make_roi_shape(owner->get_shape(), begin, end)},
          m_capacity{m_shape},
          m_offset{std::inner_product(begin.begin(), begin.end(), get_strides().begin(), static_cast<size_t>(0))} {
        OPENVINO_ASSERT(get_element_type().bitwidth() >= 8,
                        "ROI Tensor for types with bitwidths less then 8 bit is not implemented. Tensor type: ",
                        get_element_type());
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
        OPENVINO_ASSERT(new_shape.size() == m_shape.size());
        for (auto new_dim = new_shape.cbegin(), max_dim = m_capacity.cbegin(); new_dim != new_shape.cend();
             ++max_dim, ++new_dim) {
            OPENVINO_ASSERT(*new_dim <= *max_dim,
                            "Cannot set new shape: ",
                            new_shape,
                            " for ROI tensor! Dimension: ",
                            std::distance(new_shape.cbegin(), new_dim),
                            " is not compatible.");
        }

        m_shape = std::move(new_shape);
    }

    void* data(const element::Type& element_type) const override {
        auto owner_data = m_owner->data(element_type);
        return static_cast<uint8_t*>(owner_data) + m_offset;
    }

private:
    std::shared_ptr<ITensor> m_owner;
    Shape m_shape;
    const Shape m_capacity;
    const size_t m_offset;
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

namespace util {

ov::Tensor make_tensor(const std::shared_ptr<ITensor>& tensor, const std::shared_ptr<void>& so) {
    return ov::Tensor(tensor, so);
}

void get_tensor_impl(const ov::Tensor& tensor, std::shared_ptr<ITensor>& tensor_impl, std::shared_ptr<void>& so) {
    tensor_impl = tensor._impl;
    so = tensor._so;
}

}  // namespace util

ov::Tensor make_tensor(const ov::SoPtr<ITensor>& tensor) {
    return util::make_tensor(tensor._ptr, tensor._so);
}

ov::SoPtr<ov::ITensor> get_tensor_impl(const ov::Tensor& tensor) {
    std::shared_ptr<ov::ITensor> tensor_impl;
    std::shared_ptr<void> so;
    util::get_tensor_impl(tensor, tensor_impl, so);
    return ov::SoPtr<ov::ITensor>(tensor_impl, so);
}

}  // namespace ov
