// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/make_tensor.hpp"

#include <memory>
#include <mutex>

#include "openvino/core/memory_util.hpp"
#include "openvino/core/type/element_type_info.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#ifdef PROXY_PLUGIN_ENABLED
#    include "openvino/proxy/plugin.hpp"
#endif

namespace ov {

namespace {
size_t ceil_div(size_t value, size_t divisor) {
    OPENVINO_ASSERT(divisor != 0, "Division by zero");
    return (value + divisor - 1) / divisor;
}

Strides make_packed_reference_strides(const Shape& shape, const element::Type& element_type) {
    Strides strides{};

    if (shape.empty()) {
        return strides;
    }

    strides.resize(shape.size());
    if (element_type.bitwidth() >= 8) {
        strides.back() = shape.back() == 0 ? 0 : element_type.size();
        std::copy(shape.rbegin(), shape.rend() - 1, strides.rbegin() + 1);
        std::partial_sum(strides.rbegin(), strides.rend(), strides.rbegin(), std::multiplies<size_t>());
    } else {
        OPENVINO_ASSERT(8 % element_type.bitwidth() == 0,
                        "Unsupported sub-byte element type bitwidth: ",
                        element_type.bitwidth());
        const auto elements_per_byte = static_cast<size_t>(8 / element_type.bitwidth());
        strides.back() = shape.back() == 0 ? 0 : 1;
        for (size_t axis = shape.size() - 1; axis > 0; --axis) {
            const size_t inner_elements =
                std::accumulate(shape.begin() + axis, shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
            strides[axis - 1] = ceil_div(inner_elements, elements_per_byte);
        }
    }

    return strides;
}

size_t calculate_contiguous_linear_offset_in_elements(const Shape& shape, const Coordinate& coord) {
    OPENVINO_ASSERT(shape.size() == coord.size(), "Coordinate rank mismatch");

    size_t linear_offset = 0;
    size_t axis_stride = 1;
    for (size_t axis = shape.size(); axis > 0; --axis) {
        const auto idx = axis - 1;
        linear_offset += coord[idx] * axis_stride;
        axis_stride *= shape[idx];
    }

    return linear_offset;
}

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

void validate_int4_roi(const std::shared_ptr<ITensor>& owner, const Coordinate& begin, const Coordinate& end) {
    if (owner->get_element_type().bitwidth() < 8) {
        const auto elements_per_byte = static_cast<size_t>(8 / owner->get_element_type().bitwidth());
        const auto& owner_shape = owner->get_shape();
        const auto& owner_strides = owner->get_strides();
        const auto reference_strides = make_packed_reference_strides(owner_shape, owner->get_element_type());

        OPENVINO_ASSERT(owner_strides == reference_strides,
                        "Sub-byte ROI is supported only for contiguous packed tensors");

        const auto begin_offset = calculate_contiguous_linear_offset_in_elements(owner_shape, begin);
        const auto end_offset = calculate_contiguous_linear_offset_in_elements(owner_shape, end);

        OPENVINO_ASSERT(begin_offset % elements_per_byte == 0,
                        "Sub-byte ROI begin offset must be aligned to byte boundary");
        OPENVINO_ASSERT(end_offset % elements_per_byte == 0,
                        "Sub-byte ROI end offset must be aligned to byte boundary");
    }
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
        OPENVINO_ASSERT(m_element_type.is_static());
    }

    void* data() override {
        return m_ptr;
    }

    void* data(const element::Type& element_type) override {
        OPENVINO_ASSERT(is_pointer_representable(element_type),
                        "Tensor data with element type ",
                        get_element_type(),
                        ", is not representable as pointer to ",
                        element_type);
        return m_ptr;
    }

    const void* data() const override {
        return m_ptr;
    }

    const void* data(const element::Type& element_type) const override {
        OPENVINO_ASSERT(is_pointer_representable(element_type),
                        "Tensor data with element type ",
                        get_element_type(),
                        ", is not representable as pointer to ",
                        element_type);
        return m_ptr;
    }

    void* data_rw() override {
        return m_ptr;
    }

    void* data_rw(const element::Type& element_type) override {
        OPENVINO_ASSERT(is_pointer_representable(element_type),
                        "Tensor data with element type ",
                        get_element_type(),
                        ", is not representable as pointer to ",
                        element_type);
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
        /*OPENVINO_ASSERT(m_element_type.bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                        m_element_type);*/
        std::call_once(m_strides_once, &ViewTensor::update_strides, this);
        return m_strides;
    }

    std::optional<uint64_t> get_source_id() const {
        return m_source_id;
    }

    void set_source_id(uint64_t id) {
        m_source_id = id;
    }

protected:
    bool is_pointer_representable(const element::Type& element_type) const {
        if (element_type.is_dynamic()) {
            return true;
        } else {
            // gets type info to reduce validation to access speed, due to performance issues
            const auto& other_type_info = element::get_type_info(element_type);
            const auto& this_type_info = element::get_type_info(get_element_type());

            // For sub-byte types, we need special handling
            if (get_element_type().bitwidth() < 8 || element_type.bitwidth() < 8) {
                // Only allow exact type matches for sub-byte types
                return get_element_type() == element_type;
            }

            return (get_element_type() != element::string && element_type != element::string &&
                    other_type_info.m_bitwidth == this_type_info.m_bitwidth &&
                    other_type_info.m_is_real == this_type_info.m_is_real) ||
                   (element_type == element::string && element::string == get_element_type());
        }
    }

    void update_strides() const {
        auto& shape = get_shape();
        if (m_strides.empty() && !shape.empty()) {
            m_strides = make_packed_reference_strides(shape, m_element_type);
        }
    }

    element::Type m_element_type;
    Shape m_shape;
    Shape m_capacity;
    mutable Strides m_strides;
    mutable std::once_flag m_strides_once;
    void* m_ptr;
    std::optional<uint64_t> m_source_id;
};

/**
 * @brief Read-only view tensor to external memory
 * The tensor doesn't own the external memory
 */
class ReadOnlyViewTensor : public ViewTensor {
public:
    ReadOnlyViewTensor(const element::Type element_type, const Shape& shape, const void* ptr)
        : ViewTensor{element_type, shape, const_cast<void*>(ptr)} {}

    using ViewTensor::data;

    [[noreturn]] void* data_rw() override {
        OPENVINO_THROW("Can not access non-const pointer use e.g. 'static_cast<const ov::Tensor&>.data()'");
    }

    [[noreturn]] void* data_rw(const element::Type& element_type) override {
        OPENVINO_THROW("Can not access non-const pointer use e.g. 'static_cast<const ov::Tensor&>.data(element_type)'");
    }
};

/**
 * @brief View tensor on external memory with strides
 */
class StridedViewTensor : public ViewTensor {
public:
    StridedViewTensor(const element::Type element_type, const Shape& shape, void* ptr, const Strides& strides)
        : ViewTensor{element_type, shape, ptr} {
        // Remove the bitwidth restriction for int4 support
        // OPENVINO_ASSERT(
        //     get_element_type().bitwidth() >= 8,
        //     "Could not create strided access tensor for types with bitwidths less then 8 bit. Tensor type: ",
        //     get_element_type());

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

            // For sub-byte types, we need different validation logic
            if (get_element_type().bitwidth() >= 8) {
                OPENVINO_ASSERT((m_strides[i] % get_element_type().size()) == 0,
                                "shape stride: ",
                                shape_strides[i],
                                ", stride: ",
                                m_strides[i]);
            } else {
                // For int4, strides should be byte-aligned
                // Since we're dealing with packed data, ensure stride alignment makes sense
                OPENVINO_ASSERT(m_strides[i] > 0, "Invalid stride for int4 type: ", m_strides[i]);
            }

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

class ReadOnlyStridedViewTensor : public StridedViewTensor {
public:
    ReadOnlyStridedViewTensor(const element::Type element_type,
                              const Shape& shape,
                              const void* ptr,
                              const Strides& strides)
        : StridedViewTensor{element_type, shape, const_cast<void*>(ptr), strides} {}

    using StridedViewTensor::data;

    [[noreturn]] void* data_rw() override {
        OPENVINO_THROW("Can not access non-const pointer use e.g. 'static_cast<const ov::Tensor&>.data()'");
    }

    [[noreturn]] void* data_rw(const element::Type& element_type) override {
        OPENVINO_THROW("Can not access non-const pointer use e.g. 'static_cast<const ov::Tensor&>.data()'");
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
 * @brief Creates read-only view tensor on external memory
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
                                     const void* ptr,
                                     const Strides& byte_strides) {
    if (byte_strides.empty()) {
        return std::make_shared<ReadOnlyViewTensor>(element_type, shape, ptr);
    } else {
        return std::make_shared<ReadOnlyStridedViewTensor>(element_type, shape, ptr, byte_strides);
    }
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
                         const auto byte_size = util::get_memory_size_safe(element_type, shape);
                         OPENVINO_ASSERT(byte_size, bad_alloc_error_msg(element_type, shape));
                         auto data = const_cast<Allocator&>(allocator).allocate(*byte_size);
                         OPENVINO_ASSERT(*byte_size == 0 || data != nullptr, "Failed to allocate memory");
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

        const auto byte_size = util::get_memory_size_safe(m_element_type, new_shape);
        OPENVINO_ASSERT(byte_size, bad_alloc_error_msg(m_element_type, new_shape));
        m_shape = std::move(new_shape);

        if (*byte_size > get_bytes_capacity()) {
            destroy_memory();
            // allocate buffer and initialize objects from scratch
            m_capacity = m_shape;
            m_ptr = m_allocator.allocate(*byte_size);
            initialize_elements(m_ptr, m_element_type, m_shape);
        }

        m_strides.clear();
        update_strides();
    }

private:
    void destroy_elements(size_t begin_ind, size_t end_ind) {
        // it removes elements from tail
        if (m_ptr != nullptr && get_element_type() == element::string) {
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
        return util::get_memory_size(get_element_type(), get_capacity());
    }

    static std::string bad_alloc_error_msg(const element::Type& element_type, const Shape& shape) {
        return "Cannot allocate memory for type: " + element_type.to_string() + " and shape: " + shape.to_string();
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
 * @brief Base class for representing a Region of Interest (ROI) on another tensor
 * ROI tensor holds the owner
 */
class BaseRoiTensor {
public:
    BaseRoiTensor(const std::shared_ptr<ITensor>& owner, const Coordinate& begin, const Coordinate& end)
        : m_owner{owner},
          m_shape{make_roi_shape(owner->get_shape(), begin, end)},
          m_capacity{m_shape},
          m_offset{calculate_roi_offset(owner, begin)} {
        // Remove the bitwidth restriction
        // OPENVINO_ASSERT(m_owner->get_element_type().bitwidth() >= 8,
        //                 "ROI Tensor for types with bitwidths less than 8 bit is not implemented. Tensor type: ",
        //                 m_owner->get_element_type());
    }

    void set_shape(ov::Shape new_shape) {
        OPENVINO_ASSERT(new_shape.size() >= m_shape.size());
        const auto last_new_dim = new_shape.crend();
        auto new_dim = new_shape.crbegin();
        for (auto max_dim = m_capacity.crbegin(); new_dim != last_new_dim && max_dim != m_capacity.crend();
             ++max_dim, ++new_dim) {
            OPENVINO_ASSERT(*new_dim <= *max_dim,
                            "Cannot set new shape: ",
                            new_shape,
                            " for ROI tensor! New dimension at index: ",
                            std::distance(new_shape.cbegin(), new_dim.base()) - 1,
                            " is not compatible.");
        }
        new_dim = std::find_if(new_dim, last_new_dim, [](auto&& dim) {
            return dim != 1;
        });
        OPENVINO_ASSERT(
            new_dim == last_new_dim,
            "Cannot set new shape: ",
            new_shape,
            " for ROI tensor! The expanding rank dimension(s) of ROI must be ones, but it is not at index: ",
            std::distance(new_shape.cbegin(), new_dim.base()) - 1);

        m_shape = std::move(new_shape);
    }

    size_t get_offset() const {
        return m_offset;
    }

private:
    static size_t calculate_roi_offset(const std::shared_ptr<ITensor>& owner, const Coordinate& begin) {
        if (owner->get_element_type().bitwidth() < 8) {
            const auto elements_per_byte = static_cast<size_t>(8 / owner->get_element_type().bitwidth());
            const auto linear_offset = calculate_contiguous_linear_offset_in_elements(owner->get_shape(), begin);
            OPENVINO_ASSERT(linear_offset % elements_per_byte == 0,
                            "Sub-byte ROI begin offset must be aligned to byte boundary");
            return linear_offset / elements_per_byte;
        } else {
            // Original calculation for byte-aligned types
            return std::inner_product(begin.begin(), begin.end(), owner->get_strides().begin(), static_cast<size_t>(0));
        }
    }

protected:
    std::shared_ptr<ITensor> m_owner;
    Shape m_shape;
    const Shape m_capacity;
    const size_t m_offset;
};

/**
 * @brief Tensor representing a Region of Interest (ROI) on another host tensor
 * ROI tensor holds the owner
 */
class RoiTensor : public BaseRoiTensor, public ITensor {
public:
    RoiTensor(const std::shared_ptr<ITensor>& owner, const Coordinate& begin, const Coordinate& end)
        : BaseRoiTensor(owner, begin, end) {
        validate_int4_roi(owner, begin, end);
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
        BaseRoiTensor::set_shape(new_shape);
    }

    void* data() override {
        return static_cast<uint8_t*>(m_owner->data()) + m_offset;
    }

    void* data(const element::Type& element_type) override {
        return static_cast<uint8_t*>(m_owner->data()) + m_offset;
    }

    const void* data() const override {
        return static_cast<uint8_t*>(m_owner->data()) + m_offset;
    }

    const void* data(const element::Type& element_type) const override {
        return static_cast<uint8_t*>(m_owner->data()) + m_offset;
    }

    void* data_rw() override {
        return static_cast<uint8_t*>(m_owner->data_rw()) + m_offset;
    }

    void* data_rw(const element::Type& element_type) override {
        return static_cast<uint8_t*>(m_owner->data_rw(element_type)) + m_offset;
    }
};

/**
 * @brief Tensor representing a Region of Interest (ROI) on another device tensor
 * ROI tensor holds the owner
 */
class RoiRemoteTensor : public BaseRoiTensor, public IRemoteTensor {
public:
    RoiRemoteTensor(const std::shared_ptr<ITensor>& owner, const Coordinate& begin, const Coordinate& end)
        : BaseRoiTensor(owner, begin, end) {
        validate_int4_roi(owner, begin, end);
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
        BaseRoiTensor::set_shape(new_shape);
    }

    void copy_to(const std::shared_ptr<ov::ITensor>& dst) const override {
        auto owner_remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(m_owner);

        if (std::dynamic_pointer_cast<RoiRemoteTensor>(dst)) {
            OPENVINO_ASSERT(get_shape() == dst->get_shape(),
                            "Cannot copy to RoiRemoteTensor. Shapes are not equal. (src: ",
                            get_shape(),
                            " != dst: ",
                            dst->get_shape(),
                            ")");

            auto dst_roi_remote_tensor = std::dynamic_pointer_cast<RoiRemoteTensor>(dst);
            owner_remote_tensor->copy_to(dst_roi_remote_tensor->m_owner,
                                         m_offset,
                                         dst_roi_remote_tensor->m_offset,
                                         m_shape);
        } else {
            owner_remote_tensor->copy_to(dst, m_offset, 0, m_shape);
        }
    };

    void copy_from(const std::shared_ptr<const ov::ITensor>& src) override {
        auto owner_remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(m_owner);

        OPENVINO_ASSERT(src->get_shape() == get_shape(),
                        "Cannot copy to RoiRemoteTensor. Shapes are not equal. (src: ",
                        src->get_shape(),
                        " != dst: ",
                        get_shape(),
                        ")");

        if (std::dynamic_pointer_cast<const RoiRemoteTensor>(src)) {
            const auto src_roi_remote_tensor = std::dynamic_pointer_cast<const RoiRemoteTensor>(src);
            owner_remote_tensor->copy_from(src_roi_remote_tensor->m_owner,
                                           src_roi_remote_tensor->m_offset,
                                           m_offset,
                                           m_shape);
        } else {
            owner_remote_tensor->copy_from(src, 0, m_offset, m_shape);
        }
    };

    const AnyMap& get_properties() const override {
        auto remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(m_owner);
        return remote_tensor->get_properties();
    };

    const std::string& get_device_name() const override {
        auto remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(m_owner);
        return remote_tensor->get_device_name();
    }
};

/**
 * @brief Creates ROI tensor
 * It determines whether the tensor is remote tensor or regular tensor and returns the appropriate ROI tensor type
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
    if (std::dynamic_pointer_cast<IRemoteTensor>(other)) {
        return std::make_shared<RoiRemoteTensor>(other, begin, end);
    } else {
        return std::make_shared<RoiTensor>(other, begin, end);
    }
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

size_t get_tensor_data_offset(const ov::ITensor& tensor) {
    if (auto tensor_impl = dynamic_cast<const BaseRoiTensor*>(&tensor)) {
        return tensor_impl->get_offset();
    }
    return 0;
}

std::optional<uint64_t> get_tensor_source_id(const ov::Tensor& tensor) {
    if (auto itensor = std::dynamic_pointer_cast<ViewTensor>(get_tensor_impl(tensor)._ptr)) {
        return itensor->get_source_id();
    }
    return std::nullopt;
}

void set_tensor_source_id(ov::Tensor& tensor, uint64_t id) {
    if (auto itensor = std::dynamic_pointer_cast<ViewTensor>(get_tensor_impl(tensor)._ptr)) {
        itensor->set_source_id(id);
    }
}

}  // namespace ov
