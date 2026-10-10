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
        OPENVINO_ASSERT(m_element_type.bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                        m_element_type);
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
            return (get_element_type() != element::string && element_type != element::string &&
                    other_type_info.m_bitwidth == this_type_info.m_bitwidth &&
                    other_type_info.m_is_real == this_type_info.m_is_real) ||
                   (element_type == element::string && element::string == get_element_type());
        }
    }

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
    using MemSpace = std::pair<void*, size_t>;

    static MemSpace do_allocate(const element::Type& element_type, const Shape& shape, const Allocator& allocator) {
        OPENVINO_ASSERT(allocator, "Allocator was not initialized");
        const auto byte_size = util::get_memory_size_safe(element_type, shape);
        OPENVINO_ASSERT(byte_size, bad_alloc_error_msg(element_type, shape));
        auto data = const_cast<Allocator&>(allocator).allocate(*byte_size);
        OPENVINO_ASSERT(*byte_size == 0 || data != nullptr, "Failed to allocate memory");
        initialize_elements(data, element_type, shape);
        return {data, *byte_size};
    }

    AllocatedTensor(const element::Type element_type, const Shape& shape, const Allocator& allocator, MemSpace alloc)
        : ViewTensor{element_type, shape, alloc.first},
          m_allocator{allocator},
          m_bytes_capacity{alloc.second} {}

public:
    AllocatedTensor(const element::Type element_type, const Shape& shape, const Allocator& allocator)
        : AllocatedTensor{element_type, shape, allocator, do_allocate(element_type, shape, allocator)} {}

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
            m_bytes_capacity = *byte_size;
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
        return m_bytes_capacity;
    }

    static std::string bad_alloc_error_msg(const element::Type& element_type, const Shape& shape) {
        return "Cannot allocate memory for type: " + element_type.to_string() + " and shape: " + shape.to_string();
    }

    Allocator m_allocator;
    size_t m_bytes_capacity;
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
          m_begin{begin} {
        OPENVINO_ASSERT(m_owner->get_element_type().bitwidth() >= 8,
                        "ROI Tensor for types with bitwidths less than 8 bit is not implemented. Tensor type: ",
                        m_owner->get_element_type());
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

        const auto& owner_strides = m_owner->get_strides();
        calculate_offset(new_shape, owner_strides);
        m_shape = std::move(new_shape);
        update_padded_strides(owner_strides);
    }

    size_t get_offset() const {
        return calculate_offset(m_shape, m_owner->get_strides());
    }

    const Strides& get_strides() const {
        const auto& owner_strides = m_owner->get_strides();
        calculate_offset(m_shape, owner_strides);
        if (m_shape.size() == owner_strides.size()) {
            return owner_strides;
        }

        std::lock_guard<std::mutex> lock{m_strides_mutex};
        if (m_padded_strides.size() != m_shape.size() || m_owner_strides_rank != owner_strides.size() ||
            !std::equal(owner_strides.rbegin(),
                        owner_strides.rbegin() + std::min(owner_strides.size(), m_shape.size()),
                        m_padded_strides.rbegin())) {
            update_padded_strides(owner_strides);
        }
        return m_padded_strides;
    }

protected:
    size_t calculate_offset(const Shape& shape, const Strides& owner_strides) const {
        const auto& owner_shape = m_owner->get_shape();
        OPENVINO_ASSERT(owner_strides.size() == owner_shape.size(), "Owner tensor strides rank must match shape rank.");

        size_t offset = 0;
        // Align coordinates from the trailing dimensions, as in set_shape().
        for (size_t i = 0; i < std::max(shape.size(), owner_shape.size()); ++i) {
            const auto begin = i < m_begin.size() ? m_begin[m_begin.size() - 1 - i] : 0;
            const auto dim = i < shape.size() ? shape[shape.size() - 1 - i] : 1;
            const auto owner_dim = i < owner_shape.size() ? owner_shape[owner_shape.size() - 1 - i] : 1;
            OPENVINO_ASSERT(begin <= owner_dim && dim <= owner_dim - begin,
                            "ROI tensor with shape ",
                            shape,
                            " and begin coordinates ",
                            m_begin,
                            " is outside owner shape ",
                            owner_shape);
            if (i < owner_strides.size()) {
                offset += begin * owner_strides[owner_strides.size() - 1 - i];
            }
        }
        return offset;
    }

    void update_padded_strides(const Strides& owner_strides) const {
        m_owner_strides_rank = owner_strides.size();
        if (m_shape.size() == owner_strides.size() || m_shape.empty()) {
            m_padded_strides.clear();
            return;
        }
        m_padded_strides = m_owner->get_strides_for_shape(m_shape);
    }

    std::shared_ptr<ITensor> m_owner;
    Shape m_shape;
    const Shape m_capacity;
    const Coordinate m_begin;
    mutable Strides m_padded_strides;
    mutable size_t m_owner_strides_rank = 0;
    mutable std::mutex m_strides_mutex;
};

/**
 * @brief Tensor representing a Region of Interest (ROI) on another host tensor
 * ROI tensor holds the owner
 */
class RoiTensor : public BaseRoiTensor, public ITensor {
public:
    RoiTensor(const std::shared_ptr<ITensor>& owner, const Coordinate& begin, const Coordinate& end)
        : BaseRoiTensor(owner, begin, end) {}

    const element::Type& get_element_type() const override {
        return m_owner->get_element_type();
    }

    const Strides& get_strides() const override {
        return BaseRoiTensor::get_strides();
    }

    const Shape& get_shape() const override {
        return m_shape;
    }

    void set_shape(ov::Shape new_shape) override {
        BaseRoiTensor::set_shape(new_shape);
    }

    void* data() override {
        return static_cast<uint8_t*>(m_owner->data()) + get_offset();
    }

    void* data(const element::Type& element_type) override {
        return static_cast<uint8_t*>(m_owner->data()) + get_offset();
    }

    const void* data() const override {
        return static_cast<uint8_t*>(m_owner->data()) + get_offset();
    }

    const void* data(const element::Type& element_type) const override {
        return static_cast<uint8_t*>(m_owner->data()) + get_offset();
    }

    void* data_rw() override {
        return static_cast<uint8_t*>(m_owner->data_rw()) + get_offset();
    }

    void* data_rw(const element::Type& element_type) override {
        return static_cast<uint8_t*>(m_owner->data_rw(element_type)) + get_offset();
    }
};

/**
 * @brief Tensor representing a Region of Interest (ROI) on another device tensor
 * ROI tensor holds the owner
 */
class RoiRemoteTensor : public BaseRoiTensor, public IRemoteTensor {
public:
    RoiRemoteTensor(const std::shared_ptr<ITensor>& owner, const Coordinate& begin, const Coordinate& end)
        : BaseRoiTensor(owner, begin, end) {}

    const element::Type& get_element_type() const override {
        return m_owner->get_element_type();
    }

    const Strides& get_strides() const override {
        return BaseRoiTensor::get_strides();
    }

    const Shape& get_shape() const override {
        return m_shape;
    }

    void set_shape(ov::Shape new_shape) override {
        BaseRoiTensor::set_shape(new_shape);
    }

    void copy_to(const std::shared_ptr<ov::ITensor>& dst) const override {
        OPENVINO_ASSERT(dst, "Destination tensor was not initialized.");
        const auto [owner_remote_tensor, offset] = get_owner_and_offset();

        if (std::dynamic_pointer_cast<RoiRemoteTensor>(dst)) {
            OPENVINO_ASSERT(get_shape() == dst->get_shape(),
                            "Cannot copy to RoiRemoteTensor. Shapes are not equal. (src: ",
                            get_shape(),
                            " != dst: ",
                            dst->get_shape(),
                            ")");

            auto dst_roi_remote_tensor = std::dynamic_pointer_cast<RoiRemoteTensor>(dst);
            const auto [dst_owner, dst_offset] = dst_roi_remote_tensor->get_owner_and_offset();
            owner_remote_tensor->copy_to(dst_owner, offset, dst_offset, m_shape);
        } else {
            owner_remote_tensor->copy_to(dst, offset, 0, m_shape);
        }
    };

    void copy_from(const std::shared_ptr<const ov::ITensor>& src) override {
        OPENVINO_ASSERT(src, "Source tensor was not initialized.");
        const auto [owner_remote_tensor, offset] = get_owner_and_offset();

        OPENVINO_ASSERT(src->get_shape() == get_shape(),
                        "Cannot copy to RoiRemoteTensor. Shapes are not equal. (src: ",
                        src->get_shape(),
                        " != dst: ",
                        get_shape(),
                        ")");

        if (std::dynamic_pointer_cast<const RoiRemoteTensor>(src)) {
            const auto src_roi_remote_tensor = std::dynamic_pointer_cast<const RoiRemoteTensor>(src);
            const auto [src_owner, src_offset] = src_roi_remote_tensor->get_owner_and_offset();
            owner_remote_tensor->copy_from(src_owner, src_offset, offset, m_shape);
        } else {
            owner_remote_tensor->copy_from(src, 0, offset, m_shape);
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

private:
    std::pair<std::shared_ptr<IRemoteTensor>, size_t> get_owner_and_offset() const {
        auto owner = m_owner;
        auto offset = get_offset();
        while (auto roi = std::dynamic_pointer_cast<RoiRemoteTensor>(owner)) {
            offset += roi->get_offset();
            owner = roi->m_owner;
        }
        return {std::dynamic_pointer_cast<IRemoteTensor>(owner), offset};
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
