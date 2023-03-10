// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/itensor.hpp"

#include "ie_blob.h"
#include "ie_ngraph_utils.hpp"
#include "ie_remote_blob.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {

size_t ITensor::get_size() const {
    return shape_size(get_shape());
}

size_t ITensor::get_byte_size() const {
    return (get_size() * get_element_type().bitwidth() + 8 - 1) / 8;
}

AnyMap ITensor::get_properties() const {
    return {};
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
          m_ptr{ptr} {
        OPENVINO_ASSERT(m_ptr != nullptr);
        m_offsets = {m_shape.size(), 0};
        update_strides();
    }

    void* data(const element::Type& element_type) const override {
        if (element_type != element::undefined) {
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
        auto old_byte_size = get_byte_size();
        OPENVINO_ASSERT(shape_size(new_shape) * get_element_type().size() <= old_byte_size,
                        "Could set new shape: ",
                        new_shape);
        m_shape = std::move(new_shape);
        m_offsets = {m_shape.size(), 0};
        update_strides();
    }

    const Coordinate& get_offsets() const override {
        return m_offsets;
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
        Strides strides;
        if (!shape.empty()) {
            strides.resize(shape.size());
            strides.back() = m_element_type.size();
            std::copy(shape.rbegin(), shape.rend() - 1, strides.rbegin() + 1);
            std::partial_sum(strides.rbegin(), strides.rend(), strides.rbegin(), std::multiplies<size_t>());
        }
        m_strides = strides;
    }

    element::Type m_element_type;
    Shape m_shape;
    Coordinate m_offsets;
    Strides m_strides;
    void* m_ptr;
};

/**
 * @brief View tensor on external memory with strides
 */
class StridedViewTensor : public ViewTensor {
public:
    StridedViewTensor(const element::Type element_type, const Shape& shape, void* ptr, const Strides& strides)
        : ViewTensor{element_type, shape, ptr},
          m_strides{strides} {
        OPENVINO_ASSERT(
            get_element_type().bitwidth() >= 8,
            "Could not create strided access tensor for types with bitwidths less then 8 bit. Tensor type: ",
            get_element_type());
        OPENVINO_ASSERT(get_shape().size() == m_strides.size());
        auto& shape_strides = ViewTensor::get_strides();
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
        }
    }

    const Strides& get_strides() const override {
        return m_strides;
    }

private:
    Strides m_strides;
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
    RoiTensor(const std::shared_ptr<ITensor>& owner, const Coordinate& begin, const Coordinate& end)
        : m_owner{owner},
          m_offsets{begin} {
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
    }

    const element::Type& get_element_type() const override {
        return m_owner->get_element_type();
    }

    const Coordinate& get_offsets() const override {
        return m_offsets;
    }

    const Strides& get_strides() const override {
        return m_owner->get_strides();
    }

    const Shape& get_shape() const override {
        return m_shape;
    }

    void set_shape(ov::Shape new_shape) override {
        OPENVINO_UNREACHABLE("Shapes cannot be changed for ROI Tensor");
    }

    void* data(const element::Type& element_type) const override {
        auto owner_data = m_owner->data(element_type);
        auto& strides = get_strides();
        size_t byte_offset = std::inner_product(m_offsets.begin(), m_offsets.end(), strides.begin(), 0);
        return static_cast<uint8_t*>(owner_data) + byte_offset;
    }

private:
    std::shared_ptr<ITensor> m_owner;
    Coordinate m_offsets;
    Shape m_shape;
};

/**
 * @brief Creates ROI tensor
 *
 * @param other Tensor what owns the memory
 * @param begin Begin coordinates
 * @param end End coordinates
 *
 * @return
 */
std::shared_ptr<ITensor> make_tensor(const std::shared_ptr<ITensor>& other,
                                     const Coordinate& begin,
                                     const Coordinate& end) {
    return std::make_shared<RoiTensor>(other, begin, end);
}

/**
 * @brief Tensor what contains InferenceEngine::Blob inside
 * Blob owns the memory
 */
class BlobTensor : public ITensor {
    mutable element::Type m_type;
    mutable Shape m_shape;
    mutable Strides m_strides;
    Coordinate m_offsets;

public:
    std::shared_ptr<ie::Blob> blob;

    BlobTensor(const InferenceEngine::Blob::Ptr& blob) : blob{blob} {
        m_shape = blob->getTensorDesc().getBlockingDesc().getBlockDims();
        m_offsets = {m_shape.size(), 0};
    }

    const element::Type& get_element_type() const override {
        m_type = InferenceEngine::details::convertPrecision(blob->getTensorDesc().getPrecision());
        return m_type;
    }

    void set_shape(ov::Shape shape) override {
        blob->setShape({shape.begin(), shape.end()});
        m_offsets = {shape.size(), 0};
    }

    const Shape& get_shape() const override {
        m_shape = blob->getTensorDesc().getBlockingDesc().getBlockDims();
        return m_shape;
    }

    const Coordinate& get_offsets() const override {
        return m_offsets;
    }

    const Strides& get_strides() const override {
        OPENVINO_ASSERT(get_element_type().bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                        get_element_type());
        const auto& element_strides = blob->getTensorDesc().getBlockingDesc().getStrides();
        const size_t elem_size = get_element_type().size();
        m_strides.clear();
        m_strides.resize(element_strides.size());
        std::transform(element_strides.begin(), element_strides.end(), m_strides.begin(), [&elem_size](size_t stride) {
            return stride * elem_size;
        });
        return m_strides;
    }

    size_t get_size() const override {
        return blob->size();
    }

    size_t get_byte_size() const override {
        return blob->byteSize();
    }

    void* data(const element::Type& element_type) const override {
        OPENVINO_ASSERT(blob != nullptr, "Tensor was not initialized.");
#define TYPE_CHECK(TYPE) (dynamic_cast<const ie::TBlob<TYPE>*>(blob.get()) != nullptr)
        auto host_accesable_implementation = TYPE_CHECK(bool) || TYPE_CHECK(int8_t) || TYPE_CHECK(uint8_t) ||
                                             TYPE_CHECK(int16_t) || TYPE_CHECK(uint16_t) || TYPE_CHECK(int32_t) ||
                                             TYPE_CHECK(uint32_t) || TYPE_CHECK(int64_t) || TYPE_CHECK(uint64_t) ||
                                             TYPE_CHECK(float) || TYPE_CHECK(double);
#undef TYPE_CHECK
        OPENVINO_ASSERT(host_accesable_implementation,
                        "Tensor implementation type dose not contains host accessable data");
        if (element_type != element::undefined) {
            OPENVINO_ASSERT(element_type == get_element_type(),
                            "Tensor data with element type ",
                            get_element_type(),
                            ", is not representable as pointer to ",
                            element_type);
        }
        // since we don't use byte offsets, we need to explicitly multiply by element_size
        auto byte_offset = blob->getTensorDesc().getBlockingDesc().getOffsetPadding() * get_element_type().size();
        OPENVINO_ASSERT((get_element_type().bitwidth() >= 8) || (byte_offset == 0),
                        "ROI access for types with bitwidths less then 8 bit is not implemented. Tensor type: ",
                        get_element_type());
        return byte_offset + InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap().as<uint8_t*>();
    }

    AnyMap get_properties() const override {
        auto remote_impl = dynamic_cast<InferenceEngine::RemoteBlob*>(blob.get());
        if (remote_impl != nullptr) {
            auto params = remote_impl->getParams();
            params.insert(ov::device::id(remote_impl->getDeviceName()));
            return params;
        } else {
            return {};
        }
    }
};

/**
 * @brief Create InferenceEngine::RemoteBlob from the Tensor
 */
class TensorRemoteBlob : public ie::RemoteBlob {
public:
    TensorRemoteBlob(const std::shared_ptr<ITensor>& tensor)
        : ie::RemoteBlob{ie::TensorDesc{ie::details::convertPrecision(tensor->get_element_type()),
                                        tensor->get_shape(),
                                        ie::TensorDesc::getLayoutByRank(tensor->get_shape().size())}},
          tensor{tensor} {}
    AnyMap getParams() const override {
        return tensor->get_properties();
    }
    std::string getDeviceName() const noexcept override {
        try {
            return tensor->get_properties().at(ov::device::id.name()).as<std::string>();
        } catch (...) {
            return {};
        }
    }
    std::shared_ptr<ie::RemoteContext> getContext() const noexcept override {
        return {};
    }

    void allocate() noexcept override {}
    bool deallocate() noexcept override {
        return true;
    }
    ie::LockedMemory<void> buffer() noexcept override {
        return {nullptr, nullptr, 0};
    }
    ie::LockedMemory<const void> cbuffer() const noexcept override {
        return {nullptr, nullptr, 0};
    }
    ie::LockedMemory<void> rwmap() noexcept override {
        return {nullptr, nullptr, 0};
    }
    ie::LockedMemory<const void> rmap() const noexcept override {
        return {nullptr, nullptr, 0};
    }
    ie::LockedMemory<void> wmap() noexcept override {
        return {nullptr, nullptr, 0};
    }
    const std::shared_ptr<ie::IAllocator>& getAllocator() const noexcept override {
        return m_allocator;
    }
    void* getHandle() const noexcept override {
        return nullptr;
    }

    std::shared_ptr<ITensor> tensor;

private:
    std::shared_ptr<ie::IAllocator> m_allocator;
};

/**
 * @brief Create InferenceEngine::TBlob<T> from the tensor
 *
 * @tparam T Blob data type
 */
template <typename T>
class TensorMemoryBlob : public ie::TBlob<T> {
public:
    ~TensorMemoryBlob() override = default;
    explicit TensorMemoryBlob(const std::shared_ptr<ITensor>& tensor_) try : ie
        ::TBlob<T>{[&] {
                       auto element_type = tensor_->get_element_type();
                       auto shape = tensor_->get_shape();
                       ie::SizeVector blk_order(shape.size());
                       std::iota(blk_order.begin(), blk_order.end(), 0);
                       ie::SizeVector dim_offset(shape.size(), 0);
                       ie::SizeVector blk_strides;
                       auto byte_strides = element_type.bitwidth() >= 8 ? tensor_->get_strides() : Strides{};
                       if (byte_strides.empty()) {
                           blk_strides = ov::row_major_strides(shape);
                       } else {
                           blk_strides.resize(byte_strides.size());
                           std::transform(byte_strides.begin(),
                                          byte_strides.end(),
                                          blk_strides.begin(),
                                          [&element_type](size_t byte_stride) {
                                              OPENVINO_ASSERT(byte_stride % element_type.size() == 0,
                                                              "Limitation: Stride in bytes ",
                                                              byte_stride,
                                                              " should be divisible by size of element ",
                                                              element_type.size());
                                              return byte_stride / element_type.size();
                                          });
                       }
                       return ie::TensorDesc{ie::details::convertPrecision(element_type),
                                             shape,
                                             ie::BlockingDesc{shape, blk_order, 0, dim_offset, blk_strides}};
                   }(),
                   static_cast<T*>(tensor_->data()),
                   tensor_->get_byte_size()},
            tensor{tensor_} {}
    catch (const std::exception& ex) {
        throw ov::Exception(ex.what());
    }

    void setShape(const ie::SizeVector& dims) override {
        tensor->set_shape(dims);
        ie::TBlob<T>::setShape(dims);
    }

    std::shared_ptr<ITensor> tensor;
};

std::shared_ptr<ITensor> make_tensor(const std::shared_ptr<ie::Blob>& blob) {
#define ELSE_IF(type)                                                                \
    else if (auto tblob = dynamic_cast<const TensorMemoryBlob<type>*>(blob.get())) { \
        return tblob->tensor;                                                        \
    }
    if (blob == nullptr) {
        return {};
    } else if (auto tblob = dynamic_cast<const TensorRemoteBlob*>(blob.get())) {
        return tblob->tensor;
    }
    ELSE_IF(float)
    ELSE_IF(double)
    ELSE_IF(int8_t)
    ELSE_IF(int8_t)
    ELSE_IF(int16_t)
    ELSE_IF(int32_t)
    ELSE_IF(int64_t)
    ELSE_IF(uint8_t)
    ELSE_IF(uint8_t)
    ELSE_IF(uint16_t)
    ELSE_IF(uint32_t)
    ELSE_IF(uint64_t)
    ELSE_IF(int8_t)
    ELSE_IF(bool) else {
        return std::make_shared<BlobTensor>(blob);
    }
#undef IF
}

ie::Blob::Ptr tensor_to_blob(const std::shared_ptr<ITensor>& tensor) {
    if (tensor == nullptr) {
        return {};
    } else if (auto blob_tensor = dynamic_cast<const BlobTensor*>(tensor.get())) {
        return blob_tensor->blob;
    } else if (tensor->get_properties().empty()) {
#define CASE(precision, T)   \
    case element::precision: \
        return std::make_shared<TensorMemoryBlob<T>>(tensor);
        switch (tensor->get_element_type()) {
            CASE(f32, float);
            CASE(f64, double);
            CASE(i4, int8_t);
            CASE(i8, int8_t);
            CASE(i16, int16_t);
            CASE(i32, int32_t);
            CASE(i64, int64_t);
            CASE(u4, uint8_t);
            CASE(u8, uint8_t);
            CASE(u16, uint16_t);
            CASE(u32, uint32_t);
            CASE(u64, uint64_t);
            CASE(u1, int8_t);
            CASE(boolean, bool);
        case element::f16:
            return std::make_shared<TensorMemoryBlob<int16_t>>(tensor);
        case element::bf16:
            return std::make_shared<TensorMemoryBlob<int16_t>>(tensor);
        default:
            OPENVINO_UNREACHABLE("Unsupported element type");
        }
#undef CASE
    } else {
        return std::make_shared<TensorRemoteBlob>(tensor);
    }
}
}  // namespace ov
