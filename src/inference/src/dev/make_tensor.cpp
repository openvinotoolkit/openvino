// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/make_tensor.hpp"

#include <memory>

#include "ie_blob.h"
#include "ie_ngraph_utils.hpp"
#include "ie_remote_blob.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/properties.hpp"
#ifdef PROXY_PLUGIN_ENABLED
#    include "openvino/proxy/plugin.hpp"
#endif

namespace ov {

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
        OPENVINO_ASSERT(m_ptr != nullptr);
        OPENVINO_ASSERT(m_element_type != element::undefined && m_element_type.is_static());
    }

    void* data(const element::Type& element_type) const override {
        if (element_type != element::undefined && element_type != element::dynamic &&
            (element_type.bitwidth() != get_element_type().bitwidth() ||
             element_type.is_real() != get_element_type().is_real())) {
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
            m_strides.back() = m_element_type.size();
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
        m_strides.clear();
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

/**
 * @brief Tensor what contains InferenceEngine::Blob inside
 * Blob owns the memory
 */
class BlobTensor : public ITensor {
    mutable element::Type m_type;
    mutable Shape m_shape;
    mutable Strides m_strides;

    void update_strides() {
        if (get_element_type().bitwidth() >= 8) {
            const auto& element_strides = blob->getTensorDesc().getBlockingDesc().getStrides();
            const size_t elem_size = get_element_type().size();
            m_strides.clear();
            m_strides.resize(element_strides.size());
            std::transform(element_strides.begin(),
                           element_strides.end(),
                           m_strides.begin(),
                           [&elem_size](size_t stride) {
                               return stride * elem_size;
                           });
        }
    }

public:
    std::shared_ptr<ie::Blob> blob;

    BlobTensor(const InferenceEngine::Blob::Ptr& blob) : blob{blob} {
        auto remote_impl = dynamic_cast<InferenceEngine::RemoteBlob*>(blob.get());
        OPENVINO_ASSERT(!remote_impl);
        OPENVINO_ASSERT(blob);
        m_shape = blob->getTensorDesc().getBlockingDesc().getBlockDims();
        update_strides();
    }

    const element::Type& get_element_type() const override {
        m_type = InferenceEngine::details::convertPrecision(blob->getTensorDesc().getPrecision());
        return m_type;
    }

    void set_shape(ov::Shape shape) override {
        blob->setShape({shape.begin(), shape.end()});
        update_strides();
    }

    const Shape& get_shape() const override {
        m_shape = blob->getTensorDesc().getBlockingDesc().getBlockDims();
        return m_shape;
    }

    const Strides& get_strides() const override {
        OPENVINO_ASSERT(get_element_type().bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                        get_element_type());
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
        if (element_type != element::undefined && element_type.is_static()) {
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
};

/**
 * @brief Tensor what contains InferenceEngine::RemoteBlob inside
 * Blob owns the memory
 */
class RemoteBlobTensor : public IRemoteTensor {
    mutable element::Type m_type;
    mutable Shape m_shape;
    mutable Strides m_strides;
    mutable ov::AnyMap m_properties;
    mutable std::string m_dev_name;

public:
    std::shared_ptr<ie::RemoteBlob> blob;

    RemoteBlobTensor(const InferenceEngine::RemoteBlob::Ptr& blob) : blob{blob} {
        OPENVINO_ASSERT(blob);
        m_shape = blob->getTensorDesc().getBlockingDesc().getBlockDims();
    }

    const element::Type& get_element_type() const override {
        m_type = InferenceEngine::details::convertPrecision(blob->getTensorDesc().getPrecision());
        return m_type;
    }

    void set_shape(ov::Shape shape) override {
        blob->setShape({shape.begin(), shape.end()});
    }

    const Shape& get_shape() const override {
        m_shape = blob->getTensorDesc().getBlockingDesc().getBlockDims();
        return m_shape;
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

    const AnyMap& get_properties() const override {
        m_properties = blob->getParams();
        return m_properties;
    }

    const std::string& get_device_name() const override {
        m_dev_name = blob->getDeviceName();
        return m_dev_name;
    }
};

/**
 * @brief Create InferenceEngine::RemoteBlob from the Tensor
 */
class TensorRemoteBlob : public ie::RemoteBlob {
public:
    TensorRemoteBlob(const ov::SoPtr<ITensor>& tensor, ie::TensorDesc desc) : ie::RemoteBlob{desc}, tensor{tensor} {
        OPENVINO_ASSERT(this->tensor);
    }
    std::shared_ptr<ov::IRemoteTensor> cast_tensor() const {
        auto remote = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr);
        OPENVINO_ASSERT(remote);
        return remote;
    }
    AnyMap getParams() const override {
        return cast_tensor()->get_properties();
    }
    std::string getDeviceName() const noexcept override {
        try {
            return cast_tensor()->get_device_name();
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

    ov::SoPtr<ITensor> tensor;

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
    explicit TensorMemoryBlob(const ov::SoPtr<ITensor>& tensor_, ie::TensorDesc desc) try : ie
        ::TBlob<T>{desc, static_cast<T*>(tensor_->data()), tensor_->get_byte_size()}, tensor{tensor_} {
            OPENVINO_ASSERT(!std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr));
        }
    catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    }

    void setShape(const ie::SizeVector& dims) override {
        tensor->set_shape(dims);
        ie::TBlob<T>::getTensorDesc().setDims(dims);
        allocate();
    }

    void allocate() noexcept override {
        if (ie::TBlob<T>::buffer() != tensor->data()) {
            ie::TBlob<T>::_allocator =
                ie::details::make_pre_allocator(static_cast<T*>(tensor->data()), tensor->get_byte_size());
            ie::TBlob<T>::allocate();
        }
    }

    ov::SoPtr<ITensor> tensor;
};

ov::SoPtr<ITensor> make_tensor(const std::shared_ptr<ie::Blob>& blob) {
#define ELSE_IF(type)                                                                \
    else if (auto tblob = dynamic_cast<const TensorMemoryBlob<type>*>(blob.get())) { \
        return tblob->tensor;                                                        \
    }
    if (blob == nullptr) {
        return {};
    } else if (auto remote_blob = std::dynamic_pointer_cast<TensorRemoteBlob>(blob)) {
        return remote_blob->tensor;
    } else if (auto remote_blob = std::dynamic_pointer_cast<InferenceEngine::RemoteBlob>(blob)) {
        return {std::make_shared<RemoteBlobTensor>(remote_blob), nullptr};
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
        return {std::make_shared<BlobTensor>(blob), nullptr};
    }
#undef IF
}

ie::Blob* get_hardware_blob(ie::Blob* blob) {
#ifdef PROXY_PLUGIN_ENABLED
    if (auto remote_blob = dynamic_cast<TensorRemoteBlob*>(blob)) {
        const auto& tensor = ov::proxy::get_hardware_tensor(remote_blob->tensor);
        if (auto blob_tensor = std::dynamic_pointer_cast<BlobTensor>(tensor._ptr)) {
            return blob_tensor->blob.get();
        } else if (auto blob_tensor = std::dynamic_pointer_cast<RemoteBlobTensor>(tensor._ptr)) {
            return blob_tensor->blob.get();
        }
        OPENVINO_NOT_IMPLEMENTED;
    }
#endif
    return blob;
}

const ie::Blob* get_hardware_blob(const ie::Blob* blob) {
#ifdef PROXY_PLUGIN_ENABLED
    if (auto remote_blob = dynamic_cast<const TensorRemoteBlob*>(blob)) {
        const auto& tensor = ov::proxy::get_hardware_tensor(remote_blob->tensor);
        if (auto blob_tensor = std::dynamic_pointer_cast<BlobTensor>(tensor._ptr)) {
            return blob_tensor->blob.get();
        } else if (auto blob_tensor = std::dynamic_pointer_cast<RemoteBlobTensor>(tensor._ptr)) {
            return blob_tensor->blob.get();
        }
        OPENVINO_NOT_IMPLEMENTED;
    }
#endif
    return blob;
}

ie::Blob::Ptr tensor_to_blob(const ov::SoPtr<ITensor>& orig_tensor, bool unwrap, InferenceEngine::TensorDesc desc) {
    auto create_desc = [](const ov::SoPtr<ov::ITensor>& tensor,
                          const InferenceEngine::TensorDesc& desc) -> InferenceEngine::TensorDesc {
        if (desc.getLayout() != InferenceEngine::ANY ||
            desc.getPrecision() != InferenceEngine::Precision::UNSPECIFIED) {
            return desc;
        }
        auto element_type = tensor->get_element_type();
        auto shape = tensor->get_shape();
        ie::SizeVector blk_order(shape.size());
        std::iota(blk_order.begin(), blk_order.end(), 0);
        ie::SizeVector dim_offset(shape.size(), 0);
        ie::SizeVector blk_strides;
        auto byte_strides = element_type.bitwidth() >= 8 ? tensor->get_strides() : Strides{};
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
    };
#ifdef PROXY_PLUGIN_ENABLED
    const auto& tensor = unwrap ? ov::proxy::get_hardware_tensor(orig_tensor) : orig_tensor;
#else
    const auto& tensor = orig_tensor;
#endif
    if (tensor == nullptr) {
        return {};
    } else if (auto blob_tensor = std::dynamic_pointer_cast<BlobTensor>(tensor._ptr)) {
        return blob_tensor->blob;
    } else if (auto blob_tensor = std::dynamic_pointer_cast<RemoteBlobTensor>(tensor._ptr)) {
        return blob_tensor->blob;
    } else if (std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr)) {
        return std::make_shared<TensorRemoteBlob>(tensor, create_desc(tensor, desc));
    } else {
#define CASE(precision, T)   \
    case element::precision: \
        return std::make_shared<TensorMemoryBlob<T>>(tensor, create_desc(tensor, desc));
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
            return std::make_shared<TensorMemoryBlob<int16_t>>(tensor, create_desc(tensor, desc));
        case element::bf16:
            return std::make_shared<TensorMemoryBlob<int16_t>>(tensor, create_desc(tensor, desc));
        default:
            OPENVINO_THROW("Unsupported element type");
        }
#undef CASE
    }
    OPENVINO_THROW("Cannot convert tensor to blob!");
}  // namespace ov

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
