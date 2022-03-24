// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp_interfaces/interface/itensor.hpp"

#include "ie_ngraph_utils.hpp"
#include "ie_remote_blob.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {

Shape ITensor::get_shape() const {
    OPENVINO_UNREACHABLE("Not implemented");
}

element::Type ITensor::get_element_type() const {
    OPENVINO_UNREACHABLE("Not implemented");
}

size_t ITensor::get_size() const {
    return shape_size(get_shape());
}

void ITensor::set_shape(const ov::Shape& new_shape) {
    OPENVINO_UNREACHABLE("Not implemented");
}

Strides ITensor::get_strides() const {
    auto element_type = get_element_type();
    OPENVINO_ASSERT(element_type.bitwidth() >= 8,
                    "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                    element_type);
    auto shape = get_shape();
    Strides strides;
    if (!shape.empty()) {
        strides.resize(shape.size());
        strides.back() = element_type.size();
        std::copy(shape.rbegin(), shape.rend() - 1, strides.rbegin() + 1);
        std::partial_sum(strides.rbegin(), strides.rend(), strides.rbegin(), std::multiplies<size_t>());
    }
    return strides;
}

size_t ITensor::get_byte_size() const {
    return (get_size() * get_element_type().bitwidth() + 8 - 1) / 8;
}

void* ITensor::data(const element::Type) const {
    OPENVINO_UNREACHABLE("Not implemented");
}

AnyMap ITensor::get_properties() const {
    return {};
}

Coordinate ITensor::get_offsets() const {
    return {get_shape().size(), 0};
}

struct ViewTensor : public ITensor {
    ViewTensor(const element::Type element_type_, const Shape& shape_, void* ptr_)
        : element_type{element_type_},
          shape{shape_},
          ptr{ptr_} {
        OPENVINO_ASSERT(ptr != nullptr);
    }

    void* data(const element::Type element_type) const override {
        if (element_type != element::undefined) {
            OPENVINO_ASSERT(element_type == get_element_type(),
                            "Tensor data with element type ",
                            get_element_type(),
                            ", is not representable as pointer to ",
                            element_type);
        }
        return ptr;
    }

    element::Type get_element_type() const override {
        return element_type;
    }

    Shape get_shape() const override {
        return shape;
    }

    void set_shape(const ov::Shape& new_shape) override {
        auto old_byte_size = get_byte_size();
        OPENVINO_ASSERT(shape_size(new_shape) <= old_byte_size, "Could set new shape: ", new_shape);
        shape = new_shape;
    }

    element::Type element_type;
    Shape shape;
    void* ptr;
};

struct StridedViewTensor : public ViewTensor {
    StridedViewTensor(const element::Type element_type_, const Shape& shape_, void* ptr_, const Strides& strides_)
        : ViewTensor{element_type_, shape_, ptr_},
          strides{strides_} {
        OPENVINO_ASSERT(
            get_element_type().bitwidth() >= 8,
            "Could not create strided access tensor for types with bitwidths less then 8 bit. Tensor type: ",
            get_element_type());
        OPENVINO_ASSERT(get_shape().size() == strides.size());
        auto shape_strides = ITensor::get_strides();
        for (size_t i = 0; i < strides.size(); ++i) {
            OPENVINO_ASSERT(shape_strides[i] <= strides[i],
                            "shape stride: ",
                            shape_strides[i],
                            ", stride: ",
                            strides[i]);
            OPENVINO_ASSERT((strides[i] % get_element_type().size()) == 0,
                            "shape stride: ",
                            shape_strides[i],
                            ", stride: ",
                            strides[i]);
        }
    }

    Strides get_strides() const override {
        return strides;
    }

    Strides strides;
};

ITensor::Ptr make_tensor(const element::Type element_type_,
                         const Shape& shape_,
                         void* ptr,
                         const Strides& byte_strides) {
    return byte_strides.empty() ? std::make_shared<ViewTensor>(element_type_, shape_, ptr)
                                : std::make_shared<StridedViewTensor>(element_type_, shape_, ptr, byte_strides);
}

struct AllocatedTensor : public ViewTensor {
    AllocatedTensor(const element::Type element_type_, const Shape& shape_, const Allocator& allocator_)
        : ViewTensor{element_type_,
                     shape_,
                     [&, this] {
                         OPENVINO_ASSERT(allocator_, "Allocator was not initialized");
                         return const_cast<Allocator&>(allocator_).allocate(element_type_.size() * shape_size(shape_));
                     }()},
          allocator{allocator_} {}

    ~AllocatedTensor() {
        allocator.deallocate(ptr, get_byte_size());
    }

    void set_shape(const ov::Shape& new_shape) override {
        auto old_byte_size = get_byte_size();
        shape = new_shape;
        if (shape_size(new_shape) > old_byte_size) {
            allocator.deallocate(ptr, old_byte_size);
            ptr = allocator.allocate(get_byte_size());
        }
    }

    Allocator allocator;
};

ITensor::Ptr make_tensor(const element::Type element_type_, const Shape& shape_, const Allocator& allocator_) {
    return std::make_shared<AllocatedTensor>(element_type_, shape_, allocator_);
}

struct RoiTensor : public ITensor {
    RoiTensor(const ITensor::Ptr& owner_, const Coordinate& begin, const Coordinate& end)
        : owner{owner_},
          offsets{begin} {
        OPENVINO_ASSERT(owner->get_element_type().bitwidth() >= 8,
                        "ROI Tensor for types with bitwidths less then 8 bit is not implemented. Tensor type: ",
                        owner->get_element_type());
        auto owner_shape = owner->get_shape();
        OPENVINO_ASSERT(owner_shape.size() == begin.size());
        OPENVINO_ASSERT(begin.size() == end.size());
        shape.resize(begin.size());
        for (size_t i = 0; i < begin.size(); ++i) {
            OPENVINO_ASSERT(begin[i] <= owner_shape[i]);
            OPENVINO_ASSERT(end[i] <= owner_shape[i]);
            shape[i] = end[i] - begin[i];
            OPENVINO_ASSERT(shape[i] <= owner_shape[i]);
        }
    }

    element::Type get_element_type() const override {
        return owner->get_element_type();
    }

    Coordinate get_offsets() const override {
        return offsets;
    }

    Strides get_strides() const override {
        return owner->get_strides();
    }

    Shape get_shape() const override {
        return shape;
    }

    void* data(const element::Type element_type) const override {
        auto owner_data = owner->data(element_type);
        size_t byte_offset = 0;
        auto offsets = get_offsets();
        auto strides = get_strides();
        for (size_t i = 0; i < strides.size(); ++i) {
            byte_offset += offsets[i] * strides[i];
        }
        return static_cast<uint8_t*>(owner_data) + byte_offset;
    }

    ITensor::Ptr owner;
    Coordinate offsets;
    Shape shape;
};

ITensor::Ptr make_tensor(const ITensor::Ptr& other, const Coordinate& begin, const Coordinate& end) {
    return std::make_shared<RoiTensor>(other, begin, end);
}

struct BlobTensor : public ITensor {
    BlobTensor(const ie::Blob::Ptr& blob_) : blob{blob_} {}

    element::Type get_element_type() const override {
        return ie::details::convertPrecision(blob->getTensorDesc().getPrecision());
    }

    void set_shape(const ov::Shape& shape) override {
        blob->setShape({shape.begin(), shape.end()});
    }

    Shape get_shape() const override {
        return blob->getTensorDesc().getDims();
    }

    Strides get_strides() const override {
        OPENVINO_ASSERT(get_element_type().bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                        get_element_type());
        const auto& element_strides = blob->getTensorDesc().getBlockingDesc().getStrides();
        const size_t elem_size = get_element_type().size();
        Strides byte_strides;
        byte_strides.resize(element_strides.size());
        std::transform(element_strides.begin(),
                       element_strides.end(),
                       byte_strides.begin(),
                       [&elem_size](size_t stride) {
                           return stride * elem_size;
                       });
        return byte_strides;
    }

    size_t get_size() const override {
        return blob->size();
    }

    size_t get_byte_size() const override {
        return blob->byteSize();
    }

    void* data(const element::Type element_type) const override {
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
        auto remote_impl = dynamic_cast<ie::RemoteBlob*>(blob.get());
        if (remote_impl != nullptr) {
            auto params = remote_impl->getParams();
            params.insert(ov::device::id(remote_impl->getDeviceName()));
            return params;
        } else {
            return {};
        }
    }
    std::shared_ptr<ie::Blob> blob;
};

ITensor::Ptr blob_to_tensor(const std::shared_ptr<ie::Blob>& blob) {
    if (blob == nullptr) {
        return {};
    } else {
        return std::make_shared<BlobTensor>(blob);
    }
}

struct TensorRemoteBlob : public ie::RemoteBlob {
    TensorRemoteBlob(const ITensor::Ptr& tensor_)
        : ie::RemoteBlob{ie::TensorDesc{ie::details::convertPrecision(tensor_->get_element_type()),
                                        tensor_->get_shape(),
                                        ie::TensorDesc::getLayoutByRank(tensor_->get_shape().size())}},
          tensor{tensor_} {}
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
        return allocator;
    }
    void* getHandle() const noexcept override {
        return nullptr;
    }
    ITensor::Ptr tensor;
    std::shared_ptr<ie::IAllocator> allocator;
};

template <typename T>
struct TensorMemoryBlob : public ie::TBlob<T> {
    ~TensorMemoryBlob() override = default;
    explicit TensorMemoryBlob(const ITensor::Ptr& tensor_) try : ie
        ::TBlob<T>{[&] {
                       auto element_type = tensor_->get_element_type();
                       auto shape = tensor_->get_shape();
                       ie::SizeVector blk_order(shape.size());
                       std::iota(blk_order.begin(), blk_order.end(), 0);
                       ie::SizeVector dim_offset(shape.size(), 0);
                       ie::SizeVector blk_strides;
                       auto byte_strides = tensor_->get_strides();
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
    } catch (...) {
        OPENVINO_ASSERT(false, "Unexpected exception");
    }
    ITensor::Ptr tensor;
};

ie::Blob::Ptr tensor_to_blob(const ITensor::Ptr& tensor) {
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
