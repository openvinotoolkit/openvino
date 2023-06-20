// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_tensor.h"
#include "ie_ngraph_utils.hpp"
#include "utils/debug_capabilities.h"

#include "openvino/runtime/iremote_tensor.hpp"
namespace InferenceEngine {
// a nutshell allocator which blindly locks any address without check.
class NutshellAllocator final: public InferenceEngine::IAllocator {
public:
    NutshellAllocator() {}

    void* lock(void* handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override {
        return handle;
    }

    void unlock(void* handle) noexcept override {}

    void* alloc(size_t size) noexcept override {
        IE_ASSERT("SHOULD NOT BE HERE!");
        return nullptr;
    }

    bool free(void* handle) noexcept override {
        return true;
    }

private:
};

std::shared_ptr<IAllocator> make_nutshell_allocator() noexcept {
    return std::make_shared<NutshellAllocator>();
}
}  // namespace InferenceEngine

namespace ov {
namespace intel_cpu {

Tensor::Tensor(MemoryPtr memptr) : m_memptr{memptr} {
    OPENVINO_ASSERT(m_memptr != nullptr);

    // only support plain data format ncsp.
    auto memdesc = m_memptr->getDescPtr();
    OPENVINO_ASSERT(memdesc->hasLayoutType(LayoutType::ncsp), "intel_cpu::Tensor only supports memory with ncsp layout.");

    m_element_type = InferenceEngine::details::convertPrecision(memdesc->getPrecision());
}

void Tensor::set_shape(ov::Shape new_shape) {
    auto desc = m_memptr->getDescPtr();
    const auto newdesc = desc->cloneWithNewDims(new_shape, true);
    m_memptr->redefineDesc(newdesc);
}

const ov::element::Type& Tensor::get_element_type() const {
    return m_element_type;
}

const ov::Shape& Tensor::get_shape() const {
    auto& shape = m_memptr->getDescPtr()->getShape();
    OPENVINO_ASSERT(shape.isStatic(), "intel_cpu::Tensor has dynamic shape.");

    std::lock_guard<std::mutex> guard(m_lock);
    m_shape = ov::Shape{shape.getStaticDims()};
    return m_shape;
}

size_t Tensor::get_size() const {
    auto& desc = m_memptr->getDesc();
    return desc.getShape().getElementsCount();
}

size_t Tensor::get_byte_size() const {
    auto& desc = m_memptr->getDesc();
    return desc.getCurrentMemSize();
}

const ov::Strides& Tensor::get_strides() const {
    OPENVINO_ASSERT(m_memptr->getDescPtr()->isDefined(), "intel_cpu::Tensor requires memory with defined strides.");

    std::lock_guard<std::mutex> guard(m_lock);
    update_strides();
    return m_strides;
}

void Tensor::update_strides() const {
    auto blocked_desc = m_memptr->getDescWithType<BlockedMemoryDesc>();
    OPENVINO_ASSERT(blocked_desc, "not a valid blocked memory descriptor.");
    auto& strides = blocked_desc->getStrides();
    m_strides.resize(strides.size());
    std::transform(strides.cbegin(), strides.cend(), m_strides.begin(),
                std::bind1st(std::multiplies<size_t>(), m_element_type.size()));
}

void* Tensor::data(const element::Type& element_type) const {
    if (element_type != element::undefined && element_type != element::dynamic) {
        OPENVINO_ASSERT(element_type == get_element_type(),
                        "Tensor data with element type ",
                        get_element_type(),
                        ", is not representable as pointer to ",
                        element_type);
    }
    return m_memptr->getData();
}

/**
 * @brief Creates tensor on graph memory
 *
 * @param mem Memory object
 *
 * @return Shared pointer to tensor interface
 */
std::shared_ptr<ITensor> make_tensor(MemoryPtr mem) {
    return std::make_shared<Tensor>(mem);
}

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
                   ie::make_nutshell_allocator()},
            tensor{tensor_} {
            OPENVINO_ASSERT(!std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor));
        }
    catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what());
    }

    void setShape(const ie::SizeVector& dims) override {
        auto _data = tensor->data();
        tensor->set_shape(dims);
        DEBUG_LOG(_data, " -> ",  tensor->data());
        // ie::TBlob<T>::setShape(dims);
        ie::TBlob<T>::getTensorDesc().setDims(dims);
    }

    /**
     * @brief Creates a LockedMemory instance.
     *
     * @tparam S Type of the LockedMemory to be created
     * @return A created instance of LockedMemory
     */
    template <class S>
    ie::LockedMemory<S> lockme() const {
        auto _data = ie::LockedMemory<S>(ie::TBlob<T>::_allocator.get(), tensor->data(), 0);
        DEBUG_LOG(static_cast<S*>(_data));
        return _data;
    }

    ie::LockedMemory<void> buffer() noexcept override {
        return lockme<void>();
    }

    ie::LockedMemory<const void> cbuffer() const noexcept override {
        return lockme<const void>();
    }

    ie::LockedMemory<void> rwmap() noexcept override {
        return lockme<void>();
    }

    ie::LockedMemory<const void> rmap() const noexcept override {
        return lockme<const void>();
    }

    ie::LockedMemory<void> wmap() noexcept override {
        return lockme<void>();
    }

    std::shared_ptr<ITensor> tensor;
};

ie::Blob::Ptr tensor_to_blob(const std::shared_ptr<ITensor>& tensor) {
    if (tensor == nullptr) {
        return {};
    } else {
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
            OPENVINO_THROW("Unsupported element type");
        }
#undef CASE
    }
    OPENVINO_THROW("Cannot convert tensor to blob!");
}

}   // namespace intel_cpu
}   // namespace ov