// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "blob_factory.hpp"     // IE private header
#include "ie_ngraph_utils.hpp"  // IE private header
#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"
#include "runtime/blob_allocator.hpp"

namespace ov {
namespace runtime {

#define OV_TENSOR_STATEMENT(...)                                      \
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized."); \
    try {                                                             \
        __VA_ARGS__;                                                  \
    } catch (const std::exception& ex) {                              \
        throw ov::Exception(ex.what());                               \
    } catch (...) {                                                   \
        OPENVINO_ASSERT(false, "Unexpected exception");               \
    }

void Tensor::type_check(const Tensor&) {}

Tensor::Tensor(const std::shared_ptr<void>& so, const std::shared_ptr<ie::Blob>& impl) : _so{so}, _impl{impl} {
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized.");
}

Tensor::Tensor(const element::Type element_type, const Shape& shape, const Allocator& allocator) {
    OPENVINO_ASSERT(allocator, "Allocator was not initialized");
    auto allocator_impl = dynamic_cast<const BlobAllocator*>(allocator._impl.get());
    auto blob_allocator =
        (allocator_impl != nullptr) ? allocator_impl->_impl : std::make_shared<ie::BlobAllocator>(allocator._impl);
    _impl = make_blob_with_precision(
        {ie::details::convertPrecision(element_type), shape, ie::TensorDesc::getLayoutByRank(shape.size())},
        blob_allocator);
    _impl->allocate();
}

Tensor::Tensor(const element::Type element_type, const Shape& shape, void* host_ptr, const Strides& strides) {
    ie::SizeVector blk_order(shape.size());
    std::iota(blk_order.begin(), blk_order.end(), 0);
    ie::SizeVector dim_offset(shape.size(), 0);
    ie::SizeVector blk_strides;
    if (strides.empty()) {
        blk_strides = ov::row_major_strides(shape);
    } else {
        blk_strides.assign(strides.begin(), strides.end());
    }

    try {
        _impl = make_blob_with_precision(ie::details::convertPrecision(element_type),
                                         ie::TensorDesc{ie::details::convertPrecision(element_type),
                                                        shape,
                                                        ie::BlockingDesc{shape, blk_order, 0, dim_offset, blk_strides}},
                                         host_ptr);
    } catch (const std::exception& ex) {
        throw ov::Exception(ex.what());
    } catch (...) {
        OPENVINO_ASSERT(false, "Unexpected exception");
    }
}

Tensor::Tensor(const Tensor& owner, const Coordinate& begin, const Coordinate& end) : _so{owner._so} {
    OPENVINO_ASSERT(owner.get_element_type().bitwidth() >= 8,
                    "ROI Tensor for types with bitwidths less then 8 bit is not implemented. Tensor type: ",
                    owner.get_element_type());
    try {
        _impl = owner._impl->createROI(begin, end);
    } catch (const std::exception& ex) {
        throw ov::Exception(ex.what());
    } catch (...) {
        OPENVINO_ASSERT(false, "Unexpected exception");
    }
}

element::Type Tensor::get_element_type() const {
    OV_TENSOR_STATEMENT(return ie::details::convertPrecision(_impl->getTensorDesc().getPrecision()));
}

void Tensor::set_shape(const ov::Shape& shape) {
    OV_TENSOR_STATEMENT(_impl->setShape({shape.begin(), shape.end()}));
}

Shape Tensor::get_shape() const {
    OV_TENSOR_STATEMENT({ return _impl->getTensorDesc().getDims(); });
}

Strides Tensor::get_strides() const {
    OPENVINO_ASSERT(get_element_type().bitwidth() >= 8,
                    "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                    get_element_type());
    OV_TENSOR_STATEMENT(return _impl->getTensorDesc().getBlockingDesc().getStrides());
}

size_t Tensor::get_size() const {
    OV_TENSOR_STATEMENT(return _impl->size());
}

size_t Tensor::get_byte_size() const {
    OV_TENSOR_STATEMENT(return _impl->byteSize(););
}

void* Tensor::data(const element::Type element_type) const {
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized.");
#define TYPE_CHECK(TYPE) (dynamic_cast<const ie::TBlob<TYPE>*>(_impl.get()) != nullptr)
    auto host_accesable_implementation = TYPE_CHECK(bool) || TYPE_CHECK(int8_t) || TYPE_CHECK(uint8_t) ||
                                         TYPE_CHECK(int16_t) || TYPE_CHECK(uint16_t) || TYPE_CHECK(int32_t) ||
                                         TYPE_CHECK(uint32_t) || TYPE_CHECK(int64_t) || TYPE_CHECK(uint64_t) ||
                                         TYPE_CHECK(float) || TYPE_CHECK(double);
#undef TYPE_CHECK
    OPENVINO_ASSERT(host_accesable_implementation, "Tensor implementation type dose not contains host accessable data");
    if (element_type != element::undefined) {
        OPENVINO_ASSERT(element_type == get_element_type(),
                        "Tensor data with element type ",
                        get_element_type(),
                        ", is not representable as pointer to ",
                        element_type);
    }
    auto byte_offset = _impl->getTensorDesc().getBlockingDesc().getOffsetPadding() * get_element_type().size();
    OPENVINO_ASSERT((get_element_type().bitwidth() >= 8) || (byte_offset == 0),
                    "ROI access for types with bitwidths less then 8 bit is not implemented. Tensor type: ",
                    get_element_type());
    OV_TENSOR_STATEMENT(
        { return byte_offset + InferenceEngine::as<InferenceEngine::MemoryBlob>(_impl)->rmap().as<uint8_t*>(); });
}

bool Tensor::operator!() const noexcept {
    return !_impl;
}

Tensor::operator bool() const noexcept {
    return (!!_impl);
}

}  // namespace runtime
}  // namespace ov
