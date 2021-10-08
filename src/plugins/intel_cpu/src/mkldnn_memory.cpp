// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <numeric>
#include <unordered_set>

#include <mkldnn_types.h>
#include <dnnl_types.h>
#include <common/memory_desc_wrapper.hpp>
#include "mkldnn_memory.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"
#include "mkldnn/ie_mkldnn.h"
#include "cpu_shape.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/mkldnn_reorder_node.h"
#include "memory_desc/cpu_memory_desc.h"

using namespace InferenceEngine;
using namespace mkldnn;

namespace MKLDNNPlugin {
namespace {
    inline void setSubnormalsToZero(float *data, size_t size) {
        uint32_t *u32data = reinterpret_cast<uint32_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            if ((u32data[i] & (0xFF << 23)) == 0) {
                u32data[i] = 0;
            }
        }
    }
}   // namespace

MKLDNNMemory::MKLDNNMemory(const mkldnn::engine& eng) : eng(eng), pMngr(new MemoryMngrWithReuse()) {}
MKLDNNMemory::MKLDNNMemory(const mkldnn::engine& eng, std::unique_ptr<MKLDNNMemoryMngrInterface> mngr) : eng(eng), pMngr(std::move(mngr)) {}

size_t MKLDNNMemory::GetSize() const {
    auto size = getDesc().getCurrentMemSize();
    if (size  == MemoryDesc::UNDEFINED_SIZE) {
        IE_THROW() << "Can't get memory size for undefined shape";
    }
    return size;
}

void MKLDNNMemory::Create(const memory::dims& dims, memory::data_type data_type, memory::format_tag format, const void* data) {
    if (format == memory::format_tag::undef) {
        format = memory::format_tag::any;
    }

    memory::desc desc = mkldnn::memory::desc(dims, data_type, format);

    Create(desc, data);
}

void MKLDNNMemory::Create(const mkldnn::memory::desc& desc, const void *data, bool pads_zeroing) {
    // MKLDNN accepts not a const data, probably need to remove some level of consteness in a call stack

    // ========================
    // Equivalent of constructor memory(const primitive_desc &desc, void *hdl)
    // but with ability to skipp pads zeroing.
    prim.reset(new memory(desc, eng, DNNL_MEMORY_NONE));
    //
    // ========================
    if (data != nullptr) {
        if (pads_zeroing)
            prim->set_data_handle(const_cast<void*>(data));
        else
            prim->set_data_handle_no_pads_proc(const_cast<void*>(data));
    }
}

void MKLDNNMemory::Create(const MemoryDesc &desc, const void *data, bool pads_zeroing) {
    Create(desc.clone(), data, pads_zeroing);
}

void MKLDNNMemory::Create(MemoryDescPtr desc, const void* data, bool pads_zeroing) {
    pMemDesc = desc;

    size_t memSize = MemoryDesc::UNDEFINED_SIZE;
    if (pMemDesc->isDefined()) {
        memSize = pMemDesc->getCurrentMemSize();
    } else {
        memSize = pMemDesc->hasDefinedMaxSize() ? pMemDesc->getMaxMemSize() : 0;
    }

    if (nullptr != data) {
        pMngr->setExtBuff(const_cast<void*>(data), memSize);
    } else {
        pMngr->resize(memSize);
    }

    if (pMemDesc->isDefined()) {
        Create(MemoryDescUtils::convertToDnnlMemoryDesc(pMemDesc)->getDnnlDesc(), pMngr->getRawPtr(), pads_zeroing);
    } else {
        //delayed dynamic allocation
        VectorDims dummySize{ pMemDesc->hasDefinedMaxSize() ? pMemDesc->getMaxMemSize() : 0 };
        DnnlBlockedMemoryDesc dummyDesc(InferenceEngine::Precision::U8, Shape(dummySize));
        Create(dummyDesc.getDnnlDesc(), pMngr->getRawPtr(), false);  // no pads zeroing
    }
}

void MKLDNNMemory::SetData(const MKLDNNMemory& src, size_t size, bool ftz) const {
    MKLDNNReorderNode::reorderData(src, *this, size);

    if (ftz
        && src.GetDataType() == memory::data_type::f32
        && prim->get_desc().data.format_kind != dnnl_format_kind_wino
        // WA: to avoid zero filling auxiliary information
        && prim->get_desc().data.format_kind != dnnl_format_kind_rnn_packed
        && GetDataType() != memory::data_type::bf16) {
        // Internal blobs haven't strides yet.
        auto *memData = static_cast<float *>(GetData());
        memData += prim->get_desc().data.offset0;
        setSubnormalsToZero(memData, GetSize() / sizeof(float));
    }
}

void MKLDNNMemory::FillZero() {
    void* dataPtr = GetData();
    if (dataPtr != nullptr)
        memset(dataPtr, 0, getDesc().getMaxMemSize());
}

void *MKLDNNMemory::GetPtr() const  {
    auto ptr = static_cast<uint8_t*>(GetData());
    auto md = prim->get_desc().data;
    mkldnn::impl::memory_desc_wrapper wrapper(md);
    ptr += wrapper.offset0() * wrapper.data_type_size();
    return ptr;
}

void MKLDNNMemory::redefineDesc(MemoryDescPtr desc) {
    if (!desc->hasDefinedMaxSize()) {
        IE_THROW() << "Can not reset descriptor, memory upper bound is unknown.";
    }

    this->Create(desc, nullptr, false);
}

template<>
DnnlMemoryDescPtr MKLDNNMemory::GetDescWithType<DnnlMemoryDesc, 0, 0>() const {
    return MemoryDescUtils::convertToDnnlMemoryDesc(pMemDesc);
}

void MKLDNNMemory::setDataHandle(void *data) {
    size_t maxMemSize = pMemDesc->hasDefinedMaxSize() ?  pMemDesc->getMaxMemSize() : 0;
    pMngr->setExtBuff(data, maxMemSize);
}

template<>
BlockedMemoryDescPtr MKLDNNMemory::GetDescWithType<BlockedMemoryDesc, 0, 0>() const {
    return MemoryDescUtils::convertToBlockedMemoryDesc(pMemDesc);
}

void* MemoryMngrWithReuse::getRawPtr() const noexcept {
    return data.get();
}

void MemoryMngrWithReuse::setExtBuff(void *ptr, size_t size) {
    useExternalStorage = true;
    memUpperBound = size;
    data = decltype(data)(ptr, release);
}

void MemoryMngrWithReuse::resize(size_t size) {
    constexpr int cacheLineSize = 64;
    if (size > memUpperBound) {
        void *ptr = dnnl::impl::malloc(size, cacheLineSize);
        if (!ptr) {
            throw std::bad_alloc();
        }
        memUpperBound = size;
        useExternalStorage = false;
        data = decltype(data)(ptr, destroy);
    }
}

bool MemoryMngrWithReuse::hasExtBuffer() const noexcept {
    return useExternalStorage;
}

void MemoryMngrWithReuse::release(void *ptr) {}

void MemoryMngrWithReuse::destroy(void *ptr) {
    dnnl::impl::free(ptr);
}
}  // namespace MKLDNNPlugin
