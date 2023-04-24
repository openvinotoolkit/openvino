// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <oneapi/dnnl/dnnl.hpp>
#include <vector>
#include <numeric>
#include <unordered_set>

#include <dnnl_types.h>
#include <common/memory_desc_wrapper.hpp>
#include "cpu_memory.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"
#include "onednn/dnnl.h"
#include "cpu_shape.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/reorder.h"
#include "memory_desc/cpu_memory_desc.h"

using namespace InferenceEngine;
using namespace dnnl;

namespace ov {
namespace intel_cpu {
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

Memory::Memory(const dnnl::engine& eng) :
    eng(eng), mgrHandle(std::make_shared<DnnlMemoryMngr>(std::unique_ptr<MemoryMngrWithReuse>(new MemoryMngrWithReuse())), this), dnnlMemHandle(this) {}
Memory::Memory(const dnnl::engine& eng, std::unique_ptr<IMemoryMngr> mngr) :
    eng(eng), mgrHandle(std::make_shared<DnnlMemoryMngr>(std::move(mngr)), this), dnnlMemHandle(this) {}

size_t Memory::GetSize() const {
    auto size = getDesc().getCurrentMemSize();
    if (size  == MemoryDesc::UNDEFINED_SIZE) {
        IE_THROW() << "Can't get memory size for undefined shape";
    }
    return size;
}

void Memory::Create(const MemoryDesc &desc, const void *data, bool pads_zeroing) {
    Create(desc.clone(), data, pads_zeroing);
}

void Memory::Create(MemoryDescPtr desc, const void* data, bool pads_zeroing) {
    pMemDesc = desc;
    padsZeroing = pads_zeroing;
    dnnlMemHandle.resetDnnlPrim();

    size_t memSize = 0;
    if (pMemDesc->isDefined()) {
        memSize = pMemDesc->getCurrentMemSize();
    }

    if (nullptr != data) {
        mgrHandle->setExtBuff(const_cast<void*>(data), memSize);
    } else {
        mgrHandle->resize(memSize);
    }
}

void Memory::SetData(const Memory& src, bool ftz) const {
    node::Reorder::reorderData(src, *this);

    auto localPrim = GetPrimitive();
    auto desc = localPrim.get_desc();
    dnnl::impl::memory_desc_wrapper wrapper(desc.get());

    if (ftz
        && src.GetDataType() == memory::data_type::f32
        && !wrapper.is_wino_desc()
        // WA: to avoid zero filling auxiliary information
        && !wrapper.is_rnn_packed_desc()
        && GetDataType() != memory::data_type::bf16) {
        // Internal blobs haven't strides yet.
        auto *memData = static_cast<float *>(GetData());
        memData += wrapper.offset0();
        setSubnormalsToZero(memData, GetSize() / sizeof(float));
    }
}

void Memory::FillZero() {
    void* dataPtr = GetData();
    if (dataPtr != nullptr)
        memset(dataPtr, 0, getDesc().getCurrentMemSize());
}

void *Memory::GetPtr() const  {
    auto ptr = static_cast<uint8_t*>(GetData());
    ptr += pMemDesc->getOffsetPadding() * pMemDesc->getPrecision().size();
    return ptr;
}

void Memory::redefineDesc(MemoryDescPtr desc) {
    if (!desc->hasDefinedMaxSize()) {
        IE_THROW() << "Can not reset descriptor, memory upper bound is unknown.";
    }

    this->Create(desc, nullptr, false);
}

template<>
DnnlMemoryDescPtr Memory::GetDescWithType<DnnlMemoryDesc, 0, 0>() const {
    return MemoryDescUtils::convertToDnnlMemoryDesc(pMemDesc);
}

void Memory::setDataHandle(void *data) {
    if (!mgrHandle->hasExtBuffer()) {
        mgrHandle = DnnlMemMngrHandle(
            std::make_shared<DnnlMemoryMngr>(std::unique_ptr<MemoryMngrWithReuse>(new MemoryMngrWithReuse())),
            this);
    }

    size_t maxMemSize = pMemDesc->isDefined() ?  pMemDesc->getCurrentMemSize() : 0;
    mgrHandle->setExtBuff(data, maxMemSize);
    if (dnnlMemHandle.isInit()) {
        auto prim = dnnlMemHandle.getPrim();
        prim.set_data_handle(mgrHandle->getRawPtr()); // for pads zeroing, to preserve dnnl::memory::set_data_handle behaviour
    }
}

void Memory::update() {
    if (dnnlMemHandle.isInit()) {
        auto prim = dnnlMemHandle.getPrim();
        prim.set_data_handle_no_pads_proc(mgrHandle->getRawPtr());
    }
}

void Memory::Create(const MemoryDesc &desc, DnnlMemoryMngrPtr memMgr) {
    Create(desc.clone(), memMgr);
}

void Memory::Create(MemoryDescPtr desc, DnnlMemoryMngrPtr memMgr) {
    mgrHandle = DnnlMemMngrHandle(memMgr, this);
    bool memAllocated = mgrHandle->getRawPtr();

    Create(desc, nullptr, !memAllocated);
}

template<>
BlockedMemoryDescPtr Memory::GetDescWithType<BlockedMemoryDesc, 0, 0>() const {
    return MemoryDescUtils::convertToBlockedMemoryDesc(pMemDesc);
}

dnnl::memory Memory::GetPrimitive() const {
    return dnnlMemHandle.getPrim();
}

void Memory::DnnlMemPrimHandle::resetDnnlPrim() {
    m_prim = dnnl::memory();
}

bool Memory::DnnlMemPrimHandle::isInit() const {
    std::lock_guard<std::mutex> guard(m_primCachingLock);
    return m_prim.get(true) != nullptr;
}

dnnl::memory Memory::DnnlMemPrimHandle::getPrim() const {
    std::lock_guard<std::mutex> guard(m_primCachingLock);
    if (!m_prim) {
        if (!m_memObjPtr->getDesc().isDefined()) {
            IE_THROW() << "Can not create oneDNN memory from undefined memory descriptor";
        }

        // ========================
        // Equivalent of constructor memory(const primitive_desc &desc, void *hdl)
        // but with ability to skip pads zeroing.
        auto desc = MemoryDescUtils::convertToDnnlMemoryDesc(m_memObjPtr->getDescPtr());
        m_prim = memory(desc->getDnnlDesc(), m_memObjPtr->getEngine(), DNNL_MEMORY_NONE);
        //
        // ========================
        auto data = m_memObjPtr->getDataNoThrow();
        auto pads_zeroing = m_memObjPtr->padsZeroing;
        if (data != nullptr) {
            if (pads_zeroing)
                m_prim.set_data_handle(data);
            else
                m_prim.set_data_handle_no_pads_proc(data);
        }
    }
    return m_prim;
}

void* MemoryMngrWithReuse::getRawPtr() const noexcept {
    return _data.get();
}

void MemoryMngrWithReuse::setExtBuff(void *ptr, size_t size) {
    _useExternalStorage = true;
    _memUpperBound = size;
    _data = decltype(_data)(ptr, release);
}

bool MemoryMngrWithReuse::resize(size_t size) {
    constexpr int cacheLineSize = 64;
    bool sizeChanged = false;
    if (size > _memUpperBound) {
        void *ptr = dnnl::impl::malloc(size, cacheLineSize);
        if (!ptr) {
            IE_THROW() << "Failed to allocate " << size << " bytes of memory";
        }
        _memUpperBound = size;
        _useExternalStorage = false;
        _data = decltype(_data)(ptr, destroy);
        sizeChanged = true;
    }
    return sizeChanged;
}

bool MemoryMngrWithReuse::hasExtBuffer() const noexcept {
    return _useExternalStorage;
}

void MemoryMngrWithReuse::release(void *ptr) {}

void MemoryMngrWithReuse::destroy(void *ptr) {
    dnnl::impl::free(ptr);
}

void* DnnlMemoryMngr::getRawPtr() const noexcept {
    return _pMemMngr->getRawPtr();
}

void DnnlMemoryMngr::setExtBuff(void *ptr, size_t size) {
    _pMemMngr->setExtBuff(ptr, size);
    notifyUpdate();
}

bool DnnlMemoryMngr::resize(size_t size) {
    bool sizeChanged = _pMemMngr->resize(size);
    if (sizeChanged) {
        notifyUpdate();
    }
    return sizeChanged;
}

bool DnnlMemoryMngr::hasExtBuffer() const noexcept {
    return _pMemMngr->hasExtBuffer();
}

void DnnlMemoryMngr::registerMemory(Memory* memPtr) {
    if (memPtr) {
        _setMemPtrs.insert(memPtr);
    }
}

void DnnlMemoryMngr::unregisterMemory(Memory* memPtr) {
    if (memPtr) {
        _setMemPtrs.erase(memPtr);
    }
}

void DnnlMemoryMngr::notifyUpdate() {
    for (auto& item : _setMemPtrs) {
        if (item) {
            item->update();
        }
    }
}
}   // namespace intel_cpu
}   // namespace ov
