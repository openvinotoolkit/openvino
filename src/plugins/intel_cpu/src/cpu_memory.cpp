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
template <>
DnnlMemoryDescPtr IMemory::getDescWithType<DnnlMemoryDesc, 0, 0>() const {
    return MemoryDescUtils::convertToDnnlMemoryDesc(getDescPtr());
}

template <>
BlockedMemoryDescPtr IMemory::getDescWithType<BlockedMemoryDesc, 0, 0>() const {
    return MemoryDescUtils::convertToBlockedMemoryDesc(getDescPtr());
}

namespace {
    inline void setSubnormalsToZero(float *data, size_t size) {
        uint32_t *u32data = reinterpret_cast<uint32_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            if ((u32data[i] & (0xFF << 23)) == 0) {
                u32data[i] = 0;
            }
        }
    }

    void transferData(const IMemory& src, const IMemory& dst, bool ftz) {
        node::Reorder::reorderData(src, dst);

        if (!ftz) {
            return;
        }
        if (src.getDesc().getPrecision() != Precision::FP32 || dst.getDesc().getPrecision() == Precision::BF16) {
            return;
        }
        size_t offset = 0;
        if (dst.getDesc().getType() & MemoryDescType::Dnnl) {
            // here we can safely cast to DnnlMemoryDesc
            auto dnnl_desc = dst.getDescWithType<DnnlMemoryDesc>();
            auto desc = dnnl_desc->getDnnlDesc();
            dnnl::impl::memory_desc_wrapper wrapper(desc.get());
            offset = wrapper.offset0();
            if (wrapper.is_wino_desc() || wrapper.is_rnn_packed_desc()) {
                return;
            }
        }
        // actual FTZ
        auto* memData = static_cast<float*>(dst.getData());
        memData += offset;
        setSubnormalsToZero(memData, dst.getSize() / sizeof(float));
    }

}   // namespace

Memory::Memory(const dnnl::engine& eng, MemoryDescPtr desc, const void* data, bool pads_zeroing) :
    m_eng(eng),
    m_pMemDesc(desc),
    m_mgrHandle(std::make_shared<DnnlMemoryMngr>(make_unique<MemoryMngrWithReuse>()), this),
    dnnlMemHandle(this) {
        create(m_pMemDesc, data, pads_zeroing);
    }

Memory::Memory(const dnnl::engine& eng, const MemoryDesc& desc, const void* data, bool pads_zeroing) :
    Memory::Memory(eng, desc.clone(), data, pads_zeroing) {}

Memory::Memory(const dnnl::engine& eng, MemoryDescPtr desc, MemoryMngrPtr mngr) :
    m_eng(eng), m_pMemDesc(desc), m_mgrHandle(mngr, this), dnnlMemHandle(this) {
        bool memAllocated = m_mgrHandle->getRawPtr();

        create(desc, nullptr, !memAllocated);
    }

Memory::Memory(const dnnl::engine& eng, const MemoryDesc& desc, MemoryMngrPtr mngr) :
    Memory::Memory(eng, desc.clone(), mngr) {}

size_t Memory::getSize() const {
    auto size = getDesc().getCurrentMemSize();
    if (size  == MemoryDesc::UNDEFINED_SIZE) {
        IE_THROW() << "Can't get memory size for undefined shape";
    }
    return size;
}

void Memory::create(const MemoryDesc &desc, const void *data, bool pads_zeroing) {
    create(desc.clone(), data, pads_zeroing);
}

void Memory::create(MemoryDescPtr desc, const void* data, bool pads_zeroing) {
    m_pMemDesc = desc;
    m_padsZeroing = pads_zeroing;
    dnnlMemHandle.resetDnnlPrim();

    if (!m_pMemDesc->isDefined()) {
        return;
    }
    auto memSize = m_pMemDesc->getCurrentMemSize();
    if (nullptr != data) {
        m_mgrHandle->setExtBuff(const_cast<void*>(data), memSize);
    } else {
        m_mgrHandle->resize(memSize);
    }
}

void Memory::load(const IMemory& src, bool ftz) const {
    transferData(src, *this, ftz);
}

void Memory::nullify() {
    void* dataPtr = getData();
    if (dataPtr != nullptr)
        memset(dataPtr, 0, getDesc().getCurrentMemSize());
}

void Memory::redefineDesc(MemoryDescPtr desc) {
    if (!desc->hasDefinedMaxSize()) {
        IE_THROW() << "Can not reset descriptor, memory upper bound is unknown.";
    }

    this->create(desc, nullptr, false);
}

void Memory::update() {
    if (dnnlMemHandle.isInit()) {
        auto prim = dnnlMemHandle.getPrim();
        prim.set_data_handle_no_pads_proc(m_mgrHandle->getRawPtr());
    }
}

dnnl::memory Memory::getPrimitive() const {
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
        auto pads_zeroing = m_memObjPtr->m_padsZeroing;
        if (data != nullptr) {
            if (pads_zeroing)
                m_prim.set_data_handle(data);
            else
                m_prim.set_data_handle_no_pads_proc(data);
        }
    }
    return m_prim;
}

bool Memory::isAllocated() const noexcept {
    if (m_mgrHandle->getRawPtr()) {
        return true;
    }
    if (!m_pMemDesc) {
        return false;
    }
    if (!(m_pMemDesc->isDefined())) {
        return true;
    }
    if (m_pMemDesc->getCurrentMemSize() == 0) {
        return true;
    }
    return false;
}

void* Memory::getData() const {
    void* data = getDataNoThrow();
    if (data == nullptr &&
        m_pMemDesc->getShape().isStatic() &&
        m_pMemDesc->getShape().getElementsCount() != 0)
        IE_THROW() << "Memory has not been allocated";
    return data;
}

void* MemoryMngrWithReuse::getRawPtr() const noexcept {
    return m_data.get();
}

void MemoryMngrWithReuse::setExtBuff(void *ptr, size_t size) {
    m_useExternalStorage = true;
    m_memUpperBound = size;
    m_data = decltype(m_data)(ptr, release);
}

bool MemoryMngrWithReuse::resize(size_t size) {
    constexpr int cacheLineSize = 64;
    bool sizeChanged = false;
    if (size > m_memUpperBound) {
        void *ptr = dnnl::impl::malloc(size, cacheLineSize);
        if (!ptr) {
            IE_THROW() << "Failed to allocate " << size << " bytes of memory";
        }
        m_memUpperBound = size;
        m_useExternalStorage = false;
        m_data = decltype(m_data)(ptr, destroy);
        sizeChanged = true;
    }
    return sizeChanged;
}

bool MemoryMngrWithReuse::hasExtBuffer() const noexcept {
    return m_useExternalStorage;
}

void MemoryMngrWithReuse::release(void *ptr) {}

void MemoryMngrWithReuse::destroy(void *ptr) {
    dnnl::impl::free(ptr);
}

void* DnnlMemoryMngr::getRawPtr() const noexcept {
    return m_pMemMngr->getRawPtr();
}

void DnnlMemoryMngr::setExtBuff(void *ptr, size_t size) {
    m_pMemMngr->setExtBuff(ptr, size);
    notifyUpdate();
}

bool DnnlMemoryMngr::resize(size_t size) {
    bool sizeChanged = m_pMemMngr->resize(size);
    if (sizeChanged) {
        notifyUpdate();
    }
    return sizeChanged;
}

bool DnnlMemoryMngr::hasExtBuffer() const noexcept {
    return m_pMemMngr->hasExtBuffer();
}

void DnnlMemoryMngr::registerMemory(Memory* memPtr) {
    if (memPtr) {
        m_setMemPtrs.insert(memPtr);
    }
}

void DnnlMemoryMngr::unregisterMemory(Memory* memPtr) {
    if (memPtr) {
        m_setMemPtrs.erase(memPtr);
    }
}

void DnnlMemoryMngr::notifyUpdate() {
    for (auto& item : m_setMemPtrs) {
        if (item) {
            item->update();
        }
    }
}

StaticMemory::StaticMemory(const dnnl::engine& eng, MemoryDescPtr desc, const void* data, bool pads_zeroing) :
    m_eng(eng), m_pMemDesc(desc) {
    if (!m_pMemDesc->isDefined()) {
        IE_THROW() << "Can not create StaticMemory object. The memory desc is undefined";
    }

    m_size = m_pMemDesc->getCurrentMemSize();

    auto dnnl_desc = MemoryDescUtils::convertToDnnlMemoryDesc(m_pMemDesc);

    if (data) {
        m_pMemMngr = std::make_shared<StaticMemoryMngr>(const_cast<void*>(data), m_size);
    } else {
        m_pMemMngr = std::make_shared<StaticMemoryMngr>(m_size);
    }

    // ========================
    // Equivalent of constructor memory(const primitive_desc &desc, void *hdl)
    // but with ability to skip pads zeroing.
    m_prim = memory(dnnl_desc->getDnnlDesc(), m_eng, DNNL_MEMORY_NONE);
    //
    // ========================
    if (pads_zeroing)
        m_prim.set_data_handle(m_pMemMngr->getRawPtr());
    else
        m_prim.set_data_handle_no_pads_proc(m_pMemMngr->getRawPtr());
}

StaticMemory::StaticMemory(const dnnl::engine& eng, const MemoryDesc& desc, const void* data, bool pads_zeroing) :
    StaticMemory::StaticMemory(eng, desc.clone(), data, pads_zeroing) {}

bool StaticMemory::isAllocated() const noexcept {
    return 0 == m_size || getData() != nullptr;
}

const MemoryDesc& StaticMemory::getDesc() const {
    return *m_pMemDesc;
}

MemoryDescPtr StaticMemory::getDescPtr() const {
    return m_pMemDesc;
}

void* StaticMemory::getData() const {
    return m_pMemMngr->getRawPtr();
}

size_t StaticMemory::getSize() const {
    return m_size;
}

const Shape& StaticMemory::getShape() const {
    return m_pMemDesc->getShape();
}

const VectorDims& StaticMemory::getStaticDims() const {
    return getShape().getStaticDims();
}

void StaticMemory::redefineDesc(MemoryDescPtr desc) {
    IE_THROW(Unexpected) << "Memory descriptor may not be modified in StaticMemory object";
}

void StaticMemory::load(const IMemory& src, bool ftz) const {
    transferData(src, *this, ftz);
}

MemoryMngrPtr StaticMemory::getMemoryMngr() const {
    return m_pMemMngr;
}

//oneDNN specifics for backward compatibility
dnnl::memory StaticMemory::getPrimitive() const {
    return m_prim;
}

void StaticMemory::nullify() {
    void* dataPtr = getData();
    if (dataPtr != nullptr)
        memset(dataPtr, 0, getSize());
}

StaticMemory::StaticMemoryMngr::StaticMemoryMngr(size_t size) : m_size(size) {
    memMngrImpl.resize(m_size);
}

StaticMemory::StaticMemoryMngr::StaticMemoryMngr(void* data, size_t size) : m_size(size) {
    memMngrImpl.setExtBuff(data, m_size);
}

void* StaticMemory::StaticMemoryMngr::getRawPtr() const noexcept {
    return memMngrImpl.getRawPtr();
}

void StaticMemory::StaticMemoryMngr::setExtBuff(void* ptr, size_t size) {
    IE_THROW(Unexpected) << "StaticMemoryMngr may not be modified";
}

bool StaticMemory::StaticMemoryMngr::resize(size_t size) {
    if (size != m_size) {
        IE_THROW(Unexpected) << "StaticMemoryMngr may not resize the memory";
    }
    return false;
}

bool StaticMemory::StaticMemoryMngr::hasExtBuffer() const noexcept {
    return memMngrImpl.hasExtBuffer();
}

void StaticMemory::StaticMemoryMngr::registerMemory(Memory* memPtr) {
    //do nothing
}

void StaticMemory::StaticMemoryMngr::unregisterMemory(Memory* memPtr) {
    //do nothing
}
}   // namespace intel_cpu
}   // namespace ov
