// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_memory.h"

#include <common/memory_desc_wrapper.hpp>

#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/reorder.h"
#include "utils/bfloat16.hpp"
#include "utils/debug_capabilities.h"
#if defined(__linux__)
#    include <sys/syscall.h> /* Definition of SYS_* constants */
#    include <unistd.h>

#    include <cstring> /* strerror(errno) */
#    include <utility>
#endif

namespace ov::intel_cpu {
template <>
DnnlMemoryDescPtr IMemory::getDescWithType<DnnlMemoryDesc, 0, 0>() const {
    return MemoryDescUtils::convertToDnnlMemoryDesc(getDescPtr());
}

template <>
BlockedMemoryDescPtr IMemory::getDescWithType<BlockedMemoryDesc, 0, 0>() const {
    return MemoryDescUtils::convertToBlockedMemoryDesc(getDescPtr());
}

namespace {
inline void setSubnormalsToZeroAndbf16Saturation(float* data, size_t size, bool ftz, bool bf16saturation) {
    auto* u32data = reinterpret_cast<uint32_t*>(data);
    auto* floatdata = reinterpret_cast<float*>(data);
    if (ftz && bf16saturation) {
        for (size_t i = 0; i < size; ++i) {
            if ((u32data[i] & (0xFF << 23)) == 0) {
                u32data[i] = 0;
            } else if (!std::isnan(floatdata[i]) && !std::isinf(floatdata[i])) {
                floatdata[i] = (floatdata[i] < static_cast<float>(std::numeric_limits<ov::bfloat16>::lowest()))
                                   ? static_cast<float>(std::numeric_limits<ov::bfloat16>::lowest())
                               : (floatdata[i] > static_cast<float>(std::numeric_limits<ov::bfloat16>::max()))
                                   ? static_cast<float>(std::numeric_limits<ov::bfloat16>::max())
                                   : floatdata[i];
            }
        }
    } else if (ftz) {
        for (size_t i = 0; i < size; ++i) {
            if ((u32data[i] & (0xFF << 23)) == 0) {
                u32data[i] = 0;
            }
        }
    } else if (bf16saturation) {
        for (size_t i = 0; i < size; ++i) {
            if (!std::isnan(floatdata[i]) && !std::isinf(floatdata[i])) {
                floatdata[i] = (floatdata[i] < static_cast<float>(std::numeric_limits<ov::bfloat16>::lowest()))
                                   ? static_cast<float>(std::numeric_limits<ov::bfloat16>::lowest())
                               : (floatdata[i] > static_cast<float>(std::numeric_limits<ov::bfloat16>::max()))
                                   ? static_cast<float>(std::numeric_limits<ov::bfloat16>::max())
                                   : floatdata[i];
            }
        }
    }
}

void transferData(const IMemory& src, const IMemory& dst, bool ftz, bool bf16saturation) {
    node::Reorder::reorderData(src, dst);

    if (!ftz && !bf16saturation) {
        return;
    }
    if (src.getDesc().getPrecision() != ov::element::f32 || dst.getDesc().getPrecision() != ov::element::f32) {
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
    setSubnormalsToZeroAndbf16Saturation(memData, dst.getSize() / sizeof(float), ftz, bf16saturation);
}

}  // namespace

Memory::Memory(dnnl::engine eng, MemoryDescPtr desc, const void* data, bool pads_zeroing)
    : m_eng(std::move(eng)),
      m_pMemDesc(std::move(desc)),
      m_blockHandle(std::make_shared<DnnlMemoryBlock>(make_unique<MemoryBlockWithReuse>()), this),
      dnnlMemHandle(this) {
    if (m_pMemDesc->getPrecision() == element::string) {
        OPENVINO_THROW("[CPU] Memory object cannot be created for string data.");
    }
    create(m_pMemDesc, data, pads_zeroing);
}

Memory::Memory(dnnl::engine eng, const MemoryDesc& desc, const void* data, bool pads_zeroing)
    : Memory::Memory(std::move(eng), desc.clone(), data, pads_zeroing) {}

Memory::Memory(dnnl::engine eng, MemoryDescPtr desc, MemoryBlockPtr block)
    : m_eng(std::move(eng)),
      m_pMemDesc(std::move(desc)),
      m_blockHandle(std::move(block), this),
      dnnlMemHandle(this) {
    if (m_pMemDesc->getPrecision() == element::string) {
        OPENVINO_THROW("[CPU] Memory object can't be created for string data.");
    }
    bool memAllocated = m_blockHandle->getRawPtr();

    create(m_pMemDesc, nullptr, !memAllocated);
}

Memory::Memory(dnnl::engine eng, const MemoryDesc& desc, MemoryBlockPtr block)
    : Memory::Memory(std::move(eng), desc.clone(), std::move(block)) {}

size_t Memory::getSize() const {
    auto size = getDesc().getCurrentMemSize();
    if (size == MemoryDesc::UNDEFINED_SIZE) {
        OPENVINO_THROW("Can't get memory size for undefined shape");
    }
    return size;
}

void Memory::create(const MemoryDesc& desc, const void* data, bool pads_zeroing) {
    create(desc.clone(), data, pads_zeroing);
}

void Memory::create(MemoryDescPtr desc, const void* data, bool pads_zeroing) {
    m_pMemDesc = std::move(desc);
    m_padsZeroing = pads_zeroing;
    dnnlMemHandle.resetDnnlPrim();

    if (!m_pMemDesc->isDefined()) {
        return;
    }
    auto memSize = m_pMemDesc->getCurrentMemSize();
    if (nullptr != data) {
        m_blockHandle->setExtBuff(const_cast<void*>(data), memSize);
    } else {
        m_blockHandle->resize(memSize);
    }
}

void Memory::load(const IMemory& src, bool ftz, bool bf16saturation) const {
    if (src.getDesc().getPrecision() == element::string) {
        OPENVINO_THROW("[CPU] Memory object cannot load string data.");
    }
    transferData(src, *this, ftz, bf16saturation);
}

void Memory::nullify() {
    void* dataPtr = getData();
    if (dataPtr != nullptr) {
        memset(dataPtr, 0, getDesc().getCurrentMemSize());
    }
}

void Memory::redefineDesc(MemoryDescPtr desc) {
    if (desc->getPrecision() == element::string) {
        OPENVINO_THROW("[CPU] Memory object cannot accept a descriptor with a string type.");
    }
    if (!desc->hasDefinedMaxSize()) {
        OPENVINO_THROW("Can not reset descriptor, memory upper bound is unknown.");
    }

    this->create(desc, nullptr, false);
}

void Memory::update() {
    if (dnnlMemHandle.isInit()) {
        auto prim = dnnlMemHandle.getPrim();
        prim.set_data_handle(m_blockHandle->getRawPtr());
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
            OPENVINO_THROW("Can not create oneDNN memory from undefined memory descriptor");
        }

        // ========================
        // Equivalent of constructor memory(const primitive_desc &desc, void *hdl)
        // but with ability to skip pads zeroing.
        auto desc = MemoryDescUtils::convertToDnnlMemoryDesc(m_memObjPtr->getDescPtr());
        m_prim = dnnl::memory(desc->getDnnlDesc(), m_memObjPtr->getEngine(), DNNL_MEMORY_NONE);
        //
        // ========================
        auto data = m_memObjPtr->getDataNoThrow();
        if (data != nullptr) {
            m_prim.set_data_handle(data);
        }
    }
    return m_prim;
}

void* Memory::getData() const {
    void* data = getDataNoThrow();
    if (data == nullptr && m_pMemDesc->getShape().isStatic() && m_pMemDesc->getShape().getElementsCount() != 0) {
        OPENVINO_THROW("Memory has not been allocated");
    }
    return data;
}

void* MemoryBlockWithReuse::getRawPtr() const noexcept {
    return m_data.get();
}

void MemoryBlockWithReuse::setExtBuff(void* ptr, size_t size) {
    m_useExternalStorage = true;
    m_memUpperBound = size;
    m_data = decltype(m_data)(ptr, release);
}

bool MemoryBlockWithReuse::resize(size_t size) {
    constexpr int cacheLineSize = 64;
    bool sizeChanged = false;
    if (size > m_memUpperBound) {
        void* ptr = dnnl::impl::malloc(size, cacheLineSize);
        if (!ptr) {
            OPENVINO_THROW("Failed to allocate ", size, " bytes of memory");
        }
        m_memUpperBound = size;
        m_useExternalStorage = false;
        m_data = decltype(m_data)(ptr, destroy);
        sizeChanged = true;

        if (numa_node >= 0) {
            if (!mbind_move(ptr, size, numa_node)) {
                DEBUG_LOG("MemoryBlockWithReuse move_memory to node ", numa_node, " failed\n");
            }
        }
    }
    return sizeChanged;
}

bool MemoryBlockWithReuse::hasExtBuffer() const noexcept {
    return m_useExternalStorage;
}

void MemoryBlockWithReuse::free() {
    m_data = decltype(m_data)(nullptr, release);
    m_memUpperBound = 0ul;
    m_useExternalStorage = false;
}

size_t MemoryBlockWithReuse::size() const {
    return m_memUpperBound;
}

void MemoryBlockWithReuse::release(void* ptr) {}

void MemoryBlockWithReuse::destroy(void* ptr) {
    dnnl::impl::free(ptr);
}

/////////////// StringMemory ///////////////

StringMemory::StringMemory(dnnl::engine engine, MemoryDescPtr desc, const void* data)
    : m_engine(std::move(engine)),
      m_mem_desc(std::move(desc)) {
    if (m_mem_desc->getPrecision() != element::string) {
        OPENVINO_THROW("[CPU] StringMemory supports String type only.");
    }

    m_memoryBlock = std::make_shared<StringMemoryBlock>();

    if (!m_mem_desc->isDefined()) {
        return;
    }

    const auto string_size = m_mem_desc->getShape().getElementsCount();

    if (data != nullptr) {
        auto not_const_data = const_cast<void*>(data);
        m_memoryBlock->setExtBuff(reinterpret_cast<OvString*>(not_const_data), string_size);
    } else {
        m_memoryBlock->resize(string_size);
    }
}

void StringMemory::load(const IMemory& src, bool ftz, bool bf16saturation) const {
    if (src.getDesc().getPrecision() != element::string) {
        OPENVINO_THROW("[CPU] String memory cannot load a non-string object.");
    }

    transferData(src, *this, false, false);
}

void* StringMemory::getData() const {
    return m_memoryBlock->getRawPtr();
}

void StringMemory::redefineDesc(MemoryDescPtr desc) {
    if (desc->getPrecision() != element::string) {
        OPENVINO_THROW("[CPU] StringMemory supports String type only.");
    }
    if (!desc->hasDefinedMaxSize()) {
        OPENVINO_THROW("[CPU] StringMemory cannot reset descriptor. Memory upper bound is unknown.");
    }

    m_mem_desc = desc;
    const auto string_size = m_mem_desc->getShape().getElementsCount();
    m_memoryBlock->resize(string_size);
}

void StringMemory::nullify() {
    auto data_ptr = m_memoryBlock->getStringPtr();
    if (data_ptr != nullptr) {
        std::fill(data_ptr, data_ptr + m_memoryBlock->getStrLen(), OvString());
    }
}

size_t StringMemory::getSize() const {  // In bytes
    auto size = getDesc().getCurrentMemSize();
    if (size == MemoryDesc::UNDEFINED_SIZE) {
        OPENVINO_THROW("Can't get memory size for undefined shape.");
    }
    return size;
}

MemoryBlockPtr StringMemory::getMemoryBlock() const {
    OPENVINO_THROW("Unexpected call of StringMemory::getMemoryBlock()");
}

dnnl::memory StringMemory::getPrimitive() const {
    OPENVINO_THROW("Unexpected call of StringMemory::getPrimitive()");
}

void StringMemory::StringMemoryBlock::setExtBuff(OvString* ptr, size_t size) {
    m_use_external_storage = true;
    m_str_upper_bound = size;
    m_data = decltype(m_data)(ptr, release);
}

StringMemory::OvString* StringMemory::StringMemoryBlock::getStringPtr() const noexcept {
    return m_data.get();
}

bool StringMemory::StringMemoryBlock::resize(size_t size) {
    bool sizeChanged = false;
    if (size > m_str_upper_bound) {
        if (size > PTRDIFF_MAX) {
            OPENVINO_THROW("Requested allocation size { ", size, " } exceeds PTRDIFF_MAX.");
        }
        auto ptr_size = static_cast<ptrdiff_t>(size);  // WA for warning alloc-size-larger-than
        auto ptr = new OvString[ptr_size];
        if (!ptr) {
            OPENVINO_THROW("Failed to allocate ", size, " bytes of memory");
        }
        m_str_upper_bound = size;
        m_use_external_storage = false;
        m_data = decltype(m_data)(ptr, destroy);
        sizeChanged = true;
    }
    return sizeChanged;
}

bool StringMemory::StringMemoryBlock::hasExtBuffer() const noexcept {
    return m_use_external_storage;
}

size_t StringMemory::StringMemoryBlock::getStrLen() const noexcept {
    return m_str_upper_bound;
}

void StringMemory::StringMemoryBlock::destroy(OvString* ptr) {
    delete[] ptr;
}

void* StringMemory::StringMemoryBlock::getRawPtr() const noexcept {
    return reinterpret_cast<void*>(m_data.get());
}

/////////////// DnnlMemoryBlock ///////////////

void* DnnlMemoryBlock::getRawPtr() const noexcept {
    return m_pMemBlock->getRawPtr();
}

void DnnlMemoryBlock::setExtBuff(void* ptr, size_t size) {
    m_pMemBlock->setExtBuff(ptr, size);
    notifyUpdate();
}

bool DnnlMemoryBlock::resize(size_t size) {
    bool sizeChanged = m_pMemBlock->resize(size);
    if (sizeChanged) {
        notifyUpdate();
    }
    return sizeChanged;
}

bool DnnlMemoryBlock::hasExtBuffer() const noexcept {
    return m_pMemBlock->hasExtBuffer();
}

void DnnlMemoryBlock::registerMemory(Memory* memPtr) {
    if (memPtr) {
        m_setMemPtrs.insert(memPtr);
    }
}

void DnnlMemoryBlock::unregisterMemory(Memory* memPtr) {
    if (memPtr) {
        m_setMemPtrs.erase(memPtr);
    }
}

void DnnlMemoryBlock::notifyUpdate() {
    for (auto& item : m_setMemPtrs) {
        if (item) {
            item->update();
        }
    }
}

StaticMemory::StaticMemory(dnnl::engine eng, MemoryDescPtr desc, const void* data, bool pads_zeroing)
    : m_eng(std::move(eng)),
      m_pMemDesc(std::move(desc)) {
    if (m_pMemDesc->getPrecision() == element::string) {
        OPENVINO_THROW("[CPU] StaticMemory object cannot be created for string data.");
    }
    if (!m_pMemDesc->isDefined()) {
        OPENVINO_THROW("Can not create StaticMemory object. The memory desc is undefined");
    }

    m_size = m_pMemDesc->getCurrentMemSize();

    if (data) {
        m_pMemBlock = std::make_shared<StaticMemoryBlock>(const_cast<void*>(data), m_size);
    } else {
        m_pMemBlock = std::make_shared<StaticMemoryBlock>(m_size);
    }

    try {
        auto dnnl_desc = MemoryDescUtils::convertToDnnlMemoryDesc(m_pMemDesc);
        // Equivalent of constructor memory(const primitive_desc &desc, void *hdl)
        // but with ability to skip pads zeroing.
        const auto& memory_desc = dnnl_desc->getDnnlDesc();
        if (memory_desc.is_zero()) {
            // dnnl memory created using an empty memory_desc is not empty, so use a default constructor
            m_prim = dnnl::memory();
        } else {
            m_prim = dnnl::memory(memory_desc, m_eng, DNNL_MEMORY_NONE);
            m_prim.set_data_handle(m_pMemBlock->getRawPtr());
        }
    } catch (const std::exception& exc) {
        dnnlErrorCtx = exc.what();
    }
}

StaticMemory::StaticMemory(dnnl::engine eng, const MemoryDesc& desc, const void* data, bool pads_zeroing)
    : StaticMemory::StaticMemory(std::move(eng), desc.clone(), data, pads_zeroing) {}

const MemoryDesc& StaticMemory::getDesc() const {
    return *m_pMemDesc;
}

MemoryDescPtr StaticMemory::getDescPtr() const {
    return m_pMemDesc;
}

void* StaticMemory::getData() const {
    return m_pMemBlock->getRawPtr();
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
    OPENVINO_THROW("Unexpected: Memory descriptor may not be modified in StaticMemory object");
}

void StaticMemory::load(const IMemory& src, bool ftz, bool bf16saturation) const {
    if (src.getDesc().getPrecision() == element::string) {
        OPENVINO_THROW("[CPU] StaticMemory cannot load string data.");
    }
    transferData(src, *this, ftz, bf16saturation);
}

MemoryBlockPtr StaticMemory::getMemoryBlock() const {
    return m_pMemBlock;
}

// oneDNN specifics for backward compatibility
dnnl::memory StaticMemory::getPrimitive() const {
    if (!m_prim && !getDesc().empty()) {  // for an empty memory m_prim is expected to be empty
        OPENVINO_THROW("Couldn't create dnnl::memory object: ", dnnlErrorCtx);
    }

    return m_prim;
}

void StaticMemory::nullify() {
    void* dataPtr = getData();
    if (dataPtr != nullptr) {
        memset(dataPtr, 0, getSize());
    }
}

StaticMemory::StaticMemoryBlock::StaticMemoryBlock(size_t size) : m_size(size) {
    memBlockImpl.resize(m_size);
}

StaticMemory::StaticMemoryBlock::StaticMemoryBlock(void* data, size_t size) : m_size(size) {
    memBlockImpl.setExtBuff(data, m_size);
}

void* StaticMemory::StaticMemoryBlock::getRawPtr() const noexcept {
    return memBlockImpl.getRawPtr();
}

void StaticMemory::StaticMemoryBlock::setExtBuff(void* ptr, size_t size) {
    OPENVINO_THROW("Unexpected: StaticMemoryBlock may not be modified");
}

bool StaticMemory::StaticMemoryBlock::resize(size_t size) {
    if (size != m_size) {
        OPENVINO_THROW("Unexpected: StaticMemoryBlock may not resize the memory");
    }
    return false;
}

bool StaticMemory::StaticMemoryBlock::hasExtBuffer() const noexcept {
    return memBlockImpl.hasExtBuffer();
}

void StaticMemory::StaticMemoryBlock::registerMemory(Memory* memPtr) {
    // do nothing
}

void StaticMemory::StaticMemoryBlock::unregisterMemory(Memory* memPtr) {
    // do nothing
}

#if defined(__linux__)
#    define MPOL_DEFAULT   0
#    define MPOL_BIND      2
#    define MPOL_MF_STRICT (1 << 0)
#    define MPOL_MF_MOVE   (1 << 1)
#    if !defined(__NR_mbind) && defined(__x86_64__)
#        define __NR_mbind 237
#    endif
static int64_t mbind(void* start, uint64_t len, int mode, const uint64_t* nmask, uint64_t maxnode, unsigned flags) {
    return syscall(__NR_mbind,
                   reinterpret_cast<uint64_t>(start),
                   len,
                   mode,
                   reinterpret_cast<uint64_t>(nmask),
                   maxnode,
                   flags);
}
#endif

#if defined(__linux__)
bool mbind_move(void* data, size_t size, int targetNode) {
    int realNode = ov::get_org_numa_id(targetNode);
    auto pagesize = getpagesize();
    auto page_count = (size + pagesize - 1) / pagesize;
    auto* pages = reinterpret_cast<char*>(  // NOLINT(performance-no-int-to-ptr)
        ((reinterpret_cast<uintptr_t>(data)) & ~(static_cast<uintptr_t>(pagesize - 1))));
    uint64_t mask = 0;
    unsigned flags = 0;
    if (realNode < 0) {
        // restore default policy
        mask = -1;
        flags = 0;
    } else {
        mask = 1ul << realNode;
        flags = MPOL_MF_MOVE | MPOL_MF_STRICT;
    }

    auto rc = mbind(pages, page_count * pagesize, MPOL_BIND, &mask, sizeof(mask) * 8, flags);
    if (rc < 0) {
        DEBUG_LOG("mbind failed: ", strerror(errno));
        return false;
    }
    return true;
}
#else
bool mbind_move(void* data, size_t size, int targetNode) {
    return false;
}
#endif

bool mbind_move(const MemoryCPtr& mem, int numaNodeID) {
    void* data = mem->getData();
    auto size = mem->getSize();
    return mbind_move(data, size, numaNodeID);
}

bool mbind_move(const dnnl::memory& mem, int numaNodeID) {
    if (!mem) {
        return true;
    }

    void* data = mem.get_data_handle();
    auto desc = mem.get_desc();
    auto size = desc.get_size();
    return mbind_move(data, size, numaNodeID);
}

MemoryPtr split_horizontal(const dnnl::engine& eng,
                           const MemoryPtr& src,
                           int dim,
                           int w_rank,
                           int w_size,
                           bool need_fill) {
    auto desc = src->getDescPtr();
    auto shape = src->getShape();
    const auto& dims = shape.getDims();
    auto prec = src->getPrecision();
    if (dim < 0) {
        dim += dims.size();
    }
    auto split_parts = [](int len, int n) {
        int average = len / n;
        std::vector<int> parts(n, average);
        parts.back() = len - average * (n - 1);
        return parts;
    };
    if (shape.isDynamic()) {
        // if the dim is dynamic, should return a dynamic dim without any change.
        // if the dim is static, should split it indeed.
        const auto& pshape = shape.toPartialShape();
        if (pshape[dim].is_dynamic()) {
            return src;
        }
        auto new_pshape = pshape;
        auto splited_dim_vec = split_parts(new_pshape[dim].get_length(), w_size);
        new_pshape[dim] = splited_dim_vec[w_rank];

        auto new_desc = std::make_shared<CpuBlockedMemoryDesc>(prec, Shape{new_pshape});
        MemoryPtr ptr = std::make_shared<Memory>(eng, new_desc);
        return ptr;
    }
    assert(static_cast<int>(dims[dim]) >= w_size);
    auto splited_dim_vec = split_parts(dims[dim], w_size);

    // reference stride
    VectorDims stride_dims = dims;
    stride_dims[dim] = splited_dim_vec[0];
    size_t stride =
        std::accumulate(stride_dims.begin(), stride_dims.end(), static_cast<size_t>(1), std::multiplies<>()) *
        prec.size();

    // create new shape for target memory
    VectorDims new_dims = dims;
    new_dims[dim] = splited_dim_vec[w_rank];

    auto new_desc = desc->cloneWithNewDims(new_dims, true);
    if (!need_fill) {
        MemoryPtr ptr = std::make_shared<Memory>(eng, new_desc, nullptr);
        return ptr;
    }

    auto srcPtr = static_cast<uint8_t*>(src->getData());
    if (prec == ov::element::u4 || prec == ov::element::i4) {
        stride /= 2;
    }

    MemoryPtr ptr = std::make_shared<Memory>(eng, new_desc, srcPtr + w_rank * stride);
    return ptr;
}

MemoryPtr split_vertical(const dnnl::engine& eng,
                         const MemoryPtr& src,
                         int dim,
                         int w_rank,
                         int w_size,
                         bool need_fill) {
    auto desc = src->getDescPtr();
    const auto& shape = src->getShape();
    const auto& dims = shape.getDims();
    auto prec = src->getPrecision();
    if (dim < 0) {
        dim += dims.size();
    }
    auto split_parts = [](int len, int n) {
        int average = len / n;
        std::vector<int> parts(n, average);
        parts.back() = len - average * (n - 1);
        return parts;
    };
    if (shape.isDynamic()) {
        const auto& pshape = shape.toPartialShape();
        if (pshape[dim].is_dynamic()) {
            OPENVINO_THROW("Can't split data with dynamic shapes");
        }
        auto new_pshape = pshape;
        auto splited_dim_vec = split_parts(new_pshape[dim].get_length(), w_size);
        new_pshape[dim] = splited_dim_vec[w_rank];

        auto new_desc = std::make_shared<CpuBlockedMemoryDesc>(prec, Shape{new_pshape});
        MemoryPtr ptr = std::make_shared<Memory>(eng, new_desc);
        return ptr;
    }
    assert(static_cast<int>(dims[dim]) >= w_size);
    const auto splited_size = dims[dim] * prec.size();
    auto splited_dim_vec = split_parts(dims[dim], w_size);
    auto element_size = prec.size();

    VectorDims new_dims = dims;
    new_dims[dim] = splited_dim_vec[w_rank];

    auto new_desc = desc->cloneWithNewDims(new_dims, true);
    MemoryPtr ptr = std::make_shared<Memory>(eng, new_desc);
    if (!need_fill) {
        return ptr;
    }
    // copy
    auto srcPtr = static_cast<uint8_t*>(src->getData());
    auto dstPtr = static_cast<uint8_t*>(ptr->getData());
    // selected dim bytes
    auto channel_size = dims[dim] * element_size;
    // total bytes
    auto mem_size = src->getSize();
    // the steps need to copy.
    const int step = (mem_size / channel_size);
    // bytes of selected dim.
    auto strideSize = splited_dim_vec[0] * element_size;
    auto copySize = splited_dim_vec[w_rank] * element_size;
    if (prec == ov::element::u4 || prec == ov::element::i4) {
        strideSize /= 2;
        copySize /= 2;
    }
    parallel_for(step, [&](int i) {
        int dst_offset = i * copySize;
        int src_offset = i * splited_size + w_rank * strideSize;
        cpu_parallel_memcpy(dstPtr + dst_offset, srcPtr + src_offset, copySize);
    });
    return ptr;
}

}  // namespace ov::intel_cpu
