// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_layouts.h"
#include "memory_desc/cpu_memory_desc.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include <onednn/dnnl.h>
#include <cpu_shape.h>

#include "memory_desc/dnnl_memory_desc.h"

#include <string>
#include <functional>
#include <memory>
#include <vector>

/**
 * @file contains a concept classes to work with memory/tensor/blob abstractions on plugin level.
 *
 * Memory is an abstraction of some real tensor which contains some data. As in short it's a pair of
 * memory descriptor and raw buffer handler to contains data. In case of system memory raw buffer it's simple
 * "void*" on some system memory buffer.
 *
 */

namespace ov {
namespace intel_cpu {

class Memory;
class ProxyMemoryMngr;

/**
 * @interface IMemoryMngr
 * @brief An interface to memory control object
 */

class IMemoryMngr {
public:
    virtual ~IMemoryMngr() = default;

    /**
     * @brief Accessor to underlying memory buffer
     * @return A pointer to underlying memory
     */
    virtual void* getRawPtr() const noexcept = 0;

    /**
     * @brief Allows to set externally allocated memory buffer. In that case, the object has no control over the provided memory.
     * @param ptr - pointer to the memory
     * @param size - size of the memory buffer
     */
    virtual void setExtBuff(void* ptr, size_t size) = 0;

    /**
     * @brief Resize underlying memory buffer
     * @param size - new memory size in bytes
     * @return status whether the memory reallocation was performed
     */
    virtual bool resize(size_t size) = 0;

    /**
     * @brief Check if the object has control over underlying memory buffer
     * @return status whether the object has control over underlying memory buffer
     */
    virtual bool hasExtBuffer() const noexcept = 0;
};

/**
 * @brief An implementation of the mem manager where memory reallocation occurs only if a bigger buffer is requested.
 */
class MemoryMngrWithReuse : public IMemoryMngr {
public:
    MemoryMngrWithReuse() : m_data(nullptr, release) {}
    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;

private:
    bool m_useExternalStorage = false;
    size_t m_memUpperBound = 0ul;
    std::unique_ptr<void, void (*)(void *)> m_data;

    static void release(void *ptr);
    static void destroy(void *ptr);
};

class IMemoryMngrObserver : public IMemoryMngr {
public:
    virtual void registerMemory(Memory* memPtr) = 0;
    virtual void unregisterMemory(Memory* memPtr) = 0;
};

/**
 * @brief A proxy object that additionally implements observer pattern
 */
class DnnlMemoryMngr : public IMemoryMngrObserver {
public:
    explicit DnnlMemoryMngr(std::unique_ptr<IMemoryMngr> mngr) : m_pMemMngr(std::move(mngr)) {}
    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;
    void registerMemory(Memory* memPtr) override;
    void unregisterMemory(Memory* memPtr) override;

private:
    void notifyUpdate();

private:
    std::unordered_set<Memory*> m_setMemPtrs;
    std::unique_ptr<IMemoryMngr> m_pMemMngr;
};

using MemoryMngrPtr = std::shared_ptr<IMemoryMngrObserver>;
using MemoryMngrCPtr = std::shared_ptr<const IMemoryMngrObserver>;

class DnnlMemMngrHandle {
public:
    DnnlMemMngrHandle(MemoryMngrPtr pMgr, Memory* pMem) : m_pMgr(pMgr), m_pMem(pMem) {
        if (m_pMgr) {
            m_pMgr->registerMemory(m_pMem);
        }
    }

    DnnlMemMngrHandle(const DnnlMemMngrHandle&) = delete;
    DnnlMemMngrHandle& operator= (const DnnlMemMngrHandle&) = delete;

    DnnlMemMngrHandle(DnnlMemMngrHandle&& source) {
        std::swap(m_pMgr, source.m_pMgr);
        std::swap(m_pMem, source.m_pMem);
    }
    DnnlMemMngrHandle& operator= (DnnlMemMngrHandle&& rhs) {
        std::swap(m_pMgr, rhs.m_pMgr);
        std::swap(m_pMem, rhs.m_pMem);
        return *this;
    }

    ~DnnlMemMngrHandle() {
        if (m_pMgr) {
            m_pMgr->unregisterMemory(m_pMem);
        }
    }

    MemoryMngrPtr get() const {
        return m_pMgr;
    }

    MemoryMngrPtr::element_type* operator->() const noexcept {
        return m_pMgr.get();
    }

private:
    MemoryMngrPtr m_pMgr = nullptr;
    Memory* m_pMem = nullptr;
};

class IMemory {
public:
    virtual ~IMemory() = default;

    virtual bool isAllocated() const noexcept = 0;

    virtual const MemoryDesc& getDesc() const = 0;
    virtual MemoryDescPtr getDescPtr() const = 0;

    virtual void* getData() const = 0; // pointer to the actual memory

    virtual size_t getSize() const = 0; // in bytes
    virtual const Shape& getShape() const = 0;
    virtual const VectorDims& getStaticDims() const = 0;

    // Redefines descriptor. The memory descriptor will be replaced with the new one.
    // Memory will not be reallocated if the new tensor size is less or equal the upper bound.
    // Caution!!! This action invalidates the previous data layout. The old data may become unreachable.
    virtual void redefineDesc(MemoryDescPtr desc) = 0;

    virtual void load(const IMemory& src, bool ftz = true) const = 0;

    virtual MemoryMngrPtr getMemoryMngr() const = 0;

    //oneDNN specifics for backward compatibility
    virtual dnnl::memory getPrimitive() const = 0;
    dnnl::memory::data_type getDataType() const {
        return DnnlExtensionUtils::ElementTypeToDataType(getDesc().getPrecision());
    }

    virtual void nullify() = 0;

    template <typename T,
            typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
            typename std::enable_if<std::is_base_of<MemoryDesc, T>::value, int>::type = 0>
    std::shared_ptr<T> getDescWithType() const;
};

class StaticMemory final : public IMemory {
public:
    class StaticMemoryMngr : public IMemoryMngrObserver {
    public:
        explicit StaticMemoryMngr(size_t size);
        StaticMemoryMngr(void* data, size_t size);
        void* getRawPtr() const noexcept override;
        void setExtBuff(void* ptr, size_t size) override;
        bool resize(size_t size) override;
        bool hasExtBuffer() const noexcept override;
        void registerMemory(Memory* memPtr) override;
        void unregisterMemory(Memory* memPtr) override;

    private:
        size_t m_size = 0;
        MemoryMngrWithReuse memMngrImpl;
    };

    using MemMngrPtr = std::shared_ptr<StaticMemoryMngr>;

public:
    StaticMemory(const dnnl::engine& eng, MemoryDescPtr desc, const void* data = nullptr, bool pads_zeroing = true);
    StaticMemory(const dnnl::engine& eng, const MemoryDesc& desc, const void* data = nullptr, bool pads_zeroing = true);

    StaticMemory(const StaticMemory&) = delete;
    StaticMemory& operator= (const StaticMemory&) = delete;

    StaticMemory(Memory&&) = delete;
    StaticMemory& operator= (StaticMemory&&) = delete;

    bool isAllocated() const noexcept override;

    const MemoryDesc& getDesc() const override;
    MemoryDescPtr getDescPtr() const override;

    void* getData() const override; // pointer to the actual memory

    size_t getSize() const override; // in bytes
    const Shape& getShape() const override;
    const VectorDims& getStaticDims() const override;

    // Always throws since a static memory descriptor should not be modified
    void redefineDesc(MemoryDescPtr desc) override;

    void load(const IMemory& src, bool ftz = true) const override;

    MemoryMngrPtr getMemoryMngr() const override;

    //oneDNN specifics for backward compatibility
    dnnl::memory getPrimitive() const override;

    void nullify() override;

private:
    dnnl::engine m_eng;
    MemoryDescPtr m_pMemDesc;
    size_t m_size;
    dnnl::memory m_prim;
    MemMngrPtr m_pMemMngr;
    std::string dnnlErrorCtx;
};

class Memory : public IMemory {
public:
    Memory(const dnnl::engine& eng, MemoryDescPtr desc, const void* data = nullptr, bool pads_zeroing = true);
    Memory(const dnnl::engine& eng, const MemoryDesc& desc, const void* data = nullptr, bool pads_zeroing = true);
    Memory(const dnnl::engine& eng, MemoryDescPtr desc, MemoryMngrPtr mngr);
    Memory(const dnnl::engine& eng, const MemoryDesc& desc, MemoryMngrPtr mbgr);

    Memory(const Memory&) = delete;
    Memory& operator= (const Memory&) = delete;

    Memory(Memory&&) = delete;
    Memory& operator= (Memory&&) = delete;

    dnnl::memory getPrimitive() const override;

    bool isAllocated() const noexcept override;

    const MemoryDesc& getDesc() const override {
        return *m_pMemDesc;
    }

    MemoryDescPtr getDescPtr() const override {
        return m_pMemDesc;
    }

    void* getData() const override;

    size_t getSize() const override;

    const Shape& getShape() const override {
        return getDesc().getShape();
    }

    const VectorDims& getStaticDims() const override {
        return getDesc().getShape().getStaticDims();
    }

    // Redefines descriptor. The memory descriptor will be replaced with the new one.
    // Memory will not be reallocated if the new tensor size is less or equal the upper bound.
    // Caution!!! This action invalidates the previous data layout. The old data may become unreachable.
    void redefineDesc(MemoryDescPtr desc) override;

    void load(const IMemory& src, bool ftz = true) const override;
    void nullify() override;

    dnnl::engine getEngine() const {
        return m_eng;
    }

    MemoryMngrPtr getMemoryMngr() const override {
        return m_mgrHandle.get();
    }

private:
    friend DnnlMemoryMngr;
    friend ProxyMemoryMngr;

private:
    void update();

    void create(const MemoryDesc& desc, const void* data = nullptr, bool pads_zeroing = true);
    void create(MemoryDescPtr desc, const void* data = nullptr, bool pads_zeroing = true);

private:
    dnnl::engine m_eng;
    MemoryDescPtr m_pMemDesc;
    DnnlMemMngrHandle m_mgrHandle;
    bool m_padsZeroing = true;
    class DnnlMemPrimHandle {
    public:
        explicit DnnlMemPrimHandle(const Memory* memObjPtr): m_memObjPtr(memObjPtr) {}
        bool isInit() const;
        dnnl::memory getPrim() const;
        void resetDnnlPrim();

    private:
        // Since getPrim should behave as a constant method, even though it changes state, it must be thread safe.
        // To provide thead safety we use this mutex
        mutable std::mutex m_primCachingLock;
        mutable dnnl::memory m_prim;
        const Memory* m_memObjPtr;
    } dnnlMemHandle;

    void* getDataNoThrow() const noexcept {
        return m_mgrHandle->getRawPtr();
    }
};

using MemoryPtr = std::shared_ptr<IMemory>;
using MemoryCPtr = std::shared_ptr<const IMemory>;

}   // namespace intel_cpu
}   // namespace ov
