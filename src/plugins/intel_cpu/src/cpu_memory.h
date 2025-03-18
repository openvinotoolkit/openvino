// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu_shape.h>
#include <onednn/dnnl.h>

#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>

#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"

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
class ProxyMemoryBlock;

/**
 * @interface IMemoryBlock
 * @brief An interface to memory control object
 */

class IMemoryBlock {
public:
    virtual ~IMemoryBlock() = default;

    /**
     * @brief Accessor to underlying memory buffer
     * @return A pointer to underlying memory
     */
    virtual void* getRawPtr() const noexcept = 0;

    /**
     * @brief Allows to set externally allocated memory buffer. In that case, the object has no control over the
     * provided memory.
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
 * @brief An implementation of the mem block where memory reallocation occurs only if a bigger buffer is requested.
 */
class MemoryBlockWithReuse : public IMemoryBlock {
public:
    MemoryBlockWithReuse(int numa_node = -1) : m_data(nullptr, release), numa_node(numa_node) {}
    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;
    void free();
    size_t size() const;  // in bytes

private:
    bool m_useExternalStorage = false;
    size_t m_memUpperBound = 0ul;
    std::unique_ptr<void, void (*)(void*)> m_data;
    int numa_node;

    static void release(void* ptr);
    static void destroy(void* ptr);
};

using MemoryBlockPtr = std::shared_ptr<IMemoryBlock>;
using MemoryBlockCPtr = std::shared_ptr<const IMemoryBlock>;

class IMemory {
public:
    virtual ~IMemory() = default;

    virtual const MemoryDesc& getDesc() const = 0;
    virtual MemoryDescPtr getDescPtr() const = 0;

    virtual void* getData() const = 0;  // pointer to the actual memory

    template <typename T, typename datatype = typename std::decay<T>::type>
    T* getDataAs() const {
        /** @todo enabling this check requires all the nodes to follow this requirement
         * OPENVINO_ASSERT(element::from<datatype>() == getPrecision(),
         * "Memory data element type ", getPrecision(), " is not representable as ", element::from<datatype>());
         */
        return static_cast<T*>(getData());
    }

    virtual size_t getSize() const = 0;  // in bytes
    virtual const Shape& getShape() const = 0;
    virtual const VectorDims& getStaticDims() const = 0;

    // Redefines descriptor. The memory descriptor will be replaced with the new one.
    // Memory will not be reallocated according to the dynamic memory block policy
    // Caution!!! This action invalidates the previous data layout. The old data may become unreachable.
    virtual void redefineDesc(MemoryDescPtr desc) = 0;

    virtual void load(const IMemory& src, bool ftz, bool bf16saturation) const = 0;

    virtual MemoryBlockPtr getMemoryBlock() const = 0;

    virtual void nullify() = 0;

    bool isDefined() const noexcept {
        if (auto desc = getDescPtr()) {
            return desc->isDefined();
        }
        return false;
    }

    ov::element::Type getPrecision() const {
        return getDesc().getPrecision();
    }

    template <typename T,
              typename std::enable_if<!std::is_pointer<T>::value && !std::is_reference<T>::value, int>::type = 0,
              typename std::enable_if<std::is_base_of<MemoryDesc, T>::value, int>::type = 0>
    std::shared_ptr<T> getDescWithType() const;
};

class StaticMemory final : public IMemory {
public:
    class StaticMemoryBlock : public IMemoryBlock {
    public:
        explicit StaticMemoryBlock(size_t size);
        StaticMemoryBlock(void* data, size_t size);
        void* getRawPtr() const noexcept override;
        void setExtBuff(void* ptr, size_t size) override;
        bool resize(size_t size) override;
        bool hasExtBuffer() const noexcept override;

    private:
        size_t m_size = 0;
        MemoryBlockWithReuse memBlockImpl;
    };

    using MemBlockPtr = std::shared_ptr<StaticMemoryBlock>;

public:
    StaticMemory(MemoryDescPtr desc, const void* data = nullptr, bool pads_zeroing = true);
    StaticMemory(const MemoryDesc& desc, const void* data = nullptr, bool pads_zeroing = true);

    StaticMemory(const StaticMemory&) = delete;
    StaticMemory& operator=(const StaticMemory&) = delete;

    StaticMemory(Memory&&) = delete;
    StaticMemory& operator=(StaticMemory&&) = delete;

    const MemoryDesc& getDesc() const override;
    MemoryDescPtr getDescPtr() const override;

    void* getData() const override;  // pointer to the actual memory

    size_t getSize() const override;  // in bytes
    const Shape& getShape() const override;
    const VectorDims& getStaticDims() const override;

    // Always throws since a static memory descriptor should not be modified
    void redefineDesc(MemoryDescPtr desc) override;

    void load(const IMemory& src, bool ftz, bool bf16saturation) const override;

    MemoryBlockPtr getMemoryBlock() const override;

    void nullify() override;

private:
    MemoryDescPtr m_pMemDesc;
    MemBlockPtr m_pMemBlock;
    size_t m_size;
};

class Memory : public IMemory {
public:
    Memory(MemoryDescPtr desc, const void* data = nullptr, bool pads_zeroing = true);
    Memory(const MemoryDesc& desc, const void* data = nullptr, bool pads_zeroing = true);
    Memory(MemoryDescPtr desc, MemoryBlockPtr block);
    Memory(const MemoryDesc& desc, MemoryBlockPtr block);

    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    Memory(Memory&&) = delete;
    Memory& operator=(Memory&&) = delete;

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

    void redefineDesc(MemoryDescPtr desc) override;

    void load(const IMemory& src, bool ftz, bool bf16saturation) const override;
    void nullify() override;

    MemoryBlockPtr getMemoryBlock() const override {
        return m_memBlock;
    }

private:
    friend ProxyMemoryBlock;

private:
    void create(const MemoryDesc& desc, const void* data = nullptr, bool pads_zeroing = true);
    void create(MemoryDescPtr desc, const void* data = nullptr, bool pads_zeroing = true);
    void* getDataNoThrow() const noexcept {
        return m_memBlock->getRawPtr();
    }

private:
    MemoryDescPtr m_pMemDesc;
    MemoryBlockPtr m_memBlock;
    bool m_padsZeroing = true;
};

class StringMemory : public IMemory {
public:
    using OvString = ov::element_type_traits<ov::element::string>::value_type;

    class StringMemoryBlock {
    public:
        StringMemoryBlock() : m_data(nullptr, release) {}
        OvString* getStringPtr() const noexcept;
        void setExtBuff(OvString* ptr, size_t size);
        size_t getStrLen() const noexcept;
        void* getRawPtr() const noexcept;
        bool resize(size_t size /* string elements number */);
        bool hasExtBuffer() const noexcept;

    private:
        bool m_use_external_storage = false;
        size_t m_str_upper_bound = 0lu;
        std::unique_ptr<OvString, void (*)(OvString*)> m_data;

        static void release(OvString* ptr) {}
        static void destroy(OvString* ptr);
    };

    using StringMemoryBlockPtr = std::shared_ptr<StringMemoryBlock>;

    StringMemory(MemoryDescPtr desc, const void* data = nullptr);

    StringMemory(const MemoryDesc& desc, const void* data = nullptr)
        : StringMemory(desc.clone(), data) {}

    StringMemory(MemoryDescPtr desc, StringMemoryBlockPtr block)
        : m_mem_desc(std::move(desc)),
          m_memoryBlock(std::move(block)) {}

    StringMemory(const MemoryDesc& desc, StringMemoryBlockPtr block)
        : StringMemory(desc.clone(), std::move(block)) {}

    const MemoryDesc& getDesc() const override {
        return *m_mem_desc;
    }

    MemoryDescPtr getDescPtr() const override {
        return m_mem_desc;
    }

    void* getData() const override;

    size_t getSize() const override;  // In bytes

    const Shape& getShape() const override {
        return m_mem_desc->getShape();
    }

    const VectorDims& getStaticDims() const override {
        return m_mem_desc->getShape().getStaticDims();
    }

    void redefineDesc(MemoryDescPtr desc) override;

    void load(const IMemory& src, bool ftz, bool bf16saturation) const override;

    MemoryBlockPtr getMemoryBlock() const override;

    StringMemoryBlockPtr getStringMemoryBlockPtr() const {
        return m_memoryBlock;
    }

    void nullify() override;

private:
    MemoryDescPtr m_mem_desc;
    StringMemoryBlockPtr m_memoryBlock;
};

using MemoryPtr = std::shared_ptr<IMemory>;
using MemoryCPtr = std::shared_ptr<const IMemory>;
using StringMemoryPtr = std::shared_ptr<StringMemory>;

bool mbind_move(void* data, size_t size, int numaNodeID);
bool mbind_move(const MemoryCPtr& mem, int numaNodeID);
bool mbind_move(const dnnl::memory& mem, int numaNodeID);

MemoryPtr split_horizontal(const MemoryPtr& src,
                           int dim,
                           int w_rank,
                           int w_size,
                           bool need_fill = true);
MemoryPtr split_vertical(const MemoryPtr& src,
                         int dim,
                         int w_rank,
                         int w_size,
                         bool need_fill = true);

}  // namespace intel_cpu
}  // namespace ov
