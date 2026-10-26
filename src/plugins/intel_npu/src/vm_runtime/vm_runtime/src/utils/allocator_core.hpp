//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vm_export.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

// Core allocator abstraction. `Block` is the currency every allocator hands out; `AnyAllocator` is the
// type-erased handle BufferManager holds over any value type modeling the Allocator concept:
//     Block allocate(size_t bytes);   // null Block on failure; on success Block.size == bytes exactly --
//                                      // BufferManager budgets and bounds buffers by it, so never round up
//     void  deallocate(Block block);  // noexcept; per-block release (may be a no-op)
//     bool  owns(Block block);        // noexcept (const); true if this allocator would release the block
//     void  recycle();                // noexcept; reclaim everything for reuse, keeping memory
//     void  release();                // noexcept; free all underlying memory
//
// composable_allocators.hpp splits this into two roles so neither carries a no-op release(): a Backing owns
// OS/device memory and hands out whole chunks (HostAllocator, LevelZeroAllocator); a SubAllocator carves one
// borrowed chunk into sub-blocks for routing but no release(). GrowingArena ties a Backing to a per-chunk
// SubAllocator into a full Allocator, and routes deallocations through SubAllocator::owns(). Each role
// advertises a `static constexpr size_t ALIGNMENT`: a Backing's chunk-base guarantee, a SubAllocator's
// sub-block stride.

namespace intel_npu::vm {

struct Block {
    uint8_t* ptr = nullptr;
    size_t size = 0;
    explicit operator bool() const noexcept {
        return ptr != nullptr;
    }
};

// Type-erased, move-only owning handle for any Allocator value type, modeled on std::any/std::function:
// the concrete allocator is stored behind a private interface so BufferManager need not be templated.
class NPU_VM_EXPORT AnyAllocator {
    class Concept {
    public:
        Concept() = default;
        Concept(const Concept&) = delete;
        Concept& operator=(const Concept&) = delete;
        Concept(Concept&&) = delete;
        Concept& operator=(Concept&&) = delete;
        virtual ~Concept() = default;
        virtual Block allocate(size_t bytes) = 0;
        virtual void deallocate(Block block) noexcept = 0;
        virtual bool owns(Block block) const noexcept = 0;
        virtual void recycle() noexcept = 0;
        virtual void release() noexcept = 0;
    };

    template <class A>
    class Model final : public Concept {
        A _allocator;

    public:
        explicit Model(A allocator): _allocator(std::move(allocator)) {
        }
        Block allocate(size_t bytes) override {
            return _allocator.allocate(bytes);
        }
        void deallocate(Block block) noexcept override {
            _allocator.deallocate(block);
        }
        bool owns(Block block) const noexcept override {
            return _allocator.owns(block);
        }
        void recycle() noexcept override {
            _allocator.recycle();
        }
        void release() noexcept override {
            _allocator.release();
        }
    };

    std::unique_ptr<Concept> _impl;

public:
    // Empty handle; converts to false. BufferManager rejects an empty allocator.
    AnyAllocator() = default;

    // Constrained so this never shadows the move constructor.
    template <class A, class = std::enable_if_t<!std::is_same_v<std::decay_t<A>, AnyAllocator>>>
    AnyAllocator(A allocator): _impl(std::make_unique<Model<std::decay_t<A>>>(std::move(allocator))) {
    }

    Block allocate(size_t bytes) {
        return _impl ? _impl->allocate(bytes) : Block{};
    }
    void deallocate(Block block) noexcept {
        if (_impl) {
            _impl->deallocate(block);
        }
    }
    bool owns(Block block) const noexcept {
        return _impl ? _impl->owns(block) : false;
    }
    void recycle() noexcept {
        if (_impl) {
            _impl->recycle();
        }
    }
    void release() noexcept {
        if (_impl) {
            _impl->release();
        }
    }

    explicit operator bool() const noexcept {
        return _impl != nullptr;
    }
};

}  // namespace intel_npu::vm
