// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)

#    if defined _WIN32 && !defined NOMINMAX
#        define NOMINMAX
#    endif
#    include <xbyak/xbyak_util.h>

#    ifndef _WIN32
#        include <sys/mman.h>

#        include <mutex>
#        include <unordered_map>
#    endif

#    include "openvino/core/except.hpp"
#    include "openvino/core/type/bfloat16.hpp"
#    include "openvino/core/type/float16.hpp"
#    include "openvino/reference/utils/jit_generator.hpp"

namespace {
#    if (OV_ENABLE_EXTERNAL_SANITIZER == 1)
// Custom allocator that uses OS-level page-mapped memory (VirtualAlloc on Windows,
// mmap(MAP_ANONYMOUS) on Linux) instead of the malloc family.
//
// Root cause of the ASAN access violation
// =========================================
// Xbyak::AlignedMalloc calls _aligned_malloc (Windows) / posix_memalign (Linux).
// Both are intercepted by ASAN and fulfilled from ASAN's own Arena Regions -- large
// contiguous blocks that ASAN reserves with VirtualAlloc/mmap and then sub-divides
// internally.  Even when the requested size and the returned address are both
// page-aligned, the backing physical page still belongs to the Arena Region and is
// therefore subject to ASAN's internal page-lifecycle operations:
//
//   1. Arena decommit/recommit: when ASAN reclaims pages inside an Arena for memory
//      pressure it calls VirtualAlloc(..., MEM_COMMIT, PAGE_READWRITE), which drops
//      the PAGE_EXECUTE bit that Xbyak had previously set via VirtualProtect.  The
//      next call through the JIT function pointer then triggers an access violation.
//
//   2. Redzone enforcement: ASAN places redzone bytes immediately around every heap
//      allocation and marks those redzones PAGE_NOACCESS.  If the redzone boundary
//      falls on the same OS page as the JIT buffer (possible when the Arena layout
//      does not perfectly align to page boundaries at the Arena level), the
//      VirtualProtect call for the redzone strips the execute permission from the
//      shared page.
//
// Simply rounding the requested size up to a page boundary is not sufficient:
// it aligns the *allocation* but does not move the page out of ASAN's Arena, so
// ASAN's internal page management can still modify the page's protection attributes.
//
// Fix: use VirtualAlloc / mmap(MAP_ANONYMOUS) directly
// ======================================================
// These OS primitives create an independent Virtual Address Region that is entirely
// outside any ASAN-managed Arena.  ASAN uses these same primitives internally and
// therefore never instruments or tracks memory obtained through them.  No redzones
// are added, no decommit/recommit logic touches the region, and the page protection
// set by Xbyak (PAGE_EXECUTE_READWRITE) remains stable for the lifetime of the
// process.
struct PageAlignedAllocator : public Xbyak::Allocator {
    uint8_t* alloc(size_t size) override {
        const size_t page_size = Xbyak::inner::getPageSize();
        size = (size + page_size - 1) & ~(page_size - 1);
#        ifdef _WIN32
        return reinterpret_cast<uint8_t*>(VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
#        else
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED)
            return nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            alloc_sizes_[reinterpret_cast<uintptr_t>(p)] = size;
        }
        return reinterpret_cast<uint8_t*>(p);
#        endif
    }
    void free(uint8_t* p) override {
        if (!p)
            return;
#        ifdef _WIN32
        VirtualFree(p, 0, MEM_RELEASE);
#        else
        size_t sz = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = alloc_sizes_.find(reinterpret_cast<uintptr_t>(p));
            if (it != alloc_sizes_.end()) {
                sz = it->second;
                alloc_sizes_.erase(it);
            }
        }
        if (sz)
            munmap(p, sz);
#        endif
    }
#        ifndef _WIN32
    std::mutex mutex_;
    std::unordered_map<uintptr_t, size_t> alloc_sizes_;
#        endif
};

Xbyak::Allocator& getPageAlignedAllocator() {
    static PageAlignedAllocator instance;
    return instance;
}
#    endif
}  // anonymous namespace

namespace ov {
namespace reference {
namespace jit {
using namespace Xbyak;

bool Generator::mayiuse(const cpu_isa_t cpu_isa) {
    // note: MSVC 2022 (17.4) is not able to compile the next line for ARM and ARM64
    // so, we disable this code since for non-x86 platforms it returns 'false' anyway
    static Xbyak::util::Cpu cpu;

    using namespace Xbyak::util;

    switch (cpu_isa) {
    case sse42:
        return cpu.has(Cpu::tSSE42);
    case avx:
        return cpu.has(Cpu::tAVX);
    case avx2:
        return cpu.has(Cpu::tAVX2);
    case avx512_common:
        return cpu.has(Cpu::tAVX512F);
    case avx512_core:
        return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) && cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ);
    case avx512_core_vnni:
        return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) && cpu.has(Cpu::tAVX512VL) &&
               cpu.has(Cpu::tAVX512DQ) && cpu.has(Cpu::tAVX512_VNNI);
    case avx512_mic:
        return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512CD) && cpu.has(Cpu::tAVX512ER) && cpu.has(Cpu::tAVX512PF);
    case avx512_mic_4ops:
        return mayiuse(avx512_mic) && cpu.has(Cpu::tAVX512_4FMAPS) && cpu.has(Cpu::tAVX512_4VNNIW);
    case avx512_core_bf16:
        return mayiuse(avx512_core_vnni) && cpu.has(Cpu::tAVX512_BF16);
    case avx512_vpopcnt:
        return true && cpu.has(Cpu::tAVX512_VPOPCNTDQ);
    case fp16:
        return cpu.has(Cpu::tF16C);
    case cpu_isa_t::pclmulqdq:
        return cpu.has(Cpu::tPCLMULQDQ);
    case cpu_isa_t::vpclmulqdq:
        return cpu.has(Cpu::tVPCLMULQDQ);
    case isa_any:
        return true;
    }
    return false;
}

bool Generator::is_x64() {
    return sizeof(void*) == 8;
}

Generator::Generator(cpu_isa_t isa, void* code_ptr, size_t code_size)
// Note: the ASAN tools (e.g. clang's AddressSanitizer) may insert red zones around the allocated code buffer, and
// change the buffer's attribute to non-executable. To avoid this, we use a custom allocator that rounds up the
// buffer size to a full page boundary, so that each CodeArray occupies complete pages and setProtectModeRW() called
// during destruction cannot strip the execute permission from pages that are simultaneously used by another live
// CodeArray instance. This only for ASAN, the production still keeps using original logic with default allocator,
// so the performance is not affected.
#    if (OV_ENABLE_EXTERNAL_SANITIZER == 1)
    : Xbyak::CodeGenerator(code_size, code_ptr, &getPageAlignedAllocator()),
#    else
    : Xbyak::CodeGenerator(code_size, code_ptr),
#    endif
      size_of_abi_save_regs(num_abi_save_gpr_regs * rax.getBit() / 8 + xmm_to_preserve * xmm_len),
      reg_EVEX_max_8b_offt(rbp) {
    if (isa == avx512_core) {
        m_vlen = zmm_len;
    } else if (isa == avx2) {
        m_vlen = ymm_len;
    } else {
        OPENVINO_THROW("Unsupported isa: ", isa);
    }
}

void Generator::preamble() {
    if (xmm_to_preserve) {
        sub(rsp, xmm_to_preserve * xmm_len);
        for (size_t i = 0; i < xmm_to_preserve; ++i)
            movdqu(ptr[rsp + i * xmm_len], Xbyak::Xmm(static_cast<int>(xmm_to_preserve_start + i)));
    }
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
        push(Xbyak::Reg64(abi_save_gpr_regs[i]));
    if (mayiuse(avx512_common)) {
        mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
    }
}

void Generator::postamble() {
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
        pop(Xbyak::Reg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
    if (xmm_to_preserve) {
        for (size_t i = 0; i < xmm_to_preserve; ++i)
            movdqu(Xbyak::Xmm(static_cast<int>(xmm_to_preserve_start + i)), ptr[rsp + i * xmm_len]);
        add(rsp, xmm_to_preserve * xmm_len);
    }
    if (mayiuse(avx) && !mayiuse(avx512_mic))
        vzeroupper();
    ret();
}

void Generator::foreach (const Xbyak::Reg64& idx,
                         size_t step,
                         const Xbyak::Reg64& end,
                         std::function<void(const Xbyak::Reg64&)> && fn) {
    Label loop, exit;

    L(loop);
    cmp(idx, end);
    jge(exit);

    fn(idx);

    add(idx, static_cast<uint32_t>(step));
    jmp(loop);
    L(exit);
}

template <>
void Generator::copy<uint8_t>(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    push(rsi);
    push(r15);

    xor_(rsi, rsi);

    foreach (rsi, 1, size, [&, this](const Xbyak::Reg64& idx) {
        mov(r15b, byte[src + idx * sizeof(uint8_t)]);
        mov(byte[dst + idx * sizeof(uint8_t)], r15b);
    })
        ;

    pop(r15);
    pop(rsi);
}

template <>
void Generator::copy<int8_t>(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    push(rsi);
    push(r15);

    xor_(rsi, rsi);

    foreach (rsi, 1, size, [&, this](const Xbyak::Reg64& idx) {
        mov(r15b, byte[src + idx * sizeof(int8_t)]);
        mov(byte[dst + idx * sizeof(int8_t)], r15b);
    })
        ;

    pop(r15);
    pop(rsi);
}

template <>
void Generator::copy<uint16_t>(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    push(rsi);
    push(r15);

    xor_(rsi, rsi);

    foreach (rsi, 1, size, [&, this](const Xbyak::Reg64& idx) {
        mov(r15w, word[src + idx * sizeof(uint16_t)]);
        mov(word[dst + idx * sizeof(uint16_t)], r15w);
    })
        ;

    pop(r15);
    pop(rsi);
}

template <>
void Generator::copy<uint32_t>(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    push(rsi);
    push(r15);

    xor_(rsi, rsi);

    foreach (rsi, 1, size, [&, this](const Xbyak::Reg64& idx) {
        mov(r15d, dword[src + idx * sizeof(uint32_t)]);
        mov(dword[dst + idx * sizeof(uint32_t)], r15d);
    })
        ;

    pop(r15);
    pop(rsi);
}

template <>
void Generator::copy<float16>(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    copy<uint16_t>(dst, src, size);
}

template <>
void Generator::copy<float>(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    copy<uint32_t>(dst, src, size);
}

template <>
void Generator::copy<bfloat16>(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    copy<uint16_t>(dst, src, size);
}
}  // namespace jit
}  // namespace reference
}  // namespace ov

#endif  // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64
