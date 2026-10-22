// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vm_runtime/virtual_machine.h"
#include "npu_bytecode_utils/bytecode_reader.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "npu_bytecode_utils/managed_vector.hpp"
#include "npu_bytecode_utils/network_description.hpp"
#include "npu_bytecode_utils/print_utils.hpp"
#include "npu_bytecode_utils/section_header_table.hpp"
#include "npu_bytecode_utils/span.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "utils/allocator_core.hpp"
#include "utils/buffer.hpp"
#include "utils/buffer_metadata.hpp"
#include "utils/call_frame.hpp"
#include "vm_runtime/utils/function.hpp"
#include "utils/host_allocator.hpp"
#include "utils/kernel_execution.hpp"
#include "utils/level_zero_allocator.hpp"
#include "utils/math.hpp"
#include "utils/network_metadata.hpp"
#include "utils/parameters.hpp"
#include "vm_runtime/npu_vm_runtime.hpp"

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <algorithm>
#include <array>
#include <cfenv>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace intel_npu;
using namespace intel_npu::vm;

namespace {

// Represents the execution lifecycle of the VMEngine.
// Transitions: Initialized -> Running -> Finalized (normal) or Halted (error).
enum class ExecState : uint8_t { Initialized, Running, Halted, Finalized };

template <typename T>
std::optional<int64_t> evaluateCmp(T lhs, T rhs, uint8_t cmpType) {
    switch (static_cast<intel_npu::vm::CmpPredicate>(cmpType)) {
    case intel_npu::vm::CmpPredicate::EQ:
        return static_cast<int64_t>(lhs == rhs);
    case intel_npu::vm::CmpPredicate::NE:
        return static_cast<int64_t>(lhs != rhs);
    case intel_npu::vm::CmpPredicate::GT:
        return static_cast<int64_t>(lhs > rhs);
    case intel_npu::vm::CmpPredicate::GTE:
        return static_cast<int64_t>(lhs >= rhs);
    case intel_npu::vm::CmpPredicate::LT:
        return static_cast<int64_t>(lhs < rhs);
    case intel_npu::vm::CmpPredicate::LTE:
        return static_cast<int64_t>(lhs <= rhs);
    default:
        NPU_VM_LOG_ERROR("Unknown CMP predicate: {}", static_cast<unsigned>(cmpType));
        return std::nullopt;
    }
}

template <typename SignedT>
inline SignedT convertUnsignedToSignedPreserveBits(std::make_unsigned_t<SignedT> value) {
    using UnsignedT = std::make_unsigned_t<SignedT>;
    // For logical shifts:
    //   - We intentionally compute in unsigned type to get logical bit behavior
    //   - That leaves us with an unsigned result that may have the top bit set
    //   - Storing that directly to a 64-bit signed register would zero-extend
    //   - The helper converts to signed 32-bit first, so later widening to 64-bit sign-extends correctly
    constexpr auto signBit = UnsignedT{1} << ((sizeof(SignedT) * 8) - 1);
    if (value <= static_cast<UnsignedT>(std::numeric_limits<SignedT>::max())) {
        return static_cast<SignedT>(value);
    }
    return static_cast<SignedT>(value - signBit) + std::numeric_limits<SignedT>::min();
}

template <typename SignedT>
inline SignedT arithmeticShiftRightDefined(SignedT value, int64_t rawShiftAmount) {
    using UnsignedT = std::make_unsigned_t<SignedT>;
    constexpr auto bitWidth = sizeof(SignedT) * 8;
    constexpr auto shiftMask = bitWidth - 1;

    // Arithmetic right shift of a negative signed value is implementation-defined in C++.
    // The bytecode opcode contract is stricter: SRA must preserve the sign bit regardless of
    // compiler or target. Implement the operation in unsigned space, then explicitly fill the
    // high bits when the source is negative. Finally, convert the resulting bit pattern back to
    // the signed type through convertUnsignedToSignedPreserveBits().
    const auto shiftAmount = static_cast<uint64_t>(rawShiftAmount) & shiftMask;
    auto shifted = static_cast<UnsignedT>(static_cast<UnsignedT>(value) >> shiftAmount);
    if (value < 0 && shiftAmount != 0) {
        shifted |= static_cast<UnsignedT>(~UnsignedT{0} << (bitWidth - shiftAmount));
    }
    return convertUnsignedToSignedPreserveBits<SignedT>(shifted);
}

template <typename To, typename From>
inline To vmBitCast(From from) {
    static_assert(sizeof(To) == sizeof(From), "vmBitCast requires same-size types");
    To result;
    std::memcpy(&result, &from, sizeof(To));
    return result;
}

// Extract the low N bits of a register value and sign-extend to int64_t.
// This avoids relying on implementation-defined narrowing casts from wider integer types.
inline int64_t signExtend8(int64_t regValue) {
    constexpr uint64_t signBit = 0x80u;
    constexpr int64_t twosComplementOffset = 0x100;
    const auto raw = static_cast<uint64_t>(regValue) & 0xFFu;
    return raw >= signBit ? static_cast<int64_t>(raw) - twosComplementOffset : static_cast<int64_t>(raw);
}

inline int64_t signExtend16(int64_t regValue) {
    constexpr uint64_t signBit = 0x8000u;
    constexpr int64_t twosComplementOffset = 0x10000;
    const auto raw = static_cast<uint64_t>(regValue) & 0xFFFFu;
    return raw >= signBit ? static_cast<int64_t>(raw) - twosComplementOffset : static_cast<int64_t>(raw);
}

inline int64_t signExtend32(int64_t regValue) {
    constexpr uint64_t signBit = 0x80000000u;
    constexpr int64_t twosComplementOffset = 0x100000000LL;
    const auto raw = static_cast<uint64_t>(regValue) & 0xFFFFFFFFu;
    return raw >= signBit ? static_cast<int64_t>(raw) - twosComplementOffset : static_cast<int64_t>(raw);
}

// Converts a floating-point value to a signed integer type with saturating semantics:
//   - NaN        → 0
//   - +infinity or src > INT_MAX → INT_MAX
//   - -infinity  or src < INT_MIN → INT_MIN
//   - finite, in-range → truncate toward zero (standard C++ cast)
// This avoids the C++ undefined behaviour that occurs when a finite float is outside
// the representable range of the destination integer type.
template <typename IntT, typename FloatT>
inline int64_t saturatingFloatToInt(FloatT src) noexcept {
    static_assert(std::is_floating_point_v<FloatT>);
    static_assert(std::is_signed_v<IntT> && std::is_integral_v<IntT>);
    if (std::isnan(src)) {
        return 0;
    }
    // The float representation of the integer limits may not be exact (e.g. INT64_MAX rounds
    // up to 2^63 in float/double). Using the float-converted limit for comparison is still
    // correct: any src that survives the lower-bound check and is < maxF is a finite value
    // whose truncation is representable in IntT.
    constexpr auto minF = static_cast<FloatT>(std::numeric_limits<IntT>::min());
    constexpr auto maxF = static_cast<FloatT>(std::numeric_limits<IntT>::max());
    if (src <= minF) {
        return static_cast<int64_t>(std::numeric_limits<IntT>::min());
    }
    if (src >= maxF) {
        return static_cast<int64_t>(std::numeric_limits<IntT>::max());
    }
    return static_cast<int64_t>(static_cast<IntT>(src));
}

bool decodeInstruction(const uint8_t* pc, size_t bytesAvailable, intel_npu::vm::OpCode& opcode,
                       size_t& instructionSize) {
    if (bytesAvailable < intel_npu::vm::OPCODE_SIZE) {
        NPU_VM_LOG_ERROR("Reached end of function body while decoding opcode. This likely means the function body "
                         "is malformed.");
        return false;
    }
    opcode = intel_npu::vm::getOpcode(pc);
    const auto sizeOpt = intel_npu::vm::getInstructionSize(opcode, pc, bytesAvailable);
    if (!sizeOpt.has_value()) {
        NPU_VM_LOG_ERROR("Failed to decode instruction size for opcode {}: unknown opcode or malformed instruction.",
                         static_cast<uint16_t>(opcode));
        return false;
    }
    instructionSize = sizeOpt.value();
    if (bytesAvailable < instructionSize) {
        NPU_VM_LOG_ERROR("Reached end of function body while decoding an instruction. "
                         "This likely means the function body is malformed.");
        return false;
    }
    return true;
}

class BytecodeModule {
    std::vector<Function> _functions;                     // All functions from the bytecode function section(s)
    std::vector<ManagedVector<uint8_t>> _constants;       // All constant entries from the bytecode constant section(s)
    std::vector<ManagedVector<uint8_t>> _kernelBinaries;  // All kernel binaries from the bytecode kernel section(s)
    std::vector<ManagedVector<uint8_t>> _rawStrings;      // All raw strings from the bytecode string section(s)
    std::vector<uint16_t> _typeByteSizes;                 // Element byte size for each type entry
    std::optional<NetworkMetadata> _networkMetadata;      // Metadata about the network, if present in the bytecode

    /// Parses the bytecode and populates the module's internal data structures.
    /// @param bytecode The serialized bytecode to parse.
    /// @param copyBytecode If true, the module will make a copy of the bytecode and own the memory. If false, the
    /// module will reference the provided bytecode directly, meaning the caller must ensure the bytecode remains valid
    /// for the lifetime of the module
    /// @return true on success, false on failure
    bool parse(const intel_npu::vm::Span<uint8_t>& bytecode, bool copyBytecode = false) {
        intel_npu::vm::BytecodeReader reader(bytecode);
        if (!reader.parseFile()) {
            NPU_VM_LOG_ERROR("Failed to parse bytecode file.");
            return false;
        }

        auto& sectionHeaderTable = reader.getSectionHeaderTable();
        auto& sectionHeaders = sectionHeaderTable.getSectionHeaders();
        auto& sections = reader.getSections();

        const auto typeByteSizes = intel_npu::vm::extractTypeByteSizes(sectionHeaderTable, sections);
        if (!typeByteSizes.has_value()) {
            return false;
        }
        _typeByteSizes = typeByteSizes.value();

        // Iterate over section headers to find function sections and extract individual function bodies
        for (size_t headerIdx = 0; headerIdx < sectionHeaders.size(); ++headerIdx) {
            const auto& header = sectionHeaders.at(headerIdx);
            if (header.type != intel_npu::vm::SectionType::ConstantSection) {
                continue;
            }

            auto constantSectionInfo = dynamic_cast<intel_npu::vm::details::DataSectionInfo*>(header.info.get());
            if (constantSectionInfo == nullptr || headerIdx >= sections.size()) {
                NPU_VM_LOG_ERROR("Constant section header is malformed");
                return false;
            }
            const auto& section = sections.at(headerIdx);
            for (const auto& dataInfo : constantSectionInfo->dataInfos) {
                if (dataInfo.offset > section.size() || dataInfo.size > section.size() - dataInfo.offset) {
                    NPU_VM_LOG_ERROR("Constant data exceeds section bounds: offset={}, size={}, sectionSize={}",
                                     dataInfo.offset, dataInfo.size, section.size());
                    return false;
                }
                _constants.emplace_back(section.subspan(dataInfo.offset, dataInfo.size), /*copyData=*/copyBytecode);
            }
        }

        for (size_t headerIdx = 0; headerIdx < sectionHeaders.size(); ++headerIdx) {
            const auto& header = sectionHeaders.at(headerIdx);
            if (header.type != intel_npu::vm::SectionType::FuncSection) {
                continue;
            }
            auto funcSectionInfo = dynamic_cast<intel_npu::vm::details::FunctionSectionInfo*>(header.info.get());
            if (funcSectionInfo == nullptr) {
                NPU_VM_LOG_ERROR("Function section header does not contain function section info");
                return false;
            }
            for (size_t i = 0; i < funcSectionInfo->functionInfos.size(); ++i) {
                if (headerIdx >= sections.size()) {
                    NPU_VM_LOG_ERROR("Could not find associated section with section header {}", headerIdx);
                    return false;
                }

                const auto& funcInfo = funcSectionInfo->functionInfos.at(i);
                const auto functionNameOpt = reader.getString(funcInfo.nameIndex);
                if (!functionNameOpt.has_value()) {
                    NPU_VM_LOG_ERROR(
                            "Failed to retrieve function name string for function at index {} in function section "
                            "header {}",
                            i, headerIdx);
                    return false;
                }
                auto functionName = functionNameOpt.value();
                // Remove null-terminator from string
                functionName.erase(std::find(functionName.begin(), functionName.end(), '\0'), functionName.end());

                const auto functionTypeOpt = reader.getFunctionType(funcInfo.functionTypeIndex);
                if (!functionTypeOpt.has_value()) {
                    NPU_VM_LOG_ERROR(
                            "Failed to retrieve function type for function at index {} in function section header {}",
                            i, headerIdx);
                    return false;
                }
                const auto& functionType = functionTypeOpt.value();
                std::vector<intel_npu::vm::FuncParamResType> paramTypes;
                for (auto paramTypeIndex : functionType.paramTypeIndices) {
                    auto paramType = intel_npu::vm::extractFuncParamResType(reader, paramTypeIndex);
                    if (!paramType.has_value()) {
                        NPU_VM_LOG_ERROR(
                                "Failed to extract parameter type for param type index {} in function at index {} "
                                "in function section header {}",
                                paramTypeIndex, i, headerIdx);
                        return false;
                    }
                    paramTypes.push_back(std::move(paramType.value()));
                }
                std::vector<intel_npu::vm::FuncParamResType> resultTypes;
                for (auto resultTypeIndex : functionType.resultTypeIndices) {
                    auto resultType = intel_npu::vm::extractFuncParamResType(reader, resultTypeIndex);
                    if (!resultType.has_value()) {
                        NPU_VM_LOG_ERROR(
                                "Failed to extract result type for result type index {} in function at index {} "
                                "in function section header {}",
                                resultTypeIndex, i, headerIdx);
                        return false;
                    }
                    resultTypes.push_back(std::move(resultType.value()));
                }

                const auto numGeneralRegisters = funcInfo.numGeneralRegisters;
                if (numGeneralRegisters < paramTypes.size()) {
                    NPU_VM_LOG_ERROR("Function '{}' does not have enough general registers to hold all parameters. "
                                     "Required: {}, "
                                     "available: {}.",
                                     functionName, paramTypes.size(), numGeneralRegisters);
                    return false;
                }
                if (numGeneralRegisters > std::numeric_limits<int16_t>::max()) {
                    NPU_VM_LOG_ERROR("Function '{}' has too many general registers: {}. Maximum supported is {}.",
                                     functionName, numGeneralRegisters, std::numeric_limits<int16_t>::max());
                    return false;
                }

                const auto isEntrypoint = funcSectionInfo->entrypointFunctionIndex == i;
                const auto& section = sections.at(headerIdx);
                if (funcInfo.bodyOffset > section.size() || funcInfo.bodySize > section.size() - funcInfo.bodyOffset) {
                    NPU_VM_LOG_ERROR("Function body exceeds section bounds for function {}", functionName);
                    return false;
                }
                const auto functionBody = section.subspan(funcInfo.bodyOffset, funcInfo.bodySize);
                _functions.emplace_back(functionName, numGeneralRegisters, isEntrypoint, std::move(paramTypes),
                                        std::move(resultTypes), functionBody, /*copyBody=*/copyBytecode);

                if (!_functions.back().parseInstructionOffsets()) {
                    NPU_VM_LOG_ERROR("Failed to pre-decode instruction offsets for function '{}'",
                                     _functions.back().getName());
                    return false;
                }
            }
        }

        for (size_t headerIdx = 0; headerIdx < sectionHeaders.size(); ++headerIdx) {
            const auto& header = sectionHeaders.at(headerIdx);
            if (header.type != intel_npu::vm::SectionType::KernelSection) {
                continue;
            }

            auto kernelSectionInfo = dynamic_cast<intel_npu::vm::details::DataSectionInfo*>(header.info.get());
            if (kernelSectionInfo == nullptr) {
                NPU_VM_LOG_ERROR("Kernel section header does not contain data section info");
                return false;
            }
            if (headerIdx >= sections.size()) {
                NPU_VM_LOG_ERROR("Could not find associated section with section header {}", headerIdx);
                return false;
            }
            const auto& section = sections.at(headerIdx);

            for (auto& dataInfo : kernelSectionInfo->dataInfos) {
                if (dataInfo.offset > section.size() || dataInfo.size > section.size() - dataInfo.offset) {
                    NPU_VM_LOG_ERROR("Kernel binary exceeds section bounds: offset={}, size={}, sectionSize={}",
                                     dataInfo.offset, dataInfo.size, section.size());
                    return false;
                }
                const auto kernelBinary = section.subspan(dataInfo.offset, dataInfo.size);
                _kernelBinaries.emplace_back(kernelBinary, /*copyData=*/copyBytecode);
            }
        }

        for (size_t headerIdx = 0; headerIdx < sectionHeaders.size(); ++headerIdx) {
            const auto& header = sectionHeaders.at(headerIdx);
            if (header.type != intel_npu::vm::SectionType::StringSection) {
                continue;
            }

            auto stringSectionInfo = dynamic_cast<intel_npu::vm::details::DataSectionInfo*>(header.info.get());
            if (stringSectionInfo == nullptr) {
                NPU_VM_LOG_ERROR("String section header does not contain data section info");
                return false;
            }
            if (headerIdx >= sections.size()) {
                NPU_VM_LOG_ERROR("Could not find associated section with section header {}", headerIdx);
                return false;
            }
            const auto& section = sections.at(headerIdx);

            for (auto& dataInfo : stringSectionInfo->dataInfos) {
                if (dataInfo.offset > section.size() || dataInfo.size > section.size() - dataInfo.offset) {
                    NPU_VM_LOG_ERROR("String data exceeds section bounds: offset={}, size={}, sectionSize={}",
                                     dataInfo.offset, dataInfo.size, section.size());
                    return false;
                }
                const auto rawString = section.subspan(dataInfo.offset, dataInfo.size);
                _rawStrings.emplace_back(rawString, /*copyData=*/copyBytecode);
            }
        }

        _networkMetadata = parseNetworkMetadata(reader);

        return true;
    }

public:
    /// Creates a BytecodeModule from the given bytecode buffer
    /// @param bytecode The buffer containing the bytecode to parse and create the module from
    /// @param copyBytecode If true, the module will make a copy of the bytecode and own the memory. If false, the
    /// module will reference the provided bytecode directly, meaning the caller must ensure the bytecode remains valid
    /// for the lifetime of the module
    /// @return A unique pointer to the created BytecodeModule if parsing succeeded, or nullptr if parsing failed
    static std::unique_ptr<BytecodeModule> createFrom(const intel_npu::vm::Span<uint8_t>& bytecode,
                                                      bool copyBytecode = false) {
        auto bytecodeModule = std::make_unique<BytecodeModule>();
        if (!bytecodeModule->parse(bytecode, copyBytecode)) {
            return nullptr;
        }
        return bytecodeModule;
    }

    /// Returns the list of all functions loaded from the bytecode
    const std::vector<Function>& getFunctions() const {
        return _functions;
    }

    const std::vector<ManagedVector<uint8_t>>& getConstants() const {
        return _constants;
    }

    /// Returns a function by name among the loaded functions
    /// @param name The name of the function to find
    /// @return A pointer to the Function if found, or nullptr
    Function* getFunction(std::string_view name) {
        for (auto& function : _functions) {
            if (function.getName() == name) {
                return &function;
            }
        }
        return nullptr;
    }

    /// Returns the list of all kernel binaries loaded from the bytecode
    std::vector<ManagedVector<uint8_t>>& getKernelBinaries() {
        return _kernelBinaries;
    }

    const std::vector<ManagedVector<uint8_t>>& getRawStrings() const {
        return _rawStrings;
    }

    /// Returns the vector containing the byte size of each type entry, indexed by the type's index in the bytecode type
    /// section
    std::vector<uint16_t>& getTypeByteSizes() {
        return _typeByteSizes;
    }

    void setNetworkMetadata(const NetworkMetadata& metadata) {
        _networkMetadata = metadata;
    }

    const std::optional<NetworkMetadata>& getNetworkMetadata() const {
        return _networkMetadata;
    }
};

// Groups the buffer state a single inference needs: the active BufferManager, its bytecode-level metadata, and
// the Level Zero context plus bookkeeping that decide when the manager must be (re)installed. Kept together so
// the install/recycle policy lives in one place rather than scattered across EngineState members (see ensure()).
class ContextBuffers {
    // Host-arena-backed until ensure() installs a context-appropriate manager before inference.
    BufferManager _manager{DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, makeHostAllocator()};
    std::unordered_map<BufferHandle, BufferMetadata> _metadata;  // Bytecode-level metadata per handle
    ze_context_handle_t _allocatorContext = nullptr;             // nullptr for a host-backed manager
    bool _managerInstalled = false;

public:
    BufferManager& manager() {
        return _manager;
    }

    std::unordered_map<BufferHandle, BufferMetadata>& metadata() {
        return _metadata;
    }

    // Install a context-appropriate BufferManager before an inference. With a Level Zero context,
    // (re)install a device-backed manager whenever the context changes so owned buffers are NPU-visible.
    // Without a context, install a host-backed manager only if none exists yet, so an intervening predict
    // call does not clobber a device manager from a prior execute.
    void ensure(ze_context_handle_t ctx) {
        const bool contextChanged = (ctx != nullptr && ctx != _allocatorContext);
        if (_managerInstalled && !contextChanged) {
            // Same context: keep the grown arena and recycle its memory so this inference reuses the
            // previous one's. No external reset required.
            _manager.recycle();
            _metadata.clear();
            return;
        }
        // Reached only when no manager is installed yet, or the device context changed. Build one to match
        // the incoming context: a GrowingArena over Level Zero (NPU-visible) for a device context, or a
        // host-memory arena when there is none (first-time predict / offline execute). A null context with a
        // manager already installed returned above, so it never clobbers an existing (possibly device) manager.
        AnyAllocator allocator = ctx != nullptr ? makeLevelZeroAllocator(ctx) : makeHostAllocator();
        _manager = BufferManager(DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, std::move(allocator));
        _allocatorContext = ctx;
        _managerInstalled = true;
        _metadata.clear();
    }

    // Per-inference cleanup invoked from EngineState::reset(): drop the previous inference's buffers and
    // metadata but keep the grown arena, so the next inference on the same context reuses its memory instead
    // of re-growing it. Genuine release of the underlying memory happens on a context change (the rebuild in
    // ensure()) and on destruction; see recycle() vs reset() in buffer.hpp.
    void recycle() noexcept {
        _manager.recycle();
        _metadata.clear();
    }
};

class ConstantPool {
    // Bytecode parsing only gives us constant payloads as views into the parsed module. That storage is not the right
    // runtime owner for zero-copy buffers because BUFFER_FROM_CONST needs the bytes to stay alive after the VM turns
    // them into buffer handles, and createFromMemory() only borrows memory instead of extending its lifetime.
    //
    // ConstantPool is the missing owner: it copies each constant once into allocator-backed shared memory, keeps that
    // storage alive for the whole module lifetime, and stores the resulting memory as spans. BUFFER_FROM_CONST can
    // then create read-only handles directly on top of those spans without a second copy, while the pool still
    // controls when the backing memory is released (module reload, context change, or engine destruction)
    AnyAllocator _allocator{makeHostAllocator()};
    std::vector<Block> _ownedBlocks;
    std::vector<Span<const uint8_t>> _constants;
    ze_context_handle_t _allocatorContext = nullptr;
    std::shared_ptr<const BytecodeModule> _materializedModule;
    bool _materialized = false;

public:
    ConstantPool() = default;
    ConstantPool(const ConstantPool&) = delete;
    ConstantPool& operator=(const ConstantPool&) = delete;
    ConstantPool(ConstantPool&&) = default;
    ConstantPool& operator=(ConstantPool&&) = default;
    ~ConstantPool() {
        deallocateAll();
    }

    void ensure(std::shared_ptr<const BytecodeModule> module, ze_context_handle_t ctx) {
        // If we already materialized constants for this context, keep the existing
        // storage and do nothing. Rebuilding is only needed when the Level Zero
        // context changes or the module changes
        const bool contextChanged = (ctx != _allocatorContext);
        if (_materialized && !contextChanged && _materializedModule == module) {
            return;
        }

        // Build a fresh pool for the new context so all constant bytes live in a
        // stable allocator-owned block that survives VM buffer creation
        ConstantPool nextPool;
        nextPool._allocator = ctx != nullptr ? makeLevelZeroAllocator(ctx) : makeHostAllocator();
        nextPool._allocatorContext = ctx;

        // Copy every extracted constant once into allocator-backed storage.
        // The VM will later create read-only buffer handles directly from these
        // spans, so the pool owns the lifetime
        const auto& sourceConstants = module->getConstants();
        nextPool._ownedBlocks.reserve(sourceConstants.size());
        nextPool._constants.reserve(sourceConstants.size());

        for (const auto& constant : sourceConstants) {
            const auto constantData = constant.get();
            if (constantData.empty()) {
                nextPool._constants.emplace_back();
                continue;
            }

            auto block = nextPool._allocator.allocate(constantData.size());
            if (!block) {
                throw std::bad_alloc();
            }

            std::memcpy(block.ptr, constantData.data(), constantData.size());
            nextPool._ownedBlocks.push_back(block);
            nextPool._constants.emplace_back(Span<const uint8_t>(block.ptr, block.size));
        }

        deallocateAll();
        *this = std::move(nextPool);
        _materializedModule = std::move(module);
        _materialized = true;
    }

    const std::vector<Span<const uint8_t>>& getConstants() const {
        return _constants;
    }

    void deallocateAll() noexcept {
        // Release the backing blocks first, then clear the span views and reset
        // the allocator state so the next ensure() starts from a clean pool
        for (auto& block : _ownedBlocks) {
            _allocator.deallocate(block);
        }
        _ownedBlocks.clear();
        _constants.clear();
        _allocator.release();
        _allocatorContext = nullptr;
        _materializedModule.reset();
        _materialized = false;
    }
};

class EngineState {
    ExecState _execState{ExecState::Initialized};  // Current execution state of the VM
    const uint8_t* _pc{nullptr};                   // Program counter

    ContextBuffers _buffers;  // BufferManager + metadata + the context bookkeeping that drives ensure()

    using kernel_binary_handle_t = void*;
    std::unordered_map<kernel_binary_handle_t, KernelInfo>
            _kernelInfos;  // Metadata for created kernels, keyed by their handles

    ExecutionContext _executionContext;
    ze_graph_dditable_ext_t* _graphDdiTableHandle =
            nullptr;  // Ddi table handle provided at execution time, used for kernel execution

public:
    EngineState() = default;
    EngineState(const EngineState&) = delete;
    EngineState& operator=(const EngineState&) = delete;
    EngineState(EngineState&&) = delete;
    EngineState& operator=(EngineState&&) = delete;
    ~EngineState() {
        destroyGraphHandles();
    }

    /// Resets the engine state to prepare for a new inference execution
    /// Note: The _kernelInfos map is not cleared here, as the graph handles can be reused across inferences, to avoid
    /// recreating them every time
    bool reset(bool resetExecutionContext) {
        _execState = ExecState::Initialized;
        _pc = nullptr;
        _buffers.recycle();
        if (resetExecutionContext) {
            if (_executionContext.reset()) {
                NPU_VM_LOG_ERROR("Failed to reset execution context");
                return false;
            }
        }

        return true;
    }

    /// Destroys all graph handles associated with the kernels in the current engine state
    void destroyGraphHandles() {
        if (_graphDdiTableHandle) {
            for (auto& [_, kernelInfo] : _kernelInfos) {
                auto graphHandle = kernelInfo.getGraphHandle();
                if (graphHandle) {
                    _graphDdiTableHandle->pfnDestroy(graphHandle);
                }
            }
        }
        _kernelInfos.clear();
    }

    ExecState getExecState() const {
        return _execState;
    }

    void setExecState(ExecState newState) {
        _execState = newState;
    }

    const uint8_t* getPC() const {
        return _pc;
    }

    void setPC(const uint8_t* newPC) {
        _pc = newPC;
    }

    BufferManager& getBufferManager() {
        return _buffers.manager();
    }

    std::unordered_map<BufferHandle, BufferMetadata>& getBufferMetadata() {
        return _buffers.metadata();
    }

    // (Re)installs a context-appropriate BufferManager before an inference; see ContextBuffers::ensure.
    void ensureBufferManager(ze_context_handle_t ctx) {
        _buffers.ensure(ctx);
    }

    std::unordered_map<kernel_binary_handle_t, KernelInfo>& getKernelInfos() {
        return _kernelInfos;
    }

    ExecutionContext& getExecutionContext() {
        return _executionContext;
    }

    ze_graph_dditable_ext_t* getDDITableHandle() {
        return _graphDdiTableHandle;
    }

    void setDDITableHandle(ze_graph_dditable_ext_t* ddiTableHandle) {
        _graphDdiTableHandle = ddiTableHandle;
    }
};

bool applyJump(EngineState& engineState, int64_t jumpOffset, const uint8_t* functionBodyStart,
               const uint8_t* functionBodyEnd, const Function& function) {
    if (jumpOffset == 0) {
        return false;
    }
    const auto pc = engineState.getPC();
    const auto bodySize = static_cast<int64_t>(functionBodyEnd - functionBodyStart);
    const auto currentIndex = static_cast<int64_t>(pc - functionBodyStart);
    if (jumpOffset > 0) {
        if (jumpOffset >= bodySize - currentIndex) {
            return false;
        }
    } else {
        if (jumpOffset < -currentIndex) {
            return false;
        }
    }
    const auto targetIndex = currentIndex + jumpOffset;
    // Reject jumps that would land in the middle of an instruction. Only body-relative offsets that
    // correspond to a previously decoded instruction start are permitted.
    if (!function.isValidInstructionOffset(static_cast<size_t>(targetIndex))) {
        NPU_VM_LOG_ERROR("Jump target offset {} in function '{}' does not land on an instruction boundary "
                         "(jumpOffset={}, currentIndex={})",
                         targetIndex, function.getName(), jumpOffset, currentIndex);
        return false;
    }
    engineState.setPC(std::next(pc, static_cast<std::ptrdiff_t>(jumpOffset)));
    return true;
}

class VMEngine {
    std::shared_ptr<BytecodeModule> _bytecodeModule;
    ConstantPool _constantPool;
    EngineState _state{};

    // Advances the program counter by the given bytes
    // No-op if the VM is not in the Running state
    void incrementPC(size_t bytes);

    // Interprets the instruction stream of the given function until a RET instruction is encountered or an unknown
    // opcode halts execution
    bool inferImpl(std::string_view functionName, Span<npu_vm_runtime_mem_ref_handle_t> inputMemRefs,
                   Span<npu_vm_runtime_mem_ref_handle_t> outputMemRefs);

    // Interprets the instruction stream of the given function until a RET instruction is encountered or an unknown
    // opcode halts execution. Results are written as raw register values to the provided sinks
    void execute(intel_npu::vm::CallFrame entryFrame);

public:
    /// Loads a bytecode module into the engine, replacing any previously loaded module. The loaded module can then be
    /// executed by calling its functions through the engine. The state of the engine is reset when a new module is
    /// loaded
    /// @param bytecodeModule The module to load into the engine
    /// @param loadParams parameters to load a module
    /// @return true if the module was successfully loaded, or false otherwise
    bool loadModule(std::shared_ptr<BytecodeModule> bytecodeModule, npu_vm_module_load_params* loadParams);

    /// Returns the currently loaded bytecode module, or nullptr if no module is loaded
    const std::shared_ptr<BytecodeModule>& getModule() const;

    /// Resets the internal state of the engine, to prepare for a new inference execution. This includes clearing any
    /// cached data or resources from previous executions, but does not unload the currently loaded module
    /// @param resetExecutionContext Whether to reset the execution context
    /// @return true if the state was successfully reset, or false otherwise
    bool resetState(bool resetExecutionContext);

    /// Calls the given function with the specified arguments and storage for return values. The number and types of
    /// the arguments / result values must match the function's parameter & result types.
    /// @param name The name of the function to call
    /// @param arguments The argument values to pass to the function
    /// @param results The result values to store the function's output
    /// @return true if the function executed and finalized successfully, or false otherwise
    bool call(std::string_view name, Span<const npu_vm_value> arguments, Span<npu_vm_value> results);

    /// Calls the main inference function with the specified execution parameters
    /// @param params The execution parameters from the VM Runtime API, including input and output buffer information
    /// @return true if the function executed and finalized successfully, or false otherwise
    /// @details The main inference function is expected to have the name 'main' and to have the same number and types
    /// of parameters as specified in the execution parameters (i.e. buffers for all inputs and outputs)
    bool infer(npu_vm_runtime_execute_params_t* params);

    /// Calls the main inference function with the specified execution parameters
    /// @param params The execution parameters from the VM Runtime API, including input and output buffer information
    /// @return true if the function executed and finalized successfully, or false otherwise
    /// @details The main inference function is expected to have the name 'main' and to have the same number and types
    /// of parameters as specified in the execution parameters (i.e. buffers for all inputs and outputs)
    bool infer(npu_vm_runtime_execute_params2_t* params);

    /// Calls the output shape prediction function with the specified execution parameters
    /// @param params The execution parameters from the VM Runtime API, specifying the model's buffer information needed
    /// for output shape prediction, and the result buffers where to store the predicted output shapes
    /// @return true if the function executed and finalized successfully, or false otherwise
    /// @details The output shape prediction function is expected to have the name 'output_shape' and to have
    /// the same number and types of parameters as specified in the output shape prediction execution parameters (i.e.
    /// buffers)
    bool predictOutputShape(npu_vm_runtime_predict_output_shape_params_t2* params);

    /// Host synchronizes the engine, blocking until all previously submitted commands have completed execution.
    bool hostSync();

    /// Submit recorded command lists
    bool submitRecordedCmdLists(npu_vm_runtime_execute_params2_t* params);

    /// Returns true if host code execution is required for the given execution parameters
    /// @param params The execution parameters from the VM Runtime API, including input and output buffer information
    bool hostCodeExecutionRequired(npu_vm_runtime_execute_params2_t* params);

    /// Sets Level Zero objects for the engine, including the context and the DDI table
    void setLevelZeroObjects(ze_context_handle_t ctx, ze_graph_dditable_ext_t* graphDdiTableExt);

    void ensureConstantPool(ze_context_handle_t ctx) {
        if (_bytecodeModule != nullptr) {
            _constantPool.ensure(_bytecodeModule, ctx);
        }
    }
};

bool VMEngine::loadModule(std::shared_ptr<BytecodeModule> bytecodeModule, npu_vm_module_load_params* loadParams) {
    if (bytecodeModule == nullptr) {
        NPU_VM_LOG_ERROR("Cannot load a null bytecode module");
        return false;
    }
    _bytecodeModule = std::move(bytecodeModule);
    _state.reset(/*resetExecutionContext=*/true);
    _state.destroyGraphHandles();  // Ensure any graph handles from a previous module are released
    auto metadata = _bytecodeModule->getNetworkMetadata();

    uint32_t concurrentJobCount = 1;
    npu_vm_runtime_create_execution_context_params_t* params = nullptr;
    if (loadParams != nullptr) {
        if (loadParams->create_execution_context_params) {
            params = loadParams->create_execution_context_params;
            setLevelZeroObjects(params->ctx, params->graphDdiTableExt);
        }
        concurrentJobCount = loadParams->concurrentJobCount;
    }
    ensureConstantPool(params ? params->ctx : nullptr);

    auto contextType = params ? ExecutionContext::ExecutionContextType::V2 : ExecutionContext::ExecutionContextType::V1;
    if (metadata.has_value()) {
        // V2 execution context is used when create_execution_context_params is provided,
        // otherwise V1 execution context is used
        auto& executionContext = _state.getExecutionContext();

        // initialize the execution context with the number of command lists,
        // number of network arguments, and concurrent job count
        bool counterBasedEventSupported = (loadParams && loadParams->counterBasedEventSupported) ? true : false;
        bool result =
                executionContext.initialize(contextType, static_cast<uint32_t>(metadata->numCmdLists),
                                            static_cast<uint32_t>(metadata->inputs.size() + metadata->outputs.size()),
                                            concurrentJobCount, counterBasedEventSupported);
        if (result == false) {
            NPU_VM_LOG_ERROR("Failed to initialize execution context");
            return false;
        }

        if (params) {
            auto initResult = executionContext.initializeExecutionObjects(
                    params->ctx, params->device, params->commandQueue, params->graphDdiTableExt,
                    params->commandQueueDdiTableExt, loadParams->internalCommandQueue);
            if (initResult != ZE_RESULT_SUCCESS) {
                NPU_VM_LOG_ERROR("Failed to initialize execution context");
                return false;
            }
        }
    }

    return true;
}

const std::shared_ptr<BytecodeModule>& VMEngine::getModule() const {
    return _bytecodeModule;
}

bool VMEngine::resetState(bool resetExecutionContext) {
    return _state.reset(resetExecutionContext);
}

bool VMEngine::hostSync() {
    auto& executionContext = _state.getExecutionContext();
    if (executionContext.hostSync() != ZE_RESULT_SUCCESS) {
        NPU_VM_LOG_ERROR("Failed to host sync execution context");
        return false;
    }
    return true;
}

bool VMEngine::submitRecordedCmdLists(npu_vm_runtime_execute_params2_t* params) {
    auto& executionContext = _state.getExecutionContext();
    executionContext.setParams(params);
    return executionContext.submitRecordedCmdLists() == ZE_RESULT_SUCCESS;
}

bool VMEngine::hostCodeExecutionRequired(npu_vm_runtime_execute_params2_t* params) {
    auto& executionContext = _state.getExecutionContext();

    if (!executionContext.isRecordedCmdListReusable()) {
        return true;
    }

    constexpr uint32_t dirtyFlagNone = static_cast<uint32_t>(MemRef::DirtyFlag::NONE);
    uint32_t tensorDirtyBits = dirtyFlagNone;
    if (params->pInputs) {
        for (uint32_t i = 0; i < params->numOfInputs; ++i) {
            auto inputMemRefHandle = params->pInputs[i];
            auto inputMemRef = reinterpret_cast<npu_vm_runtime_mem_ref*>(inputMemRefHandle);
            if (inputMemRef != nullptr && inputMemRef->dirtyFlags != dirtyFlagNone) {
                tensorDirtyBits |= inputMemRef->dirtyFlags;
            }
        }
    }

    if (params->pOutputs) {
        for (uint32_t i = 0; i < params->numOfOutputs; ++i) {
            auto outputMemRefHandle = params->pOutputs[i];
            auto outputMemRef = reinterpret_cast<npu_vm_runtime_mem_ref*>(outputMemRefHandle);
            if (outputMemRef != nullptr && outputMemRef->dirtyFlags != dirtyFlagNone) {
                tensorDirtyBits |= outputMemRef->dirtyFlags;
            }
        }
    }

    return tensorDirtyBits != dirtyFlagNone;
}

void VMEngine::setLevelZeroObjects(ze_context_handle_t ctx, ze_graph_dditable_ext_t* graphDdiTableExt) {
    // Install/refresh the BufferManager for this context before allocating buffers (see ensureBufferManager).
    _state.ensureBufferManager(ctx);
    // Materialize constants into the same memory space so BUFFER_FROM_CONST can
    // borrow them through createFromMemory without a second copy
    ensureConstantPool(ctx);
    // Keep the DDI table to release graph handle later, as graph handles are created with the DDI table
    _state.setDDITableHandle(graphDdiTableExt);
}

bool VMEngine::call(std::string_view name, Span<const npu_vm_value> arguments, Span<npu_vm_value> results) {
    if (_bytecodeModule == nullptr) {
        NPU_VM_LOG_ERROR("No bytecode module loaded in the engine");
        return false;
    }
    auto function = _bytecodeModule->getFunction(name);
    if (function == nullptr) {
        NPU_VM_LOG_ERROR("Function '{}' not found among loaded functions", name);
        return false;
    }
    const auto& paramTypes = function->getParamTypes();
    if (arguments.size() != paramTypes.size()) {
        NPU_VM_LOG_ERROR("Number of arguments passed ({}) does not match the function's parameter count ({})",
                         arguments.size(), paramTypes.size());
        return false;
    }

    const auto& resultTypes = function->getResultTypes();
    if (results.size() != resultTypes.size()) {
        NPU_VM_LOG_ERROR("Number of result slots provided ({}) does not match the function's result count ({}).",
                         results.size(), resultTypes.size());
        return false;
    }

    std::vector<int64_t> rawResults(resultTypes.size(), int64_t{0});
    std::vector<int64_t*> resultTargets;
    resultTargets.reserve(rawResults.size());
    for (auto& value : rawResults) {
        resultTargets.push_back(&value);
    }

    CallFrame frame(*function, EXIT_RETURN_ADDR, std::move(resultTargets));

    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access) - npu_vm_value is an union type from the C API

    // Set the arguments into the parameter registers from the call frame
    const auto firstParamRegIndex = function->getNumGeneralRegisters() - paramTypes.size();
    for (size_t i = 0; i < paramTypes.size(); ++i) {
        const auto paramType = paramTypes.at(i);
        const auto& argument = arguments.at(i);
        const auto typeCode = getTypeCode(paramType.type);
        switch (typeCode) {
        case TypeCode::INTEGER: {
            const auto isSigned = isTypeSigned(paramType.type);
            const auto bitWidth = getBitWidth(paramType.type);
            if (bitWidth == sizeof(int8_t) * CHAR_BIT) {
                auto value = isSigned ? static_cast<int64_t>(argument.i8) : static_cast<int64_t>(argument.u8);
                frame.setReg(static_cast<int16_t>(firstParamRegIndex + i), value);
                NPU_VM_LOG_DEBUG("  Setting parameter {} (type {}) to value: {}", i, isSigned ? "i8" : "u8", value);
            } else if (bitWidth == sizeof(int16_t) * CHAR_BIT) {
                auto value = isSigned ? static_cast<int64_t>(argument.i16) : static_cast<int64_t>(argument.u16);
                frame.setReg(static_cast<int16_t>(firstParamRegIndex + i), value);
                NPU_VM_LOG_DEBUG("  Setting parameter {} (type {}) to value: {}", i, isSigned ? "i16" : "u16", value);
            } else if (bitWidth == sizeof(int32_t) * CHAR_BIT) {
                auto value = isSigned ? static_cast<int64_t>(argument.i32) : static_cast<int64_t>(argument.u32);
                frame.setReg(static_cast<int16_t>(firstParamRegIndex + i), value);
                NPU_VM_LOG_DEBUG("  Setting parameter {} (type {}) to value: {}", i, isSigned ? "i32" : "u32", value);
            } else if (bitWidth == sizeof(int64_t) * CHAR_BIT) {
                auto value = isSigned ? argument.i64 : vmBitCast<int64_t>(argument.u64);
                frame.setReg(static_cast<int16_t>(firstParamRegIndex + i), value);
                NPU_VM_LOG_DEBUG("  Setting parameter {} (type {}) to value: {}", i, isSigned ? "i64" : "u64", value);
            } else {
                NPU_VM_LOG_ERROR("Unsupported integer bit width for argument {}: {} bits", i,
                                 static_cast<int>(bitWidth));
                return false;
            }
            break;
        }
        case TypeCode::FLOAT: {
            const auto bitWidth = getBitWidth(paramType.type);
            if (bitWidth == sizeof(float) * CHAR_BIT) {
                std::memcpy(&frame.getReg(static_cast<int16_t>(firstParamRegIndex + i)), &argument.f32, sizeof(float));
                NPU_VM_LOG_DEBUG("  Setting parameter {} (type f32) to value: {}", i, argument.f32);
            } else if (bitWidth == sizeof(double) * CHAR_BIT) {
                std::memcpy(&frame.getReg(static_cast<int16_t>(firstParamRegIndex + i)), &argument.f64, sizeof(double));
                NPU_VM_LOG_DEBUG("  Setting parameter {} (type f64) to value: {}", i, argument.f64);
            } else {
                NPU_VM_LOG_ERROR("Unsupported float bit width for argument {}: {} bits", i, static_cast<int>(bitWidth));
                return false;
            }
            break;
        }
        case TypeCode::BUFFER: {
            auto* data = static_cast<uint8_t*>(argument.buffer.data);
            const auto size = argument.buffer.size;
            if (data == nullptr) {
                NPU_VM_LOG_ERROR("Null buffer pointer for argument {}", i);
                return false;
            }
            if (size == 0) {
                NPU_VM_LOG_ERROR("Zero-sized buffer for argument {}", i);
                return false;
            }
            BufferHandle handle = 0;
            try {
                // npu_vm_call* arguments can alias host memory; create an unowned read-only VM handle over the buffer
                handle = _state.getBufferManager().createFromMemory(data, size, Permission::Read);
            } catch (const std::exception& e) {
                NPU_VM_LOG_ERROR("Failed to bind buffer argument {}: {}", i, e.what());
                return false;
            }

            frame.setReg(static_cast<int16_t>(firstParamRegIndex + i), static_cast<int64_t>(handle));
            NPU_VM_LOG_DEBUG("  Setting parameter {} (type buffer) to handle={} size={}", i, handle, size);
            break;
        }
        default: {
            NPU_VM_LOG_ERROR("Unsupported parameter type for argument {}", i);
            return false;
        }
        }
    }

    _state.setExecState(ExecState::Running);
    execute(std::move(frame));
    if (_state.getExecState() != ExecState::Finalized) {
        return false;
    }

    // Internal CALL/RETV propagation now moves raw register payloads through int64_t sinks.
    // This conversion keeps the public call() API contract by decoding those payloads into
    // the caller-provided ValueType variants at the API boundary
    for (size_t i = 0; i < resultTypes.size(); ++i) {
        const auto resultType = resultTypes.at(i);
        auto& result = results.at(i);
        const auto rawResult = rawResults.at(i);
        const auto typeCode = getTypeCode(resultType.type);
        if (typeCode == TypeCode::INTEGER) {
            const auto isSigned = isTypeSigned(resultType.type);
            const auto bitWidth = getBitWidth(resultType.type);
            if (bitWidth == sizeof(int8_t) * CHAR_BIT) {
                if (isSigned) {
                    result.i8 = static_cast<int8_t>(rawResult);
                } else {
                    result.u8 = static_cast<uint8_t>(rawResult);
                }
            } else if (bitWidth == sizeof(int16_t) * CHAR_BIT) {
                if (isSigned) {
                    result.i16 = static_cast<int16_t>(rawResult);
                } else {
                    result.u16 = static_cast<uint16_t>(rawResult);
                }
            } else if (bitWidth == sizeof(int32_t) * CHAR_BIT) {
                if (isSigned) {
                    result.i32 = static_cast<int32_t>(rawResult);
                } else {
                    result.u32 = static_cast<uint32_t>(rawResult);
                }
            } else if (bitWidth == sizeof(int64_t) * CHAR_BIT) {
                if (isSigned) {
                    result.i64 = rawResult;
                } else {
                    result.u64 = vmBitCast<uint64_t>(rawResult);
                }
            } else {
                NPU_VM_LOG_ERROR("Unsupported integer bit width for result {}: {} bits", i, static_cast<int>(bitWidth));
                return false;
            }
        } else if (typeCode == TypeCode::FLOAT) {
            const auto bitWidth = getBitWidth(resultType.type);
            if (bitWidth == sizeof(float) * CHAR_BIT) {
                std::memcpy(&result.f32, &rawResult, sizeof(float));
            } else if (bitWidth == sizeof(double) * CHAR_BIT) {
                std::memcpy(&result.f64, &rawResult, sizeof(double));
            } else {
                NPU_VM_LOG_ERROR("Unsupported float bit width for result {}: {} bits", i, static_cast<int>(bitWidth));
                return false;
            }
        } else if (typeCode == TypeCode::BUFFER) {
            const auto handle = static_cast<BufferHandle>(rawResult);

            if (!_state.getBufferManager().exists(handle)) {
                NPU_VM_LOG_ERROR("Unknown buffer handle in result {}: {}", i, handle);
                return false;
            }
            auto& buffer = _state.getBufferManager().getBuffer(handle);
            auto& resultBuffer = result.buffer;

            // The user provided an empty buffer slot for the result, so allocate memory for the result buffer and set
            // the data pointer and size in the result buffer
            if (resultBuffer.data == nullptr) {
                if (resultBuffer.size != 0) {
                    NPU_VM_LOG_ERROR("Result {} has null data pointer but non-zero size in the provided results array.",
                                     i);
                    return false;
                }
                // NOLINTNEXTLINE - malloc is used here to allow the caller to free the buffer memory with free()
                resultBuffer.data = reinterpret_cast<uint8_t*>(malloc(buffer.getSize()));
                if (resultBuffer.data == nullptr) {
                    NPU_VM_LOG_ERROR("Failed to allocate memory for buffer result {}", i);
                    return false;
                }
                resultBuffer.size = static_cast<uint32_t>(buffer.getSize());
            } else {
                // The user provided a buffer for the result, so write the result to the provided buffer. The caller is
                // responsible for ensuring that the provided buffer has sufficient size to hold the result.
                if (resultBuffer.size < buffer.getSize()) {
                    NPU_VM_LOG_ERROR("Provided buffer for result {} is too small. Required size: {}, provided size: {}",
                                     i, buffer.getSize(), resultBuffer.size);
                    return false;
                }
            }
            std::memcpy(resultBuffer.data, buffer.getData(), buffer.getSize());
        } else {
            NPU_VM_LOG_ERROR("Unsupported result type for result {}", i);
            return false;
        }
    }
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)

    return _state.getExecState() == ExecState::Finalized;
}

bool VMEngine::inferImpl(std::string_view functionName, Span<npu_vm_runtime_mem_ref_handle_t> inputMemRefs,
                         Span<npu_vm_runtime_mem_ref_handle_t> outputMemRefs) {
    if (_bytecodeModule == nullptr) {
        NPU_VM_LOG_ERROR("No bytecode module loaded in the engine");
        return false;
    }
    auto function = _bytecodeModule->getFunction(functionName);
    if (function == nullptr) {
        NPU_VM_LOG_ERROR("Could not find entrypoint function with name '{}'", functionName);
        return false;
    }

    const auto funcNumParams = function->getParamTypes().size();
    if (funcNumParams != (inputMemRefs.size() + outputMemRefs.size())) {
        NPU_VM_LOG_ERROR("Number of inputs and outputs passed ({}) does not match the function's parameter count ({})",
                         inputMemRefs.size() + outputMemRefs.size(), funcNumParams);
        return false;
    }

    CallFrame frame(*function, EXIT_RETURN_ADDR, std::vector<int64_t*>{});

    const auto firstInputParamRegIndex = function->getNumGeneralRegisters() - funcNumParams;
    const auto firstOutputParamRegIndex = firstInputParamRegIndex + inputMemRefs.size();

    for (uint32_t i = 0; i < inputMemRefs.size(); ++i) {
        auto handle =
                extractMemRef(inputMemRefs.at(i), _state.getBufferManager(), _state.getBufferMetadata(),
                              Permission::Read, function->getParamTypes().at(i), _bytecodeModule->getTypeByteSizes());
        if (!handle.has_value()) {
            NPU_VM_LOG_ERROR("Failed to extract memref for input {}", i);
            return false;
        }
        frame.setReg(static_cast<int16_t>(firstInputParamRegIndex + i), static_cast<int64_t>(handle.value()));
        NPU_VM_LOG_DEBUG("  Set input {} buffer handle {}", i, handle.value());
    }
    for (uint32_t i = 0; i < outputMemRefs.size(); ++i) {
        const auto outputMemRef = outputMemRefs.at(i);
        auto handle = extractMemRef(outputMemRef, _state.getBufferManager(), _state.getBufferMetadata(),
                                    Permission::ReadWrite, function->getParamTypes().at(i + inputMemRefs.size()),
                                    _bytecodeModule->getTypeByteSizes());
        if (!handle.has_value()) {
            NPU_VM_LOG_ERROR("Failed to extract memref for output {}", i);
            return false;
        }
        frame.setReg(static_cast<int16_t>(firstOutputParamRegIndex + i), static_cast<int64_t>(handle.value()));
        NPU_VM_LOG_DEBUG("  Set output {} buffer handle {}", i, handle.value());
    }

    // The infer method does not return values directly, but instead stores the inference results into output
    // buffers whose handles are passed as parameters
    _state.setExecState(ExecState::Running);
    execute(std::move(frame));

    return _state.getExecState() == ExecState::Finalized;
}

bool VMEngine::infer(npu_vm_runtime_execute_params_t* params) {
    if (_state.getDDITableHandle() == nullptr) {
        setLevelZeroObjects(params->ctx, params->graphDdiTableExt);
    }

    auto& executionContext = _state.getExecutionContext();
    executionContext.setParams(params);

    return inferImpl(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                     Span<npu_vm_runtime_mem_ref_handle_t>(params->pInputs, params->numOfInputs),
                     Span<npu_vm_runtime_mem_ref_handle_t>(params->pOutputs, params->numOfOutputs));
}

bool VMEngine::infer(npu_vm_runtime_execute_params2_t* params) {
    auto& executionContext = _state.getExecutionContext();
    executionContext.setParams(params);

    auto result = inferImpl(NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
                            Span<npu_vm_runtime_mem_ref_handle_t>(params->pInputs, params->numOfInputs),
                            Span<npu_vm_runtime_mem_ref_handle_t>(params->pOutputs, params->numOfOutputs));

    // Update tensor dirty flag to NONE after successful inference,
    // so that the next inference can skip host code execution if the tensor is not modified
    if (result) {
        constexpr uint32_t dirtyFlagNone = static_cast<uint32_t>(MemRef::DirtyFlag::NONE);
        if (params->pInputs) {
            for (uint32_t i = 0; i < params->numOfInputs; ++i) {
                auto inputMemRefHandle = params->pInputs[i];
                auto inputMemRef = reinterpret_cast<npu_vm_runtime_mem_ref*>(inputMemRefHandle);
                inputMemRef->dirtyFlags = dirtyFlagNone;
            }
        }

        if (params->pOutputs) {
            for (uint32_t i = 0; i < params->numOfOutputs; ++i) {
                auto outputMemRefHandle = params->pOutputs[i];
                auto outputMemRef = reinterpret_cast<npu_vm_runtime_mem_ref*>(outputMemRefHandle);
                outputMemRef->dirtyFlags = dirtyFlagNone;
            }
        }
    }

    return result;
}

bool VMEngine::predictOutputShape(npu_vm_runtime_predict_output_shape_params_t2* params) {
    // No Level Zero context here: keep any device manager from a prior execute, else install a host one.
    _state.ensureBufferManager(/*ctx=*/nullptr);
    if (params->numOfOutputs == 0 || params->pOutputs == nullptr) {
        NPU_VM_LOG_ERROR("Number of outputs is zero or output pointer is null. At least one output is required for "
                         "output shape prediction.");
        return false;
    }

    const auto numOutputs = params->numOfOutputs;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* const outputs = reinterpret_cast<npu_vm_runtime_mem_ref**>(params->pOutputs);

    // Backing storage for the predicted shapes: one 1D vector per output, each sized to that output's rank
    std::vector<std::vector<int64_t>> predictedOutputShapes(numOutputs);
    // 1D MemRefs describing predictedOutputShapes, passed to the prediction function as output arguments
    std::vector<npu_vm_runtime_mem_ref> predictedOutputShapeMemrefs(numOutputs, npu_vm_runtime_mem_ref(1));
    std::vector<npu_vm_runtime_mem_ref*> predictedOutputShapeMemrefPtrs(numOutputs);

    for (uint32_t i = 0; i < numOutputs; ++i) {
        const auto* output = *std::next(outputs, static_cast<std::ptrdiff_t>(i));
        if (output == nullptr || output->dimsCount <= 0) {
            NPU_VM_LOG_ERROR("Invalid output tensor");
            return false;
        }

        auto& predictedShape = predictedOutputShapes.at(i);
        predictedShape.assign(static_cast<size_t>(output->dimsCount), 0);

        // Describe the predicted shape as a packed 1D tensor of length dimsCount
        auto& memref = predictedOutputShapeMemrefs.at(i);
        memref.basePtr = memref.data = predictedShape.data();
        memref.sizes.at(0) = output->dimsCount;
        memref.strides.at(0) = 1;

        predictedOutputShapeMemrefPtrs.at(i) = &memref;
    }

    const auto outputStoragePtr =
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            reinterpret_cast<npu_vm_runtime_mem_ref_handle_t*>(predictedOutputShapeMemrefPtrs.data());
    // Inputs and outputs are passed as function arguments to the output shape prediction function
    if (!inferImpl(NPU_VM_OUTPUT_SHAPE_PREDICTION_FUNCTION_NAME,
                   Span<npu_vm_runtime_mem_ref_handle_t>(params->pInputs, params->numOfInputs),
                   Span<npu_vm_runtime_mem_ref_handle_t>(outputStoragePtr, numOutputs))) {
        NPU_VM_LOG_ERROR("Output shape prediction function execution failed.");
        return false;
    }

    // Write the predicted shapes and matching packed strides back into the caller-provided memrefs
    for (uint32_t i = 0; i < numOutputs; ++i) {
        auto* output = *std::next(outputs, static_cast<std::ptrdiff_t>(i));
        const auto& predictedShape = predictedOutputShapes.at(i);

        output->sizes = predictedShape;

        // Packed (row-major) strides: the innermost dimension has stride 1 and each outer
        // dimension's stride is the product of the sizes to its right
        std::vector<int64_t> strides(predictedShape.size());
        int64_t stride = 1;
        for (size_t dim = predictedShape.size(); dim-- > 0;) {
            strides.at(dim) = stride;
            stride *= predictedShape.at(dim);
        }
        output->strides = std::move(strides);
    }

    return true;
}

void VMEngine::incrementPC(size_t bytes) {
    if (_state.getExecState() != ExecState::Running) {
        return;
    }
    const auto pc = _state.getPC();
    _state.setPC(std::next(pc, static_cast<std::ptrdiff_t>(bytes)));
    NPU_VM_LOG_DEBUG("  pc: {}", static_cast<const void*>(_state.getPC()));
}

void VMEngine::execute(CallFrame entryFrame) {
    // maxCallDepth bounds the number of nested calls. Reserving the worst-case capacity up front guarantees the
    // call stack never reallocates, so the caller register buffers that callee result sinks point into stay at
    // stable addresses for the whole call chain. Each frame carries the function it executes and the result sinks
    // that RETV writes into, so arbitrarily deep bytecode call chains run on this heap-allocated stack instead of
    // the native C++ call stack.
    constexpr size_t maxCallDepth = 1000;
    std::vector<CallFrame> callStack;
    callStack.reserve(maxCallDepth + 1);
    callStack.push_back(std::move(entryFrame));

    const Function& entryFunction = callStack.back().getFunction();
    _state.setPC(entryFunction.getBody().begin());

    NPU_VM_LOG_DEBUG("Executing function '{}' with {} parameter(s) and {} result(s)", entryFunction.getName(),
                     entryFunction.getParamTypes().size(), entryFunction.getResultTypes().size());
    NPU_VM_LOG_DEBUG("  call depth: {}", callStack.size() - 1);
    NPU_VM_LOG_DEBUG("  pc: {}", static_cast<const void*>(_state.getPC()));

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)

    // Main dispatch loop: decode the opcode at the current program counter, execute the corresponding operation, then
    // advance the PC by the instruction size. CALL/RET/RETV mutate the call stack rather than recursing. The
    // loop exits when the call stack drains or the state transitions away from Running.
    while (_state.getExecState() == ExecState::Running && !callStack.empty()) {
        CallFrame& frame = callStack.back();
        const Function& function = frame.getFunction();
        std::vector<int64_t*>& results = frame.getResults();

        const auto functionBodyStart = function.getBody().begin();
        const auto endOfFunction = function.getBody().end();

        const auto pc = _state.getPC();
        if (pc >= endOfFunction) {
            // The function body ran out before an explicit RET/RETV. Well-formed bytecode always returns, so treat a
            // run past the end as a malformed-body error rather than silently unwinding.
            NPU_VM_LOG_ERROR("Reached end of function body '{}' without a return instruction. This likely means the "
                             "function body is malformed.",
                             function.getName());
            _state.setExecState(ExecState::Halted);
            break;
        }
        const auto bytesAvailable = static_cast<size_t>(endOfFunction - pc);
        OpCode opcode{};
        size_t instructionSize{};
        if (!decodeInstruction(pc, bytesAvailable, opcode, instructionSize)) {
            _state.setExecState(ExecState::Halted);
            break;
        }

        switch (opcode) {
        case OpCode::ADD_I64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            // Cast to uint64_t first to avoid UB from signed overflow, then cast back to int64_t after the addition
            const auto lhs = static_cast<uint64_t>(frame.getReg(srcReg1Num));
            const auto rhs = static_cast<uint64_t>(frame.getReg(srcReg2Num));
            frame.setReg(dstRegNum, static_cast<int64_t>(lhs + rhs));
            NPU_VM_LOG_DEBUG("  add.i64 {}, {}, {} ({} + {} = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             frame.getReg(srcReg1Num), frame.getReg(srcReg2Num), frame.getReg(dstRegNum));
            break;
        }
        case OpCode::SET: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            frame.setReg(dstRegNum, frame.getReg(srcRegNum));
            NPU_VM_LOG_DEBUG("  set {}, {} (reg[{}] = {})", dstRegNum, srcRegNum, dstRegNum, frame.getReg(srcRegNum));
            break;
        }
        case OpCode::SET_IMM: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto immValue = get64BitImm(pc, 1);
            frame.setReg(dstRegNum, immValue);
            NPU_VM_LOG_DEBUG("  set.imm {}, {} (reg[{}] = {})", dstRegNum, immValue, dstRegNum, immValue);
            break;
        }
        case OpCode::ASSERT: {
            const auto conditionRegNum = getOperand(pc, 0);
            const auto msgSymIndex = getOperand(pc, 1);
            NPU_VM_LOG_DEBUG("  assert {}, {}", conditionRegNum, msgSymIndex);
            if (frame.getReg(conditionRegNum) == 0) {
                const auto& rawStrings = _bytecodeModule->getRawStrings();
                if (msgSymIndex >= 0 && static_cast<size_t>(msgSymIndex) < rawStrings.size()) {
                    const auto rawString = rawStrings.at(msgSymIndex).get();
                    std::string message(rawString.begin(), rawString.end());
                    NPU_VM_LOG_ERROR("Assertion failed: {}", message);
                } else {
                    NPU_VM_LOG_ERROR("Assertion failed: ? (message not found; index {})", msgSymIndex);
                }
                _state.setExecState(ExecState::Halted);
            }
            break;
        }
        case OpCode::MUL_I64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            // Cast to uint64_t first to avoid UB from signed overflow, then cast back to int64_t after the
            // multiplication
            const auto lhs = static_cast<uint64_t>(frame.getReg(srcReg1Num));
            const auto rhs = static_cast<uint64_t>(frame.getReg(srcReg2Num));
            frame.setReg(dstRegNum, static_cast<int64_t>(lhs * rhs));
            NPU_VM_LOG_DEBUG("  mul.i64 {}, {}, {} ({} * {} = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             frame.getReg(srcReg1Num), frame.getReg(srcReg2Num), frame.getReg(dstRegNum));
            break;
        }
        case OpCode::MIN_I64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            frame.setReg(dstRegNum, std::min(frame.getReg(srcReg1Num), frame.getReg(srcReg2Num)));
            NPU_VM_LOG_DEBUG("  min.i64 {}, {}, {} (min({}, {}) = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             frame.getReg(srcReg1Num), frame.getReg(srcReg2Num), frame.getReg(dstRegNum));
            break;
        }
        case OpCode::MAX_I64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            frame.setReg(dstRegNum, std::max(frame.getReg(srcReg1Num), frame.getReg(srcReg2Num)));
            NPU_VM_LOG_DEBUG("  max.i64 {}, {}, {} (max({}, {}) = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             frame.getReg(srcReg1Num), frame.getReg(srcReg2Num), frame.getReg(dstRegNum));
            break;
        }
        case OpCode::CMP_I64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            const auto flag = static_cast<uint16_t>(getOperand(pc, 3));
            const auto cmpType = static_cast<uint8_t>(flag & 0xFF);
            const bool isSigned = (flag >> 8) & 0x1;
            const auto result = isSigned ? evaluateCmp(frame.getReg(srcReg1Num), frame.getReg(srcReg2Num), cmpType)
                                         : evaluateCmp(static_cast<uint64_t>(frame.getReg(srcReg1Num)),
                                                       static_cast<uint64_t>(frame.getReg(srcReg2Num)), cmpType);
            if (!result.has_value()) {
                _state.setExecState(ExecState::Halted);
                break;
            }
            frame.setReg(dstRegNum, *result);
            NPU_VM_LOG_DEBUG("  cmp.i64 {}, {}, {}, {} (cmp({}, {}) = {})", dstRegNum, srcReg1Num, srcReg2Num, flag,
                             isSigned ? frame.getReg(srcReg1Num) : static_cast<uint64_t>(frame.getReg(srcReg1Num)),
                             isSigned ? frame.getReg(srcReg2Num) : static_cast<uint64_t>(frame.getReg(srcReg2Num)),
                             *result);
            break;
        }
        case OpCode::CMP_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            const auto flag = static_cast<uint16_t>(getOperand(pc, 3));
            const auto cmpType = static_cast<uint8_t>(flag & 0xFF);
            double lhs = 0.0, rhs = 0.0;
            std::memcpy(&lhs, &frame.getReg(srcReg1Num), sizeof(double));
            std::memcpy(&rhs, &frame.getReg(srcReg2Num), sizeof(double));
            const auto result = evaluateCmp(lhs, rhs, cmpType);
            if (!result.has_value()) {
                _state.setExecState(ExecState::Halted);
                break;
            }
            frame.setReg(dstRegNum, *result);
            NPU_VM_LOG_DEBUG("  cmp.f64 {}, {}, {}, {} (cmp({}, {}) = {})", dstRegNum, srcReg1Num, srcReg2Num, flag,
                             lhs, rhs, *result);
            break;
        }
        case OpCode::SUB_I64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            // Cast to uint64_t first to avoid UB from signed overflow, then cast back to int64_t after the subtraction
            const auto lhs = static_cast<uint64_t>(frame.getReg(srcReg1Num));
            const auto rhs = static_cast<uint64_t>(frame.getReg(srcReg2Num));
            frame.setReg(dstRegNum, static_cast<int64_t>(lhs - rhs));
            NPU_VM_LOG_DEBUG("  sub.i64 {}, {}, {} ({} - {} = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             frame.getReg(srcReg1Num), frame.getReg(srcReg2Num), frame.getReg(dstRegNum));
            break;
        }
        case OpCode::DIV_I64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            const auto rhs = frame.getReg(srcReg2Num);
            if (rhs == 0) {
                NPU_VM_LOG_ERROR("Division by zero error in DIV_I64 instruction");
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto lhs = frame.getReg(srcReg1Num);
            if (lhs == std::numeric_limits<int64_t>::min() && rhs == -1) {
                // Handle overflow case where result cannot be represented as int64_t. In this case, the result of the
                // division operation is defined to be INT64_MIN (INT64_MAX + 1)
                frame.setReg(dstRegNum, std::numeric_limits<int64_t>::min());

            } else {
                frame.setReg(dstRegNum, lhs / rhs);
            }
            NPU_VM_LOG_DEBUG("  div.i64 {}, {}, {} ({} / {} = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             frame.getReg(srcReg1Num), frame.getReg(srcReg2Num), frame.getReg(dstRegNum));
            break;
        }
        case OpCode::DIV_U64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            const auto rhs = static_cast<uint64_t>(frame.getReg(srcReg2Num));
            if (rhs == 0) {
                NPU_VM_LOG_ERROR("Division by zero error in DIV_U64 instruction");
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto lhs = static_cast<uint64_t>(frame.getReg(srcReg1Num));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(lhs / rhs));
            NPU_VM_LOG_DEBUG("  div.u64 {}, {}, {} ({} / {} = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             static_cast<uint64_t>(frame.getReg(srcReg1Num)),
                             static_cast<uint64_t>(frame.getReg(srcReg2Num)),
                             static_cast<uint64_t>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::MIN_U64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            const auto lhs = static_cast<uint64_t>(frame.getReg(srcReg1Num));
            const auto rhs = static_cast<uint64_t>(frame.getReg(srcReg2Num));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(std::min(lhs, rhs)));
            NPU_VM_LOG_DEBUG("  min.u64 {}, {}, {} (min({}, {}) = {})", dstRegNum, srcReg1Num, srcReg2Num, lhs, rhs,
                             static_cast<uint64_t>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::ADD_U64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            const auto lhs = static_cast<uint64_t>(frame.getReg(srcReg1Num));
            const auto rhs = static_cast<uint64_t>(frame.getReg(srcReg2Num));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(lhs + rhs));
            NPU_VM_LOG_DEBUG("  add.u64 {}, {}, {} ({} + {} = {})", dstRegNum, srcReg1Num, srcReg2Num, lhs, rhs,
                             static_cast<uint64_t>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::MAX_U64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            const auto lhs = static_cast<uint64_t>(frame.getReg(srcReg1Num));
            const auto rhs = static_cast<uint64_t>(frame.getReg(srcReg2Num));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(std::max(lhs, rhs)));
            NPU_VM_LOG_DEBUG("  max.u64 {}, {}, {} (max({}, {}) = {})", dstRegNum, srcReg1Num, srcReg2Num, lhs, rhs,
                             static_cast<uint64_t>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::MUL_U64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            const auto lhs = static_cast<uint64_t>(frame.getReg(srcReg1Num));
            const auto rhs = static_cast<uint64_t>(frame.getReg(srcReg2Num));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(lhs * rhs));
            NPU_VM_LOG_DEBUG("  mul.u64 {}, {}, {} ({} * {} = {})", dstRegNum, srcReg1Num, srcReg2Num, lhs, rhs,
                             static_cast<uint64_t>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::REM_U64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            const auto rhs = static_cast<uint64_t>(frame.getReg(srcReg2Num));
            if (rhs == 0) {
                NPU_VM_LOG_ERROR("Division by zero error in REM_U64 instruction");
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto lhs = static_cast<uint64_t>(frame.getReg(srcReg1Num));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(lhs % rhs));
            NPU_VM_LOG_DEBUG("  rem.u64 {}, {}, {} ({} %% {} = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             static_cast<uint64_t>(frame.getReg(srcReg1Num)),
                             static_cast<uint64_t>(frame.getReg(srcReg2Num)),
                             static_cast<uint64_t>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::SUB_U64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            const auto lhs = static_cast<uint64_t>(frame.getReg(srcReg1Num));
            const auto rhs = static_cast<uint64_t>(frame.getReg(srcReg2Num));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(lhs - rhs));
            NPU_VM_LOG_DEBUG("  sub.u64 {}, {}, {} ({} - {} = {})", dstRegNum, srcReg1Num, srcReg2Num, lhs, rhs,
                             static_cast<uint64_t>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::REM_I64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            const auto rhs = frame.getReg(srcReg2Num);
            if (rhs == 0) {
                NPU_VM_LOG_ERROR("Division by zero error in REM_I64 instruction");
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto lhs = frame.getReg(srcReg1Num);
            if (lhs == std::numeric_limits<int64_t>::min() && rhs == -1) {
                // Handle overflow case where result cannot be represented as int64_t. In this case, the
                // result of the remainder operation is defined to be 0
                frame.setReg(dstRegNum, 0);
            } else {
                frame.setReg(dstRegNum, lhs % rhs);
            }
            NPU_VM_LOG_DEBUG("  rem.i64 {}, {}, {} ({} % {} = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             frame.getReg(srcReg1Num), frame.getReg(srcReg2Num), frame.getReg(dstRegNum));
            break;
        }
        case OpCode::ABS_I64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto value = frame.getReg(srcRegNum);
            if (value == std::numeric_limits<int64_t>::min()) {
                // Handle overflow case where abs(INT64_MIN) cannot be represented in an int64_t. In this case, the
                // result of the absolute value operation is defined to be INT64_MIN.
                frame.setReg(dstRegNum, std::numeric_limits<int64_t>::min());
            } else {
                frame.setReg(dstRegNum, std::abs(value));
            }
            NPU_VM_LOG_DEBUG("  abs.i64 {}, {} (abs({}) = {})", dstRegNum, srcRegNum, frame.getReg(srcRegNum),
                             frame.getReg(dstRegNum));
            break;
        }
        case OpCode::AND_64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            frame.setReg(dstRegNum, frame.getReg(srcReg1Num) & frame.getReg(srcReg2Num));
            NPU_VM_LOG_DEBUG("  and.64 {}, {}, {} ({} & {} = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             frame.getReg(srcReg1Num), frame.getReg(srcReg2Num), frame.getReg(dstRegNum));
            break;
        }
        case OpCode::NOT_64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            frame.setReg(dstRegNum, ~frame.getReg(srcRegNum));
            NPU_VM_LOG_DEBUG("  not.64 {}, {} (~{} = {})", dstRegNum, srcRegNum, frame.getReg(srcRegNum),
                             frame.getReg(dstRegNum));
            break;
        }
        case OpCode::OR_64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            frame.setReg(dstRegNum, frame.getReg(srcReg1Num) | frame.getReg(srcReg2Num));
            NPU_VM_LOG_DEBUG("  or.64 {}, {}, {} ({} | {} = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             frame.getReg(srcReg1Num), frame.getReg(srcReg2Num), frame.getReg(dstRegNum));
            break;
        }
        case OpCode::XOR_64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            frame.setReg(dstRegNum, frame.getReg(srcReg1Num) ^ frame.getReg(srcReg2Num));
            NPU_VM_LOG_DEBUG("  xor.64 {}, {}, {} ({} ^ {} = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             frame.getReg(srcReg1Num), frame.getReg(srcReg2Num), frame.getReg(dstRegNum));
            break;
        }
        case OpCode::SLL_64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            // Force the shift amount into the valid 0-63 range
            const auto shiftAmount = frame.getReg(srcReg2Num) & 63;
            const auto unsignedResult = static_cast<uint64_t>(frame.getReg(srcReg1Num)) << shiftAmount;
            const auto signedResult = convertUnsignedToSignedPreserveBits<int64_t>(unsignedResult);
            frame.setReg(dstRegNum, signedResult);
            NPU_VM_LOG_DEBUG("  sll.64 {}, {}, {} ({} << {} = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             frame.getReg(srcReg1Num), shiftAmount, signedResult);
            break;
        }
        case OpCode::SRL_64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            // Force the shift amount into the valid 0-63 range
            const auto shiftAmount = frame.getReg(srcReg2Num) & 63;
            const auto unsignedResult = static_cast<uint64_t>(frame.getReg(srcReg1Num)) >> shiftAmount;
            const auto signedResult = convertUnsignedToSignedPreserveBits<int64_t>(unsignedResult);
            frame.setReg(dstRegNum, signedResult);
            NPU_VM_LOG_DEBUG("  srl.64 {}, {}, {} ({} >> {} = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             frame.getReg(srcReg1Num), shiftAmount, signedResult);
            break;
        }
        case OpCode::SRA_64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcReg1Num = getOperand(pc, 1);
            const auto srcReg2Num = getOperand(pc, 2);
            // Arithmetic right shift with defined behavior for negative operands.
            // Returns int64_t directly; no sign extension needed.
            const auto shiftedValue =
                    arithmeticShiftRightDefined<int64_t>(frame.getReg(srcReg1Num), frame.getReg(srcReg2Num));
            frame.setReg(dstRegNum, shiftedValue);
            NPU_VM_LOG_DEBUG("  sra.64 {}, {}, {} ({} >> {} = {})", dstRegNum, srcReg1Num, srcReg2Num,
                             frame.getReg(srcReg1Num), (frame.getReg(srcReg2Num) & 63), frame.getReg(dstRegNum));
            break;
        }
        case OpCode::SELECT: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto condRegNum = getOperand(pc, 1);
            const auto trueRegNum = getOperand(pc, 2);
            const auto falseRegNum = getOperand(pc, 3);
            const auto result = frame.getReg(condRegNum) != 0 ? frame.getReg(trueRegNum) : frame.getReg(falseRegNum);
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  select {}, {}, {}, {} (if {} then {} else {} = {})", dstRegNum, condRegNum, trueRegNum,
                             falseRegNum, frame.getReg(condRegNum), frame.getReg(trueRegNum), frame.getReg(falseRegNum),
                             result);
            break;
        }
        case OpCode::ADD_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto lhsRegNum = getOperand(pc, 1);
            const auto rhsRegNum = getOperand(pc, 2);
            const auto lhs = vmBitCast<double>(frame.getReg(lhsRegNum));
            const auto rhs = vmBitCast<double>(frame.getReg(rhsRegNum));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(lhs + rhs));
            NPU_VM_LOG_DEBUG("  add.f64 {}, {}, {} ({} + {} = {})", dstRegNum, lhsRegNum, rhsRegNum, lhs, rhs,
                             vmBitCast<double>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::SUB_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto lhsRegNum = getOperand(pc, 1);
            const auto rhsRegNum = getOperand(pc, 2);
            const auto lhs = vmBitCast<double>(frame.getReg(lhsRegNum));
            const auto rhs = vmBitCast<double>(frame.getReg(rhsRegNum));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(lhs - rhs));
            NPU_VM_LOG_DEBUG("  sub.f64 {}, {}, {} ({} - {} = {})", dstRegNum, lhsRegNum, rhsRegNum, lhs, rhs,
                             vmBitCast<double>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::MUL_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto lhsRegNum = getOperand(pc, 1);
            const auto rhsRegNum = getOperand(pc, 2);
            const auto lhs = vmBitCast<double>(frame.getReg(lhsRegNum));
            const auto rhs = vmBitCast<double>(frame.getReg(rhsRegNum));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(lhs * rhs));
            NPU_VM_LOG_DEBUG("  mul.f64 {}, {}, {} ({} * {} = {})", dstRegNum, lhsRegNum, rhsRegNum, lhs, rhs,
                             vmBitCast<double>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::DIV_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto lhsRegNum = getOperand(pc, 1);
            const auto rhsRegNum = getOperand(pc, 2);
            const auto lhs = vmBitCast<double>(frame.getReg(lhsRegNum));
            const auto rhs = vmBitCast<double>(frame.getReg(rhsRegNum));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(lhs / rhs));
            NPU_VM_LOG_DEBUG("  div.f64 {}, {}, {} ({} / {} = {})", dstRegNum, lhsRegNum, rhsRegNum, lhs, rhs,
                             vmBitCast<double>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::REM_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto lhsRegNum = getOperand(pc, 1);
            const auto rhsRegNum = getOperand(pc, 2);
            const auto lhs = vmBitCast<double>(frame.getReg(lhsRegNum));
            const auto rhs = vmBitCast<double>(frame.getReg(rhsRegNum));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(std::fmod(lhs, rhs)));
            NPU_VM_LOG_DEBUG("  rem.f64 {}, {}, {} ({} % {} = {})", dstRegNum, lhsRegNum, rhsRegNum, lhs, rhs,
                             vmBitCast<double>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::MAX_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto lhsRegNum = getOperand(pc, 1);
            const auto rhsRegNum = getOperand(pc, 2);
            const auto lhs = vmBitCast<double>(frame.getReg(lhsRegNum));
            const auto rhs = vmBitCast<double>(frame.getReg(rhsRegNum));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(std::fmax(lhs, rhs)));
            NPU_VM_LOG_DEBUG("  max.f64 {}, {}, {} (max({}, {}) = {})", dstRegNum, lhsRegNum, rhsRegNum, lhs, rhs,
                             vmBitCast<double>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::MIN_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto lhsRegNum = getOperand(pc, 1);
            const auto rhsRegNum = getOperand(pc, 2);
            const auto lhs = vmBitCast<double>(frame.getReg(lhsRegNum));
            const auto rhs = vmBitCast<double>(frame.getReg(rhsRegNum));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(std::fmin(lhs, rhs)));
            NPU_VM_LOG_DEBUG("  min.f64 {}, {}, {} (min({}, {}) = {})", dstRegNum, lhsRegNum, rhsRegNum, lhs, rhs,
                             vmBitCast<double>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::ABS_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto src = vmBitCast<double>(frame.getReg(srcRegNum));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(std::fabs(src)));
            NPU_VM_LOG_DEBUG("  abs.f64 {}, {} (abs({}) = {})", dstRegNum, srcRegNum, src,
                             vmBitCast<double>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::NEG_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto src = vmBitCast<double>(frame.getReg(srcRegNum));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(-src));
            NPU_VM_LOG_DEBUG("  neg.f64 {}, {} (-{} = {})", dstRegNum, srcRegNum, src,
                             vmBitCast<double>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::CEIL_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto src = vmBitCast<double>(frame.getReg(srcRegNum));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(std::ceil(src)));
            NPU_VM_LOG_DEBUG("  ceil.f64 {}, {} (ceil({}) = {})", dstRegNum, srcRegNum, src,
                             vmBitCast<double>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::FLOOR_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto src = vmBitCast<double>(frame.getReg(srcRegNum));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(std::floor(src)));
            NPU_VM_LOG_DEBUG("  floor.f64 {}, {} (floor({}) = {})", dstRegNum, srcRegNum, src,
                             vmBitCast<double>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::ROUND_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto mode = static_cast<RoundingMode>(getOperand(pc, 2));
            const auto src = vmBitCast<double>(frame.getReg(srcRegNum));
            double result{};
            switch (mode) {
            case RoundingMode::RNE: {
                // Change the rounding mode to FE_TONEAREST for the call to std::nearbyint, then restore the original
                // rounding mode. The change of the rounding mode in the environment is thread local
                const auto origRoundMode = std::fegetround();
                if (std::fesetround(FE_TONEAREST) != 0) {
                    NPU_VM_LOG_ERROR("Failed to set rounding mode to FE_TONEAREST for round.f64 instruction");
                    _state.setExecState(ExecState::Halted);
                    break;
                }
                result = std::nearbyint(src);
                if (std::fesetround(origRoundMode) != 0) {
                    NPU_VM_LOG_ERROR("Failed to restore original rounding mode after round.f64 instruction");
                    _state.setExecState(ExecState::Halted);
                    break;
                }
                break;
            }
            case RoundingMode::RNA: {
                result = std::round(src);
                break;
            }
            case RoundingMode::RDN:
                result = std::floor(src);
                break;
            case RoundingMode::RUP:
                result = std::ceil(src);
                break;
            case RoundingMode::RTZ:
                result = std::trunc(src);
                break;
            default:
                NPU_VM_LOG_ERROR("Unknown rounding mode {} in round.f64 instruction", static_cast<int16_t>(mode));
                _state.setExecState(ExecState::Halted);
                break;
            }
            if (_state.getExecState() == ExecState::Halted) {
                break;
            }
            frame.setReg(dstRegNum, vmBitCast<int64_t>(result));
            NPU_VM_LOG_DEBUG("  round.f64 {}, {}, {} (round({}) = {})", dstRegNum, srcRegNum,
                             static_cast<int16_t>(mode), src, vmBitCast<double>(frame.getReg(dstRegNum)));
            break;
        }
        case OpCode::BUFFER_CREATE: {
            const auto dstReg = getOperand(pc, 0);
            const auto elemTypeIdx = static_cast<int64_t>(getOperand(pc, 1));
            const auto rank = static_cast<int64_t>(getOperand(pc, 2));
            if (rank < 0) {
                NPU_VM_LOG_ERROR("buffer.create requires non-negative rank, got {}", rank);
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto elemByteSize =
                    intel_npu::vm::lookupTypeByteSize(_bytecodeModule->getTypeByteSizes(), elemTypeIdx);
            if (elemByteSize == 0) {
                NPU_VM_LOG_ERROR("buffer.create references unknown or non-primitive type index {}, type table size={}",
                                 elemTypeIdx, _bytecodeModule->getTypeByteSizes().size());
                _state.setExecState(ExecState::Halted);
                break;
            }
            BufferMetadata meta;
            meta.elemTypeIndex = elemTypeIdx;
            meta.elemByteSize = elemByteSize;
            meta.shape.reserve(rank);
            meta.strides.reserve(rank);
            for (int64_t i = 0; i < rank; ++i) {
                meta.shape.push_back(frame.getReg(getOperand(pc, 3 + i)));
            }
            for (int64_t i = 0; i < rank; ++i) {
                meta.strides.push_back(frame.getReg(getOperand(pc, 3 + rank + i)));
            }
            if (!intel_npu::vm::validateShapeAndStrides("buffer.create", meta)) {
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto byteSize = intel_npu::vm::computeLogicalByteSize(meta);
            if (!byteSize.has_value()) {
                NPU_VM_LOG_ERROR("buffer.create size overflows the addressable range, shape={}, strides={}, element "
                                 "byte size={}",
                                 intel_npu::vm::formatVector(meta.shape), intel_npu::vm::formatVector(meta.strides),
                                 meta.elemByteSize);
                _state.setExecState(ExecState::Halted);
                break;
            }
            try {
                const auto handle = _state.getBufferManager().create(static_cast<size_t>(*byteSize),
                                                                     Permission::ReadWrite, /*zeroInitialized*/ true);
                _state.getBufferMetadata().emplace(handle, std::move(meta));
                frame.setReg(dstReg, static_cast<int64_t>(handle));
                NPU_VM_LOG_DEBUG("  buffer.create -> handle {}, ({} bytes)", handle, *byteSize);
            } catch (const std::exception& e) {
                NPU_VM_LOG_ERROR("buffer.create failed: {}", e.what());
                _state.setExecState(ExecState::Halted);
            }
            break;
        }
        case OpCode::BUFFER_FROM_CONST: {
            const auto dstReg = getOperand(pc, 0);
            const auto constantIndex = getOperand(pc, 1);
            const auto elemTypeIdx = static_cast<int64_t>(getOperand(pc, 2));
            const auto rank = static_cast<int64_t>(getOperand(pc, 3));
            if (constantIndex < 0 || rank < 0 || rank > intel_npu::vm::BufferType::MAX_RANK) {
                NPU_VM_LOG_ERROR(
                        "buffer.from_const requires non-negative constant index and rank in [0, {}], got {} and {}",
                        intel_npu::vm::BufferType::MAX_RANK, constantIndex, rank);
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto elemByteSize =
                    intel_npu::vm::lookupTypeByteSize(_bytecodeModule->getTypeByteSizes(), elemTypeIdx);
            const auto& constants = _constantPool.getConstants();
            if (elemByteSize == 0 || static_cast<size_t>(constantIndex) >= constants.size()) {
                NPU_VM_LOG_ERROR("buffer.from_const references unknown type {} or constant {}", elemTypeIdx,
                                 constantIndex);
                _state.setExecState(ExecState::Halted);
                break;
            }
            BufferMetadata meta;
            meta.elemTypeIndex = elemTypeIdx;
            meta.elemByteSize = elemByteSize;
            meta.shape.reserve(rank);
            meta.strides.reserve(rank);
            for (int64_t i = 0; i < rank; ++i) {
                meta.shape.push_back(frame.getReg(getOperand(pc, 4 + i)));
                meta.strides.push_back(frame.getReg(getOperand(pc, 4 + rank + i)));
            }
            if (!intel_npu::vm::validateShapeAndStrides("buffer.from_const", meta)) {
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto byteSize = intel_npu::vm::computeLogicalByteSize(meta);
            const auto constantData = constants.at(static_cast<size_t>(constantIndex));
            if (!byteSize.has_value() || *byteSize != constantData.size()) {
                NPU_VM_LOG_ERROR("buffer.from_const descriptor size {} does not match constant {} size {}",
                                 byteSize.has_value() ? std::to_string(*byteSize) : "<overflow>", constantIndex,
                                 constantData.size());
                _state.setExecState(ExecState::Halted);
                break;
            }
            try {
                const auto handle = _state.getBufferManager().createFromMemory(
                        const_cast<uint8_t*>(constantData.data()), constantData.size(), Permission::Read);
                _state.getBufferMetadata().emplace(handle, std::move(meta));
                frame.setReg(dstReg, static_cast<int64_t>(handle));
                NPU_VM_LOG_DEBUG("  buffer.from_const -> handle {}, constant {}", handle, constantIndex);
            } catch (const std::exception& e) {
                NPU_VM_LOG_ERROR("buffer.from_const failed: {}", e.what());
                _state.setExecState(ExecState::Halted);
            }
            break;
        }
        case OpCode::BUFFER_GET_DIM: {
            const auto dstReg = getOperand(pc, 0);
            const auto bufReg = getOperand(pc, 1);
            const auto dimReg = getOperand(pc, 2);
            const auto handle = static_cast<BufferHandle>(frame.getReg(bufReg));
            const auto dim = frame.getReg(dimReg);
            auto& bufferManager = _state.getBufferManager();
            auto& bufferMetadata = _state.getBufferMetadata();
            const auto it = bufferMetadata.find(handle);
            if (it == bufferMetadata.end() || !bufferManager.exists(handle)) {
                NPU_VM_LOG_ERROR("buffer.get_dim on unknown buffer handle {}", handle);
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto& shape = it->second.shape;
            if (dim < 0 || static_cast<size_t>(dim) >= shape.size()) {
                NPU_VM_LOG_ERROR("buffer.get_dim dimension index {} is out of range for buffer {} with rank {}", dim,
                                 handle, shape.size());
                _state.setExecState(ExecState::Halted);
                break;
            }
            frame.setReg(dstReg, shape.at(static_cast<size_t>(dim)));
            NPU_VM_LOG_DEBUG("  buffer.get_dim {}, {}, {} (source handle {}, dim {}, value {})", dstReg, bufReg, dimReg,
                             handle, dim, frame.getReg(dstReg));
            break;
        }
        case OpCode::BUFFER_SUBVIEW: {
            const auto dstReg = getOperand(pc, 0);
            const auto srcReg = getOperand(pc, 1);
            const auto rank = static_cast<int64_t>(getOperand(pc, 2));
            if (rank < 0) {
                NPU_VM_LOG_ERROR("buffer.subview requires non-negative rank, got {}", rank);
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto srcHandle = static_cast<BufferHandle>(frame.getReg(srcReg));
            auto& bufferManager = _state.getBufferManager();
            auto& bufferMetadata = _state.getBufferMetadata();
            auto srcIt = bufferMetadata.find(srcHandle);
            if (srcIt == bufferMetadata.end() || !bufferManager.exists(srcHandle) ||
                static_cast<int64_t>(srcIt->second.shape.size()) != rank ||
                static_cast<int64_t>(srcIt->second.strides.size()) != rank) {
                NPU_VM_LOG_ERROR("buffer.subview invalid handle or rank mismatch: handle={}, encoded rank={}, buffer "
                                 "rank={}, stride rank={}",
                                 srcHandle, rank, srcIt != bufferMetadata.end() ? srcIt->second.shape.size() : 0,
                                 srcIt != bufferMetadata.end() ? srcIt->second.strides.size() : 0);
                _state.setExecState(ExecState::Halted);
                break;
            }
            std::vector<int64_t> offsets(rank), sizes(rank), strides(rank);
            for (int64_t i = 0; i < rank; ++i) {
                offsets.at(i) = frame.getReg(getOperand(pc, 3 + i));
            }
            for (int64_t i = 0; i < rank; ++i) {
                sizes.at(i) = frame.getReg(getOperand(pc, 3 + rank + i));
            }
            for (int64_t i = 0; i < rank; ++i) {
                strides.at(i) = frame.getReg(getOperand(pc, 3 + 2 * rank + i));
            }
            const auto startElem = computeSubviewStartElement(srcIt->second, offsets, sizes, strides);
            if (!startElem.has_value()) {
                _state.setExecState(ExecState::Halted);
                break;
            }

            BufferMetadata newMeta;
            newMeta.elemTypeIndex = srcIt->second.elemTypeIndex;
            newMeta.elemByteSize = srcIt->second.elemByteSize;
            newMeta.shape = std::move(sizes);
            newMeta.strides.reserve(rank);
            for (int64_t i = 0; i < rank; ++i) {
                const auto dimIdx = static_cast<size_t>(i);
                int64_t composedStride = 0;
                if (!checkedMultiplyNonNegative(strides.at(dimIdx), srcIt->second.strides.at(dimIdx), composedStride)) {
                    NPU_VM_LOG_ERROR("buffer.subview stride overflows at dim {}: subview stride={}, source stride={}",
                                     i, strides.at(dimIdx), srcIt->second.strides.at(dimIdx));
                    _state.setExecState(ExecState::Halted);
                    break;
                }
                newMeta.strides.push_back(composedStride);
            }
            if (_state.getExecState() == ExecState::Halted) {
                break;
            }

            try {
                uint64_t byteOffset = 0;
                const auto byteSize = computeLogicalByteSize(newMeta);
                const auto elemByteSize = static_cast<uint64_t>(srcIt->second.elemByteSize);
                if (!byteSize.has_value() || !checkedMultiply(*startElem, elemByteSize, byteOffset)) {
                    NPU_VM_LOG_ERROR("buffer.subview byte range overflows for source handle {}: start element={}, "
                                     "element byte size={}, byte offset={}, byte size={}",
                                     srcHandle, *startElem, elemByteSize, byteOffset,
                                     byteSize.has_value() ? std::to_string(*byteSize) : "<overflow>");
                    _state.setExecState(ExecState::Halted);
                    break;
                }
                const auto newHandle = bufferManager.createFromBuffer(srcHandle, byteOffset, *byteSize);
                bufferMetadata.emplace(newHandle, std::move(newMeta));
                frame.setReg(dstReg, static_cast<int64_t>(newHandle));
                NPU_VM_LOG_DEBUG("  buffer.subview -> handle {}, (source handle {}, byteOffset {}, byteSize {})",
                                 newHandle, srcHandle, byteOffset, *byteSize);
            } catch (const std::exception& e) {
                NPU_VM_LOG_ERROR("buffer.subview failed: {}", e.what());
                _state.setExecState(ExecState::Halted);
            }
            break;
        }
        case OpCode::BUFFER_VIEW: {
            const auto dstReg = getOperand(pc, 0);
            const auto srcReg = getOperand(pc, 1);
            const auto byteOffsetReg = getOperand(pc, 2);
            const auto elemTypeIdx = static_cast<int64_t>(getOperand(pc, 3));
            const auto rank = static_cast<int64_t>(getOperand(pc, 4));
            if (rank < 0) {
                NPU_VM_LOG_ERROR("buffer.view requires non-negative rank, got {}", rank);
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto srcHandle = static_cast<BufferHandle>(frame.getReg(srcReg));
            auto& bufferManager = _state.getBufferManager();
            auto& bufferMetadata = _state.getBufferMetadata();
            const auto srcIt = bufferMetadata.find(srcHandle);
            if (srcIt == bufferMetadata.end() || !bufferManager.exists(srcHandle)) {
                NPU_VM_LOG_ERROR("buffer.view on unknown buffer handle {}", srcHandle);
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto byteOffset = static_cast<uint64_t>(frame.getReg(byteOffsetReg));
            const auto elemByteSize =
                    intel_npu::vm::lookupTypeByteSize(_bytecodeModule->getTypeByteSizes(), elemTypeIdx);
            if (elemByteSize == 0) {
                NPU_VM_LOG_ERROR("buffer.view references unknown or non-primitive type index {}, type table size={}",
                                 elemTypeIdx, _bytecodeModule->getTypeByteSizes().size());
                _state.setExecState(ExecState::Halted);
                break;
            }
            BufferMetadata newMeta;
            newMeta.elemTypeIndex = elemTypeIdx;
            newMeta.elemByteSize = elemByteSize;
            newMeta.shape.reserve(rank);
            newMeta.strides.reserve(rank);
            constexpr auto shapeStartIndex = 5;
            for (int64_t i = 0; i < rank; ++i) {
                newMeta.shape.push_back(frame.getReg(getOperand(pc, shapeStartIndex + i)));
            }
            for (int64_t i = 0; i < rank; ++i) {
                newMeta.strides.push_back(frame.getReg(getOperand(pc, shapeStartIndex + rank + i)));
            }
            if (!intel_npu::vm::validateShapeAndStrides("buffer.view", newMeta)) {
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto byteSize = intel_npu::vm::computeLogicalByteSize(newMeta);
            if (!byteSize.has_value()) {
                NPU_VM_LOG_ERROR(
                        "buffer.view size overflows the addressable range, shape={}, strides={}, element byte size={}",
                        intel_npu::vm::formatVector(newMeta.shape), intel_npu::vm::formatVector(newMeta.strides),
                        newMeta.elemByteSize);
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto& srcBuf = bufferManager.getBuffer(srcHandle);
            if (byteOffset > srcBuf.getSize() || *byteSize > srcBuf.getSize() - byteOffset) {
                NPU_VM_LOG_ERROR("buffer.view byte offset {} + view size {} exceeds source buffer size {}", byteOffset,
                                 *byteSize, srcBuf.getSize());
                _state.setExecState(ExecState::Halted);
                break;
            }
            try {
                const auto newHandle = bufferManager.createFromBuffer(srcHandle, byteOffset, *byteSize);
                bufferMetadata.emplace(newHandle, std::move(newMeta));
                frame.setReg(dstReg, static_cast<int64_t>(newHandle));
                NPU_VM_LOG_DEBUG("  buffer.view -> handle {} ({} bytes)", newHandle, *byteSize);
            } catch (const std::exception& e) {
                NPU_VM_LOG_ERROR("buffer.view failed: {}", e.what());
                _state.setExecState(ExecState::Halted);
            }
            break;
        }
        case OpCode::BUFFER_STORE: {
            const auto bufReg = getOperand(pc, 0);
            const auto valReg = getOperand(pc, 1);
            const auto rank = static_cast<int64_t>(getOperand(pc, 2));
            if (rank < 0) {
                NPU_VM_LOG_ERROR("buffer.store requires non-negative rank, got {}", rank);
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto handle = static_cast<BufferHandle>(frame.getReg(bufReg));
            auto& bufferManager = _state.getBufferManager();
            auto& bufferMetadata = _state.getBufferMetadata();
            const auto it = bufferMetadata.find(handle);
            if (it == bufferMetadata.end() || !bufferManager.exists(handle) ||
                static_cast<int64_t>(it->second.shape.size()) != rank ||
                static_cast<int64_t>(it->second.strides.size()) != rank) {
                NPU_VM_LOG_ERROR("buffer.store invalid handle or rank mismatch: handle={}, encoded rank={}, buffer "
                                 "rank={}, stride rank={}",
                                 handle, rank, it != bufferMetadata.end() ? it->second.shape.size() : 0,
                                 it != bufferMetadata.end() ? it->second.strides.size() : 0);
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto& meta = it->second;
            uint64_t elemOffset = 0;
            bool indexFailure = false;
            for (int64_t i = 0; i < rank; ++i) {
                const auto idx = frame.getReg(getOperand(pc, 3 + i));
                const auto dimIdx = static_cast<size_t>(i);
                if (idx < 0 || idx >= meta.shape.at(dimIdx) || meta.strides.at(dimIdx) < 0) {
                    NPU_VM_LOG_ERROR(
                            "buffer.store index {} at dim {} is out of range for buffer {} (shape={}, strides={})", idx,
                            i, handle, intel_npu::vm::formatVector(meta.shape),
                            intel_npu::vm::formatVector(meta.strides));
                    indexFailure = true;
                    break;
                }
                if (!checkedAddProduct(static_cast<uint64_t>(idx), static_cast<uint64_t>(meta.strides.at(dimIdx)),
                                       elemOffset)) {
                    NPU_VM_LOG_ERROR("buffer.store linear offset overflows at dim {}: index={}, stride={}", i, idx,
                                     meta.strides.at(dimIdx));
                    indexFailure = true;
                    break;
                }
            }
            if (indexFailure) {
                _state.setExecState(ExecState::Halted);
                break;
            }
            uint64_t byteOffset = 0;
            const auto elemByteSize = static_cast<uint64_t>(meta.elemByteSize);
            if (!checkedMultiply(elemOffset, elemByteSize, byteOffset)) {
                NPU_VM_LOG_ERROR(
                        "buffer.store byte offset overflows for handle {}: element offset={}, element byte size={}",
                        handle, elemOffset, elemByteSize);
                _state.setExecState(ExecState::Halted);
                break;
            }
            try {
                auto& buf = bufferManager.getBuffer(handle);
                if (buf.getPermission() != Permission::ReadWrite) {
                    NPU_VM_LOG_ERROR("buffer.store on read-only buffer handle {}", handle);
                    _state.setExecState(ExecState::Halted);
                    break;
                }
                if ((byteOffset > buf.getSize()) || (elemByteSize > (buf.getSize() - byteOffset))) {
                    NPU_VM_LOG_ERROR("buffer.store out-of-bounds: byteOffset={}, elemByteSize={}, buffer size={}",
                                     byteOffset, elemByteSize, buf.getSize());
                    _state.setExecState(ExecState::Halted);
                    break;
                }
                const int64_t valueWord = frame.getReg(valReg);
                if (elemByteSize > sizeof(valueWord)) {
                    NPU_VM_LOG_ERROR("buffer.store element size {} exceeds register-backed value width {}",
                                     elemByteSize, sizeof(valueWord));
                    _state.setExecState(ExecState::Halted);
                    break;
                }
                std::array<uint8_t, sizeof(valueWord)> valueBytes{};
                std::memcpy(valueBytes.data(), &valueWord, sizeof(valueWord));
                buf.writeData(byteOffset, valueBytes.data(), static_cast<size_t>(elemByteSize));
                NPU_VM_LOG_DEBUG("  buffer.store handle={}, byteOffset={}, elemByteSize={}, value={}", handle,
                                 byteOffset, elemByteSize, valueWord);
            } catch (const std::exception& e) {
                NPU_VM_LOG_ERROR("buffer.store failed: {}", e.what());
                _state.setExecState(ExecState::Halted);
            }
            break;
        }
        case OpCode::BUFFER_LOAD: {
            const auto dstReg = getOperand(pc, 0);
            const auto bufReg = getOperand(pc, 1);
            const auto rank = static_cast<int64_t>(getOperand(pc, 2));
            if (rank < 0) {
                NPU_VM_LOG_ERROR("buffer.load requires non-negative rank, got {}", rank);
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto handle = static_cast<BufferHandle>(frame.getReg(bufReg));
            auto& bufferManager = _state.getBufferManager();
            auto& bufferMetadata = _state.getBufferMetadata();
            // Load shares the same shape/rank contract as buffer.store: the encoded rank must match the
            // buffer metadata so the variadic index operands are interpreted against the right tensor rank
            const auto it = bufferMetadata.find(handle);
            if (it == bufferMetadata.end() || !bufferManager.exists(handle) ||
                static_cast<int64_t>(it->second.shape.size()) != rank ||
                static_cast<int64_t>(it->second.strides.size()) != rank) {
                NPU_VM_LOG_ERROR("buffer.load invalid handle or rank mismatch: handle={}, encoded rank={}, buffer "
                                 "rank={}, stride rank={}",
                                 handle, rank, it != bufferMetadata.end() ? it->second.shape.size() : 0,
                                 it != bufferMetadata.end() ? it->second.strides.size() : 0);
                _state.setExecState(ExecState::Halted);
                break;
            }

            const auto& meta = it->second;
            uint64_t elemOffset = 0;
            bool indexFailure = false;
            for (int64_t i = 0; i < rank; ++i) {
                const auto idx = frame.getReg(getOperand(pc, 3 + i));
                const auto dimIdx = static_cast<size_t>(i);
                // Reject negative, out-of-bounds, and negatively-strided accesses before computing the flattened
                // element offset. The VM only supports forward linearization for direct buffer loads
                if (idx < 0 || idx >= meta.shape.at(dimIdx) || meta.strides.at(dimIdx) < 0) {
                    NPU_VM_LOG_ERROR(
                            "buffer.load index {} at dim {} is out of range for buffer {} (shape={}, strides={})", idx,
                            i, handle, intel_npu::vm::formatVector(meta.shape),
                            intel_npu::vm::formatVector(meta.strides));
                    indexFailure = true;
                    break;
                }
                // Accumulate the logical element offset using the precomputed stride for each dimension
                if (!checkedAddProduct(static_cast<uint64_t>(idx), static_cast<uint64_t>(meta.strides.at(dimIdx)),
                                       elemOffset)) {
                    NPU_VM_LOG_ERROR("buffer.load linear offset overflows at dim {}: index={}, stride={}", i, idx,
                                     meta.strides.at(dimIdx));
                    indexFailure = true;
                    break;
                }
            }
            if (indexFailure) {
                _state.setExecState(ExecState::Halted);
                break;
            }

            uint64_t byteOffset = 0;
            const auto elemByteSize = static_cast<uint64_t>(meta.elemByteSize);
            // Convert element offset to bytes only after the full linear index is known, so overflow handling stays
            // centralized in the checked multiply helper
            if (!checkedMultiply(elemOffset, elemByteSize, byteOffset)) {
                NPU_VM_LOG_ERROR(
                        "buffer.load byte offset overflows for handle {}: element offset={}, element byte size={}",
                        handle, elemOffset, elemByteSize);
                _state.setExecState(ExecState::Halted);
                break;
            }
            if (elemByteSize > sizeof(int64_t)) {
                NPU_VM_LOG_ERROR("buffer.load element size {} exceeds register-backed value width {}", elemByteSize,
                                 sizeof(int64_t));
                _state.setExecState(ExecState::Halted);
                break;
            }

            try {
                auto& buf = bufferManager.getBuffer(handle);
                // Write directly into the destination register storage. Zero the whole register first so narrower
                // element types still have a deterministic widened representation in the 64-bit register file
                auto& valueWord = frame.getReg(dstReg);
                valueWord = 0;
                buf.readData(byteOffset, reinterpret_cast<uint8_t*>(&valueWord), static_cast<size_t>(elemByteSize));
                NPU_VM_LOG_DEBUG("  buffer.load handle={}, byteOffset={}, elemByteSize={}, value={}", handle,
                                 byteOffset, elemByteSize, valueWord);
            } catch (const std::exception& e) {
                NPU_VM_LOG_ERROR("buffer.load failed: {}", e.what());
                _state.setExecState(ExecState::Halted);
            }
            break;
        }
        case OpCode::BUFFER_COPY: {
            const auto dstHandle = static_cast<BufferHandle>(frame.getReg(getOperand(pc, 0)));
            const auto srcHandle = static_cast<BufferHandle>(frame.getReg(getOperand(pc, 1)));
            auto& bufferManager = _state.getBufferManager();
            auto& bufferMetadata = _state.getBufferMetadata();
            const auto dstIt = bufferMetadata.find(dstHandle);
            const auto srcIt = bufferMetadata.find(srcHandle);
            if (dstIt == bufferMetadata.end() || srcIt == bufferMetadata.end() || !bufferManager.exists(dstHandle) ||
                !bufferManager.exists(srcHandle)) {
                NPU_VM_LOG_ERROR("buffer.copy uses an unknown buffer handle: dst={}, src={}", dstHandle, srcHandle);
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto& dstMeta = dstIt->second;
            const auto& srcMeta = srcIt->second;
            if (dstMeta.elemTypeIndex != srcMeta.elemTypeIndex || dstMeta.shape != srcMeta.shape ||
                dstMeta.strides != srcMeta.strides) {
                NPU_VM_LOG_ERROR("buffer.copy requires matching buffer descriptors: dst={} elemType={} shape={} "
                                 "strides={}, src={} elemType={} shape={} strides={}",
                                 dstHandle, dstMeta.elemTypeIndex, intel_npu::vm::formatVector(dstMeta.shape),
                                 intel_npu::vm::formatVector(dstMeta.strides), srcHandle, srcMeta.elemTypeIndex,
                                 intel_npu::vm::formatVector(srcMeta.shape),
                                 intel_npu::vm::formatVector(srcMeta.strides));
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto byteSize = intel_npu::vm::computeLogicalByteSize(dstMeta);
            if (!byteSize.has_value()) {
                NPU_VM_LOG_ERROR("buffer.copy descriptor size overflows");
                _state.setExecState(ExecState::Halted);
                break;
            }
            try {
                auto& dstBuffer = bufferManager.getBuffer(dstHandle);
                const auto& srcBuffer = static_cast<const BufferManager&>(bufferManager).getBuffer(srcHandle);
                if (dstBuffer.getSize() != *byteSize || srcBuffer.getSize() != *byteSize) {
                    NPU_VM_LOG_ERROR("buffer.copy descriptor size does not match source or destination buffer size");
                    _state.setExecState(ExecState::Halted);
                    break;
                }
                dstBuffer.copyDataFrom(srcBuffer);
                NPU_VM_LOG_DEBUG("  buffer.copy dst={}, src={}, size={}", dstHandle, srcHandle, *byteSize);
            } catch (const std::exception& e) {
                NPU_VM_LOG_ERROR("buffer.copy failed: {}", e.what());
                _state.setExecState(ExecState::Halted);
            }
            break;
        }
        case OpCode::CMD_LIST_CREATE: {
            ze_command_list_handle_t handle = nullptr;
            if (_state.getExecutionContext().createCmdList(handle) != ZE_RESULT_SUCCESS) {
                _state.setExecState(ExecState::Halted);
                break;
            }

            const auto dstRegNum = getOperand(pc, 0);
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            frame.setReg(dstRegNum, static_cast<int64_t>(reinterpret_cast<uint64_t>(handle)));
            break;
        }
        case OpCode::CMD_LIST_CLOSE: {
            const auto srcRegNum = getOperand(pc, 0);
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            auto cmdListHandle = reinterpret_cast<ze_command_list_handle_t>(frame.getReg(srcRegNum));
            if (_state.getExecutionContext().closeCmdList(cmdListHandle) != ZE_RESULT_SUCCESS) {
                _state.setExecState(ExecState::Halted);
            }
            break;
        }
        case OpCode::CMD_LIST_EXEC: {
            const auto srcRegNum = getOperand(pc, 0);
            const auto hostSyncFlag = getOperand(pc, 1);
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            auto cmdListHandle = reinterpret_cast<ze_command_list_handle_t>(frame.getReg(srcRegNum));
            if (submitCmdList(_state.getExecutionContext(), cmdListHandle, hostSyncFlag != 0) != ZE_RESULT_SUCCESS) {
                _state.setExecState(ExecState::Halted);
            }
            break;
        }
        case OpCode::CMD_LIST_ADD_KERNEL: {
            size_t index = 0;
            const auto cmdListHandleRegNum = getOperand(pc, index++);
            const auto kernelHandleRegNum = getOperand(pc, index++);
            const auto numSignalEvents = getOperand(pc, index++);
            if (numSignalEvents != 0) {
                NPU_VM_LOG_ERROR("cmd_list.add_kernel with non-zero signal events is not yet supported");
                _state.setExecState(ExecState::Halted);
                break;
            }
            const auto numWaitEvents = getOperand(pc, index++);
            if (numSignalEvents != 0 || numWaitEvents != 0) {
                NPU_VM_LOG_ERROR("cmd_list.add_kernel with non-zero wait events is not yet supported");
                _state.setExecState(ExecState::Halted);
                break;
            }

            // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
            auto cmdListHandle = reinterpret_cast<ze_command_list_handle_t>(frame.getReg(cmdListHandleRegNum));
            auto graphHandle = reinterpret_cast<ze_graph_handle_t>(frame.getReg(kernelHandleRegNum));
            // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
            if (executeGraph(_state.getExecutionContext(), cmdListHandle, graphHandle, nullptr) != ZE_RESULT_SUCCESS) {
                _state.setExecState(ExecState::Halted);
            }
            break;
        }
        case OpCode::KERNEL_CREATE: {
            std::vector<BufferMapperItem> inputs;
            std::vector<BufferMapperItem> outputs;
            size_t operandIndex = 0;
            const auto dstRegNum = getOperand(pc, operandIndex++);
            const auto kernelIndex = getOperand(pc, operandIndex++);
            const auto kernelNameIndex = getOperand(pc, operandIndex++);

            auto& kernelBinaries = _bytecodeModule->getKernelBinaries();
            if (static_cast<size_t>(kernelIndex) >= kernelBinaries.size()) {
                NPU_VM_LOG_ERROR("kernel_create with explicit kernel index is not supported in this stub. index: {}, "
                                 "kernelBinaries.size(): {}",
                                 kernelIndex, kernelBinaries.size());
                _state.setExecState(ExecState::Halted);
                break;
            }

            auto kernelBinary = kernelBinaries.at(kernelIndex).get();
            auto* kernelBlob = kernelBinary.begin();
            auto& kernelInfos = _state.getKernelInfos();
            auto iter = kernelInfos.find(kernelBlob);
            ze_graph_handle_t graphHandle = nullptr;
            KernelInfo* kernelInfoPtr = nullptr;
            if (iter == kernelInfos.end()) {
                KernelInfo kernelInfo;
                auto& rawStrings = _bytecodeModule->getRawStrings();
                const auto kernelName = (static_cast<size_t>(kernelNameIndex) >= rawStrings.size())
                                                ? "Unknown"
                                                : std::string(rawStrings.at(kernelNameIndex).get().begin(),
                                                              rawStrings.at(kernelNameIndex).get().end());
                if (createKernel(_state.getExecutionContext(), kernelBlob, kernelBinary.size(), kernelName, graphHandle,
                                 kernelInfo) != ZE_RESULT_SUCCESS) {
                    _state.setExecState(ExecState::Halted);
                    break;
                }
                auto insertResult = kernelInfos.emplace(kernelBlob, std::move(kernelInfo));
                kernelInfoPtr = &(insertResult.first->second);
            } else {
                graphHandle = iter->second.getGraphHandle();
                kernelInfoPtr = &(iter->second);
            }

            auto& bufferManager = _state.getBufferManager();
            auto& bufferMetadata = _state.getBufferMetadata();

            const auto inputCount = getOperand(pc, operandIndex++);
            for (int16_t i = 0; i < inputCount; ++i) {
                auto bufReg = getOperand(pc, operandIndex++);  // input register numbers (not used in this stub)
                const auto handle = static_cast<BufferHandle>(frame.getReg(bufReg));
                const auto bufferMetadataIt = bufferMetadata.find(handle);
                if (bufferMetadataIt == bufferMetadata.end() || !bufferManager.exists(handle)) {
                    NPU_VM_LOG_ERROR("buffer.get_dim on unknown buffer handle {}", handle);
                    _state.setExecState(ExecState::Halted);
                    break;
                }

                auto& buffer = bufferManager.getBuffer(handle);
                inputs.emplace_back(&buffer, &bufferMetadataIt->second);
            }

            const auto outputCount = getOperand(pc, operandIndex++);
            for (int16_t i = 0; i < outputCount; ++i) {
                auto bufReg = getOperand(pc, operandIndex++);  // input register numbers (not used in this stub)
                const auto handle = static_cast<BufferHandle>(frame.getReg(bufReg));
                const auto bufferMetadataIt = bufferMetadata.find(handle);
                if (bufferMetadataIt == bufferMetadata.end() || !bufferManager.exists(handle)) {
                    NPU_VM_LOG_ERROR("buffer.get_dim on unknown buffer handle {}", handle);
                    _state.setExecState(ExecState::Halted);
                    break;
                }
                auto& buffer = bufferManager.getBuffer(handle);
                outputs.emplace_back(&buffer, &bufferMetadataIt->second);
            }

            if (setBindings(_state.getExecutionContext(), graphHandle, inputs, outputs, *kernelInfoPtr) !=
                ZE_RESULT_SUCCESS) {
                _state.setExecState(ExecState::Halted);
                break;
            }

            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            frame.setReg(dstRegNum, static_cast<int64_t>(reinterpret_cast<uint64_t>(graphHandle)));
            break;
        }
        case OpCode::RET: {
            // Capture the return address before unwinding: pop_back invalidates `frame`.
            const auto returnAddr = frame.getReturnAddress();
            callStack.pop_back();
            if (returnAddr == EXIT_RETURN_ADDR) {
                // Returning from the entrypoint function: signal end of execution
                NPU_VM_LOG_DEBUG("  ret (return address={}) (finishing execution)",
                                 static_cast<const void*>(returnAddr));
                _state.setExecState(ExecState::Finalized);
            } else {
                // Resume the caller frame at the instruction following its CALL
                _state.setPC(returnAddr);
                NPU_VM_LOG_DEBUG("  ret (return address={})", static_cast<const void*>(returnAddr));
            }
            // Skip incrementPC: the PC is already positioned at the caller's resume address (or execution is done)
            continue;
        }
        case OpCode::CALL: {
            // CALL transfers control to a callee function using the bytecode calling convention.
            // See bytecode_format.md 6.3 for calling convention specification
            //
            // Bytecode encoding: `call rs, N, rN..., M, rM...`
            //   rs: register containing function index (index into _functions)
            //   N:  immediate value; number of destination registers (for return values)
            //   rN...: N destination register operands (caller's storage for return values)
            //   M:  immediate value; number of argument registers
            //   rM...: M source register operands (caller's argument values)

            // 1. Decode operands from CALL instruction
            const auto funcIdxReg = getOperand(pc, 0);
            const auto n = static_cast<uint16_t>(getOperand(pc, 1));

            // Collect caller's destination registers (rN...) for return-value storage.
            // Per bytecode spec, destination registers are zero-indexed; when callee
            // executes RETV, these are the targets for storing return values
            std::vector<int16_t> destRegs(n);
            for (uint16_t i = 0; i < n; ++i) {
                destRegs.at(i) = getOperand(pc, 2 + i);
            }

            const auto m = static_cast<uint16_t>(getOperand(pc, 2 + n));

            // Collect argument values from caller's frame. These will be copied into
            // the callee's parameter registers after the callee frame is created
            std::vector<int64_t> argValues(m);
            for (uint16_t i = 0; i < m; ++i) {
                argValues.at(i) = frame.getReg(getOperand(pc, 3 + n + i));
            }

            // 2. Resolve and validate callee function reference
            // funcIndex is read from the caller's register
            const auto funcIndex = static_cast<size_t>(frame.getReg(funcIdxReg));
            const auto& functions = _bytecodeModule->getFunctions();
            if (funcIndex >= functions.size()) {
                NPU_VM_LOG_ERROR("CALL refers to invalid function index {}", funcIndex);
                _state.setExecState(ExecState::Halted);
                break;
            }

            // Enforce recursion depth limit to prevent stack overflow
            if (callStack.size() > maxCallDepth) {
                NPU_VM_LOG_ERROR("Maximum call depth ({}) exceeded.", maxCallDepth);
                _state.setExecState(ExecState::Halted);
                break;
            }

            const auto& callee = functions.at(funcIndex);

            // Validate: argument count must not exceed callee's total register capacity.
            // Per bytecode spec: callee frame has G general registers + P parameter registers,
            // where P = M (argument count). So M <= num_general_registers must hold
            if (m > callee.getNumGeneralRegisters()) {
                NPU_VM_LOG_ERROR("CALL passes {} argument registers, but callee frame has only {} registers.", m,
                                 callee.getNumGeneralRegisters());
                _state.setExecState(ExecState::Halted);
                break;
            }

            // Validate: argument count must exactly match the callee parameter count.
            // CALL frame setup uses M to define the parameter register window
            if (m != static_cast<uint16_t>(callee.getParamTypes().size())) {
                NPU_VM_LOG_ERROR("CALL passes {} arguments, but callee function {} expects {} parameters.", m,
                                 callee.getName(), callee.getParamTypes().size());
                _state.setExecState(ExecState::Halted);
                break;
            }

            // Validate: destination register count must match callee's return type count.
            // This ensures caller provides exactly the right number of result storage slots
            if (n != static_cast<uint16_t>(callee.getResultTypes().size())) {
                NPU_VM_LOG_ERROR("CALL specifies {} destination registers, but callee function {} returns {} values.",
                                 n, callee.getName(), callee.getResultTypes().size());
                _state.setExecState(ExecState::Halted);
                break;
            }

            // 3. Save the address of the instruction following CALL; the callee returns here.
            // The outer decodeInstruction() has already computed the size of this CALL, so reuse it.
            const auto returnAddr = std::next(pc, static_cast<std::ptrdiff_t>(instructionSize));

            // 4. Create and initialize callee's call frame
            // Per bytecode spec 6.3, callee frame layout is:
            //   registers [0, 1, ..., G - 1]:         general (scratch) registers
            //   registers [G, G + 1, ..., G + P - 1]: parameter registers
            // where G = num_general_registers - P and P = M (argument count)
            //
            // Route callee RETV values directly into the caller's destination registers. These pointers stay valid for
            // the whole call because callStack is reserved up front and therefore never reallocates its frames. They
            // read the caller frame, so they must be collected before the callee frame is pushed
            std::vector<int64_t*> calleeResultTargets;
            calleeResultTargets.reserve(n);
            for (uint16_t i = 0; i < n; ++i) {
                calleeResultTargets.push_back(&frame.getReg(destRegs.at(i)));
            }

            CallFrame calleeFrame(callee, returnAddr, std::move(calleeResultTargets));
            const auto g = callee.getNumGeneralRegisters() - m;

            // Populate callee's parameter registers with caller's argument values.
            // Register G + i receives the i-th argument value
            for (uint16_t i = 0; i < m; ++i) {
                calleeFrame.setReg(static_cast<int16_t>(g + i), argValues.at(i));
            }

            NPU_VM_LOG_DEBUG("  call '{}' (index {}), arguments={}, destRegs={}", callee.getName(), funcIndex,
                             intel_npu::vm::formatVector(argValues), intel_npu::vm::formatVector(destRegs));

            // 5. Transfer control to the callee by pushing its frame and jumping to its body
            callStack.push_back(std::move(calleeFrame));
            _state.setPC(callee.getBody().begin());

            // Skip incrementPC: the PC is now positioned at the callee body start
            continue;
        }
        case OpCode::RETV: {
            // The first operand is an immediate value specifying the number of results to return. The same number of
            // operands follow, each specifying a register containing a result value
            //
            // RETV returns from current function, transferring values to caller's destination registers.
            // See bytecode_format.md 6.3 for calling convention specification.
            //
            // Bytecode encoding: `retv N, rN...`
            //   N:   immediate value; number of values to return (must match callee function signature)
            //   rN...: N source register operands containing return values
            //
            // When RETV executes, return values are copied from specified registers into caller-provided sinks.

            const auto numResults = getOperand(pc, 0);

            // Validate return value count matches function signature
            if (static_cast<size_t>(numResults) != results.size()) {
                NPU_VM_LOG_ERROR("RETV instruction specifies {} results, but function signature expects {}", numResults,
                                 results.size());
                _state.setExecState(ExecState::Halted);
                break;
            }

            // Collect return values from specified registers into the provided result sinks.
            // Per bytecode spec 6.3, RETV operand order corresponds to destination register
            // order in the calling CALL instruction
            constexpr auto variadicOperandOffset = 1;
            for (int64_t i = 0; i < numResults; ++i) {
                auto reg = getOperand(pc, variadicOperandOffset + i);
                if (results.at(i) == nullptr) {
                    NPU_VM_LOG_ERROR("RETV sink pointer is null for result {}", i);
                    _state.setExecState(ExecState::Halted);
                    break;
                }
                *results.at(i) = frame.getReg(reg);
            }
            if (_state.getExecState() == ExecState::Halted) {
                break;
            }

            // Determine return target: entrypoint finalization or nested call return. All reads of `frame`/`results`
            // must happen before pop_back, which invalidates the current frame references
            const auto returnAddr = frame.getReturnAddress();
            if (returnAddr == EXIT_RETURN_ADDR) {
                // Returning from entrypoint function: signal end of execution
                NPU_VM_LOG_DEBUG("  retv {} (values={}, return address={}) (finishing execution)", numResults,
                                 formatPtrVectorContent(results), static_cast<const void*>(returnAddr));
                _state.setExecState(ExecState::Finalized);
            } else {
                // Nested call: resume the caller frame. Result sinks have already been written above
                NPU_VM_LOG_DEBUG("  retv {} (values={}, return address={})", numResults,
                                 formatPtrVectorContent(results), static_cast<const void*>(returnAddr));
                _state.setPC(returnAddr);
            }
            callStack.pop_back();
            // Skip incrementPC: the PC is already positioned at the caller's resume address (or execution is done)
            continue;
        }
        case OpCode::JMP: {
            const auto offset = get64BitImm(pc, 0);
            if (!applyJump(_state, offset, functionBodyStart, endOfFunction, function)) {
                const auto pcOffset = static_cast<int64_t>(_state.getPC() - functionBodyStart);
                NPU_VM_LOG_ERROR("jmp invalid target: pc+{} offset={} target={}", pcOffset, offset,
                                 (pcOffset + offset));
                _state.setExecState(ExecState::Halted);
                break;
            }
            NPU_VM_LOG_DEBUG("  jmp {} (new pc={})", offset, static_cast<const void*>(_state.getPC()));
            continue;
        }
        case OpCode::JE: {
            const auto lhsReg = getAsOperand<int16_t>(pc, OPCODE_SIZE + sizeof(int64_t));
            const auto rhsReg = getAsOperand<int16_t>(pc, OPCODE_SIZE + sizeof(int64_t) + sizeof(int16_t));
            const auto lhs = frame.getReg(lhsReg);
            const auto rhs = frame.getReg(rhsReg);
            if (lhs == rhs) {
                const auto offset = get64BitImm(pc, 0);
                if (!applyJump(_state, offset, functionBodyStart, endOfFunction, function)) {
                    const auto pcOffset = static_cast<int64_t>(_state.getPC() - functionBodyStart);
                    NPU_VM_LOG_ERROR("je invalid target: pc+{} offset={} target={} (lhs={} rhs={})", pcOffset, offset,
                                     (pcOffset + offset), lhs, rhs);
                    _state.setExecState(ExecState::Halted);
                    break;
                }
                NPU_VM_LOG_DEBUG("  je {}, {}, {} ({} == {} -> taken)", offset, lhsReg, rhsReg, lhs, rhs);
                continue;
            }
            NPU_VM_LOG_DEBUG("  je {}, {}, {} ({} == {} -> not taken)", get64BitImm(_state.getPC(), 0), lhsReg, rhsReg,
                             lhs, rhs);
            break;
        }
        case OpCode::JNE: {
            const auto lhsReg = getAsOperand<int16_t>(pc, OPCODE_SIZE + sizeof(int64_t));
            const auto rhsReg = getAsOperand<int16_t>(pc, OPCODE_SIZE + sizeof(int64_t) + sizeof(int16_t));
            const auto lhs = frame.getReg(lhsReg);
            const auto rhs = frame.getReg(rhsReg);
            if (lhs != rhs) {
                const auto offset = get64BitImm(pc, 0);
                if (!applyJump(_state, offset, functionBodyStart, endOfFunction, function)) {
                    const auto pcOffset = static_cast<int64_t>(_state.getPC() - functionBodyStart);
                    NPU_VM_LOG_ERROR("jne invalid target: pc+{} offset={} target={} (lhs={} rhs={})", pcOffset, offset,
                                     (pcOffset + offset), lhs, rhs);
                    _state.setExecState(ExecState::Halted);
                    break;
                }
                NPU_VM_LOG_DEBUG("  jne {}, {}, {} ({} != {} -> taken)", offset, lhsReg, rhsReg, lhs, rhs);
                continue;
            }
            NPU_VM_LOG_DEBUG("  jne {}, {}, {} ({} != {} -> not taken)", get64BitImm(_state.getPC(), 0), lhsReg, rhsReg,
                             lhs, rhs);
            break;
        }
        case OpCode::CONVERT_I8_TO_F32: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto floatResult = static_cast<float>(signExtend8(frame.getReg(srcRegNum)));
            int64_t regVal = 0;
            std::memcpy(&regVal, &floatResult, sizeof(float));
            frame.setReg(dstRegNum, regVal);
            NPU_VM_LOG_DEBUG("  convert.i8tof32 {}, {} = {}", dstRegNum, srcRegNum, floatResult);
            break;
        }
        case OpCode::CONVERT_I8_TO_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto result = static_cast<double>(signExtend8(frame.getReg(srcRegNum)));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(result));
            NPU_VM_LOG_DEBUG("  convert.i8tof64 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_I16_TO_I8: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto result = signExtend8(signExtend16(frame.getReg(srcRegNum)));
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.i16toi8 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_I16_TO_F32: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto floatResult = static_cast<float>(signExtend16(frame.getReg(srcRegNum)));
            int64_t regVal = 0;
            std::memcpy(&regVal, &floatResult, sizeof(float));
            frame.setReg(dstRegNum, regVal);
            NPU_VM_LOG_DEBUG("  convert.i16tof32 {}, {} = {}", dstRegNum, srcRegNum, floatResult);
            break;
        }
        case OpCode::CONVERT_I16_TO_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto result = static_cast<double>(signExtend16(frame.getReg(srcRegNum)));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(result));
            NPU_VM_LOG_DEBUG("  convert.i16tof64 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_I32_TO_I8: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto result = signExtend8(signExtend32(frame.getReg(srcRegNum)));
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.i32toi8 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_I32_TO_I16: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto result = signExtend16(signExtend32(frame.getReg(srcRegNum)));
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.i32toi16 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_I32_TO_F32: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto floatResult = static_cast<float>(signExtend32(frame.getReg(srcRegNum)));
            int64_t regVal = 0;
            std::memcpy(&regVal, &floatResult, sizeof(float));
            frame.setReg(dstRegNum, regVal);
            NPU_VM_LOG_DEBUG("  convert.i32tof32 {}, {} = {}", dstRegNum, srcRegNum, floatResult);
            break;
        }
        case OpCode::CONVERT_I32_TO_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto result = static_cast<double>(signExtend32(frame.getReg(srcRegNum)));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(result));
            NPU_VM_LOG_DEBUG("  convert.i32tof64 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_I64_TO_I8: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto result = signExtend8(frame.getReg(srcRegNum));
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.i64toi8 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_I64_TO_I16: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto result = signExtend16(frame.getReg(srcRegNum));
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.i64toi16 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_I64_TO_I32: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto result = signExtend32(frame.getReg(srcRegNum));
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.i64toi32 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_I64_TO_F32: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto floatResult = static_cast<float>(frame.getReg(srcRegNum));
            int64_t regVal = 0;
            std::memcpy(&regVal, &floatResult, sizeof(float));
            frame.setReg(dstRegNum, regVal);
            NPU_VM_LOG_DEBUG("  convert.i64tof32 {}, {} = {}", dstRegNum, srcRegNum, floatResult);
            break;
        }
        case OpCode::CONVERT_I64_TO_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto result = static_cast<double>(frame.getReg(srcRegNum));
            frame.setReg(dstRegNum, vmBitCast<int64_t>(result));
            NPU_VM_LOG_DEBUG("  convert.i64tof64 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_F32_TO_I8: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            float src = 0.0f;
            std::memcpy(&src, &frame.getReg(srcRegNum), sizeof(float));
            const auto result = saturatingFloatToInt<int8_t>(src);
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.f32toi8 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_F32_TO_I16: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            float src = 0.0f;
            std::memcpy(&src, &frame.getReg(srcRegNum), sizeof(float));
            const auto result = saturatingFloatToInt<int16_t>(src);
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.f32toi16 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_F32_TO_I32: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            float src = 0.0f;
            std::memcpy(&src, &frame.getReg(srcRegNum), sizeof(float));
            const auto result = saturatingFloatToInt<int32_t>(src);
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.f32toi32 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_F32_TO_I64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            float src = 0.0f;
            std::memcpy(&src, &frame.getReg(srcRegNum), sizeof(float));
            const auto result = saturatingFloatToInt<int64_t>(src);
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.f32toi64 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_F32_TO_F64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            float src = 0.0f;
            std::memcpy(&src, &frame.getReg(srcRegNum), sizeof(float));
            const auto result = static_cast<double>(src);
            frame.setReg(dstRegNum, vmBitCast<int64_t>(result));
            NPU_VM_LOG_DEBUG("  convert.f32tof64 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_F64_TO_I8: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto src = vmBitCast<double>(frame.getReg(srcRegNum));
            const auto result = saturatingFloatToInt<int8_t>(src);
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.f64toi8 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_F64_TO_I16: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto src = vmBitCast<double>(frame.getReg(srcRegNum));
            const auto result = saturatingFloatToInt<int16_t>(src);
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.f64toi16 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_F64_TO_I32: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto src = vmBitCast<double>(frame.getReg(srcRegNum));
            const auto result = saturatingFloatToInt<int32_t>(src);
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.f64toi32 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_F64_TO_I64: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto src = vmBitCast<double>(frame.getReg(srcRegNum));
            const auto result = saturatingFloatToInt<int64_t>(src);
            frame.setReg(dstRegNum, result);
            NPU_VM_LOG_DEBUG("  convert.f64toi64 {}, {} = {}", dstRegNum, srcRegNum, result);
            break;
        }
        case OpCode::CONVERT_F64_TO_F32: {
            const auto dstRegNum = getOperand(pc, 0);
            const auto srcRegNum = getOperand(pc, 1);
            const auto src = vmBitCast<double>(frame.getReg(srcRegNum));
            const auto floatResult = static_cast<float>(src);
            int64_t regVal = 0;
            std::memcpy(&regVal, &floatResult, sizeof(float));
            frame.setReg(dstRegNum, regVal);
            NPU_VM_LOG_DEBUG("  convert.f64tof32 {}, {} = {}", dstRegNum, srcRegNum, floatResult);
            break;
        }
        default:
            NPU_VM_LOG_ERROR("Unknown opcode {}", static_cast<uint16_t>(opcode));
            _state.setExecState(ExecState::Halted);
            break;
        }
        incrementPC(instructionSize);
    }
}

}  // namespace

struct _npu_vm_engine {
    std::unique_ptr<VMEngine> impl;
};

struct _npu_vm_module {
    std::shared_ptr<BytecodeModule> impl;
};

std::optional<intel_npu::vm::NetworkMetadata> intel_npu::vm::getNetworkMetadata(npu_vm_module* bytecodeModule) {
    if (bytecodeModule == nullptr || bytecodeModule->impl == nullptr) {
        return std::nullopt;
    }
    return bytecodeModule->impl->getNetworkMetadata();
}

namespace {

std::optional<npu_vm_type> convertToVmType(const FuncParamResType& value) {
    switch (getTypeCode(value.type)) {
    case TypeCode::INTEGER: {
        const auto isSigned = isTypeSigned(value.type);
        auto bitWidth = getBitWidth(value.type);
        if (bitWidth == sizeof(int8_t) * CHAR_BIT) {
            return isSigned ? npu_vm_type_int8 : npu_vm_type_uint8;
        } else if (bitWidth == sizeof(int16_t) * CHAR_BIT) {
            return isSigned ? npu_vm_type_int16 : npu_vm_type_uint16;
        } else if (bitWidth == sizeof(int32_t) * CHAR_BIT) {
            return isSigned ? npu_vm_type_int32 : npu_vm_type_uint32;
        } else if (bitWidth == sizeof(int64_t) * CHAR_BIT) {
            return isSigned ? npu_vm_type_int64 : npu_vm_type_uint64;
        } else {
            return std::nullopt;
        }
        break;
    }
    case TypeCode::FLOAT: {
        auto bitWidth = getBitWidth(value.type);
        if (bitWidth == sizeof(float) * CHAR_BIT) {
            return npu_vm_type_float32;
        } else if (bitWidth == sizeof(double) * CHAR_BIT) {
            return npu_vm_type_float64;
        } else {
            return std::nullopt;
        }
        break;
    }
    case TypeCode::BUFFER: {
        return npu_vm_type_buffer;
        break;
    }
    default:
        return std::nullopt;
    }
}

template <typename T>
bool isValid(T* ptr) {
    if (ptr == nullptr) {
        return false;
    }
    if (ptr->impl == nullptr) {
        return false;
    }
    return true;
}

npu_vm_result vmCallImpl(npu_vm_engine* engine, const char* name, uint32_t numArguments, const npu_vm_value* arguments,
                         uint32_t numResults, npu_vm_value* results, const char* callerFunction) {
    if (!isValid(engine)) {
        NPU_VM_LOG_ERROR("Null pointer to engine instance");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    if (name == nullptr) {
        NPU_VM_LOG_ERROR("Null pointer passed for function name");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    try {
        const auto bytecodeModule = engine->impl->getModule();
        if (bytecodeModule == nullptr) {
            NPU_VM_LOG_ERROR("Bytecode module is not loaded");
            return NPU_VM_ERROR_MODULE_NOT_FOUND;
        }
        const auto function = bytecodeModule->getFunction(name);
        if (function == nullptr) {
            NPU_VM_LOG_ERROR("Function not found: {}", name);
            return NPU_VM_ERROR_FUNCTION_NOT_FOUND;
        }
        if (numArguments > 0 && arguments == nullptr) {
            NPU_VM_LOG_ERROR("Argument array is null while argument count is {}", numArguments);
            return NPU_VM_ERROR_INVALID_NULL_POINTER;
        }
        if (numResults > 0 && results == nullptr) {
            NPU_VM_LOG_ERROR("Result array is null while result count is {}", numResults);
            return NPU_VM_ERROR_INVALID_NULL_POINTER;
        }
        Span<const npu_vm_value> argSpan(arguments, numArguments);
        Span<npu_vm_value> resultSpan(results, numResults);
        if (!engine->impl->call(name, argSpan, resultSpan)) {
            NPU_VM_LOG_ERROR("Error during function call: {}", name);
            return NPU_VM_ERROR_FUNCTION_CALL_FAILED;
        }
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Error during {}: {}", callerFunction, e.what());
        return NPU_VM_ERROR_UNKNOWN;
    }
    return NPU_VM_SUCCESS;
}

}  // namespace

// NOLINTBEGIN(readability-identifier-naming) - C API identifiers must follow C naming conventions

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_print(const uint8_t* bytecode, uint32_t bytecode_size, int print_full,
                                                        uint32_t indent_level) {
    if (bytecode == nullptr || bytecode_size == 0) {
        NPU_VM_LOG_ERROR("Null pointer or empty bytecode passed");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    try {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast) - function receives a pointer to const data
        const auto bytecodeView = Span<uint8_t>(const_cast<uint8_t*>(bytecode), bytecode_size);
        BytecodeReader reader(bytecodeView);
        if (!reader.parseFile()) {
            NPU_VM_LOG_ERROR("Failed to parse bytecode file.");
            return NPU_VM_ERROR_UNKNOWN;
        }
        reader.printFile(static_cast<bool>(print_full), indent_level);
        return NPU_VM_SUCCESS;
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Error during npu_vm_print: {}", e.what());
        return NPU_VM_ERROR_UNKNOWN;
    }
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_parse_module(const uint8_t* bytecode, uint32_t bytecode_size,
                                                               npu_vm_module** module_out) {
    if (module_out == nullptr) {
        NPU_VM_LOG_ERROR("Null pointer passed for module_out");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    if (bytecode == nullptr || bytecode_size == 0) {
        NPU_VM_LOG_ERROR("Null pointer or empty bytecode passed");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    try {
        // The provided bytecode is expected to live beyond the lifetime of the module, so the created module does not
        // take ownership
        const auto copyBytecode = false;
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast) - function receives a pointer to const data
        const auto bytecodeView = Span<uint8_t>(const_cast<uint8_t*>(bytecode), bytecode_size);
        auto moduleImpl = BytecodeModule::createFrom(bytecodeView, copyBytecode);
        if (moduleImpl == nullptr) {
            NPU_VM_LOG_ERROR("Bytecode parsing failed");
            return NPU_VM_ERROR_UNKNOWN;
        }
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory) - using raw pointer for C API struct
        *module_out = new (std::nothrow) npu_vm_module;
        if (*module_out == nullptr) {
            NPU_VM_LOG_ERROR("Memory allocation failed for bytecode module");
            return NPU_VM_ERROR_MEMORY_ALLOCATION_FAILED;
        }
        (*module_out)->impl = std::move(moduleImpl);
        return NPU_VM_SUCCESS;
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Error during npu_vm_parse_module: {}", e.what());
        return NPU_VM_ERROR_UNKNOWN;
    }
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_destroy_module(npu_vm_module* module) {
    if (module == nullptr) {
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    module->impl.reset();
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory) - using raw pointer for C API struct
    delete module;
    return NPU_VM_SUCCESS;
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_get_function_info(npu_vm_module* module, const char* name,
                                                                    npu_vm_function_info** info_out) {
    if (!isValid(module)) {
        NPU_VM_LOG_ERROR("Null pointer to module instance");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    if (name == nullptr) {
        NPU_VM_LOG_ERROR("Null pointer passed for function name");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    if (info_out == nullptr) {
        NPU_VM_LOG_ERROR("Null pointer passed for info_out");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }

    const auto function = module->impl->getFunction(name);
    if (function == nullptr) {
        NPU_VM_LOG_ERROR("Function not found: {}", name);
        return NPU_VM_ERROR_FUNCTION_NOT_FOUND;
    }

    // NOLINTBEGIN(cppcoreguidelines-owning-memory) - using C API struct with raw pointers
    auto info = new (std::nothrow) npu_vm_function_info;
    if (info == nullptr) {
        NPU_VM_LOG_ERROR("Memory allocation failed for npu_vm_function_info");
        return NPU_VM_ERROR_MEMORY_ALLOCATION_FAILED;
    }
    const auto& funcName = function->getName();
    info->name = new (std::nothrow) char[funcName.size() + 1];
    if (info->name == nullptr) {
        NPU_VM_LOG_ERROR("Memory allocation failed for function name");
        delete info;
        return NPU_VM_ERROR_MEMORY_ALLOCATION_FAILED;
    }
    std::memcpy(info->name, funcName.c_str(), funcName.size() + 1);

    info->num_params = static_cast<uint32_t>(function->getParamTypes().size());
    info->num_results = static_cast<uint32_t>(function->getResultTypes().size());
    // Allocate arrays for param_types and result_types (caller must free)
    if (info->num_params > 0) {
        info->param_types = new (std::nothrow) npu_vm_type[info->num_params];
        if (info->param_types == nullptr) {
            NPU_VM_LOG_ERROR("Memory allocation failed for param_types");
            delete[] info->name;
            delete info;
            return NPU_VM_ERROR_MEMORY_ALLOCATION_FAILED;
        }
        Span<npu_vm_type> paramTypesSpan(info->param_types, info->num_params);
        for (uint32_t i = 0; i < info->num_params; ++i) {
            const auto type = convertToVmType(function->getParamTypes().at(i));
            if (!type.has_value()) {
                NPU_VM_LOG_ERROR("Unsupported parameter type for parameter {} in function {}", i, name);
                delete[] info->param_types;
                delete[] info->name;
                delete info;
                return NPU_VM_ERROR_UNKNOWN;
            }
            paramTypesSpan.at(i) = type.value();
        }
    } else {
        info->param_types = nullptr;
    }
    if (info->num_results > 0) {
        info->result_types = new (std::nothrow) npu_vm_type[info->num_results];
        if (info->result_types == nullptr) {
            NPU_VM_LOG_ERROR("Memory allocation failed for result_types");
            delete[] info->param_types;
            delete[] info->name;
            delete info;
            return NPU_VM_ERROR_MEMORY_ALLOCATION_FAILED;
        }
        Span<npu_vm_type> resultTypesSpan(info->result_types, info->num_results);
        for (uint32_t i = 0; i < info->num_results; ++i) {
            const auto type = convertToVmType(function->getResultTypes().at(i));
            if (!type.has_value()) {
                NPU_VM_LOG_ERROR("Unsupported result type for result {} in function {}", i, name);
                delete[] info->result_types;
                delete[] info->param_types;
                delete[] info->name;
                delete info;
                return NPU_VM_ERROR_UNKNOWN;
            }
            resultTypesSpan.at(i) = type.value();
        }
    } else {
        info->result_types = nullptr;
    }
    // NOLINTEND(cppcoreguidelines-owning-memory)

    *info_out = info;

    return NPU_VM_SUCCESS;
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_destroy_function_info(npu_vm_function_info* info) {
    if (info == nullptr) {
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    // NOLINTBEGIN(cppcoreguidelines-owning-memory) - using C API struct with raw pointers
    delete[] info->name;
    delete[] info->param_types;
    delete[] info->result_types;
    delete info;
    // NOLINTEND(cppcoreguidelines-owning-memory)
    return NPU_VM_SUCCESS;
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_new_engine(npu_vm_engine** engine_out) {
    if (engine_out == nullptr) {
        NPU_VM_LOG_ERROR("Null pointer passed for engine_out");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    try {
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory) - using raw pointer for C API struct
        *engine_out = new (std::nothrow) npu_vm_engine;
        if (*engine_out == nullptr) {
            NPU_VM_LOG_ERROR("Memory allocation failed for npu_vm_engine");
            return NPU_VM_ERROR_MEMORY_ALLOCATION_FAILED;
        }
        (*engine_out)->impl = std::make_unique<VMEngine>();
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Error during npu_vm_new_engine: {}", e.what());
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory) - cannot use smart pointer for the C API
        delete *engine_out;
        *engine_out = nullptr;
        return NPU_VM_ERROR_UNKNOWN;
    }
    return NPU_VM_SUCCESS;
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_destroy_engine(npu_vm_engine* engine) {
    if (engine == nullptr) {
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    engine->impl.reset();
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory) - using raw pointer for C API struct
    delete engine;
    return NPU_VM_SUCCESS;
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_load_module(npu_vm_engine* engine, npu_vm_module_load_params* params,
                                                              npu_vm_module* module) {
    if (!isValid(engine)) {
        NPU_VM_LOG_ERROR("Null pointer to engine instance");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    if (!isValid(module)) {
        NPU_VM_LOG_ERROR("Null pointer to module instance");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    try {
        if (!engine->impl->loadModule(module->impl, params)) {
            NPU_VM_LOG_ERROR("Failed to load module into engine");
            return NPU_VM_ERROR_UNKNOWN;
        }
        return NPU_VM_SUCCESS;
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Error during npu_vm_load_module: {}", e.what());
        return NPU_VM_ERROR_UNKNOWN;
    }
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_reset_state(npu_vm_engine* engine, int resetExecutionContext) {
    if (!isValid(engine)) {
        NPU_VM_LOG_ERROR("Null pointer to engine instance");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    try {
        if (!engine->impl->resetState(static_cast<bool>(resetExecutionContext))) {
            NPU_VM_LOG_ERROR("Failed to reset engine state");
            return NPU_VM_ERROR_UNKNOWN;
        }
        return NPU_VM_SUCCESS;
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Error during npu_vm_reset_state: {}", e.what());
        return NPU_VM_ERROR_UNKNOWN;
    }
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_call(npu_vm_engine* engine, const char* name, uint32_t num_arguments,
                                                       const npu_vm_value* arguments) {
    return vmCallImpl(engine, name, num_arguments, arguments, /*numResults=*/0, /*results=*/nullptr, "npu_vm_call");
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_call_with_results(npu_vm_engine* engine, const char* name,
                                                                    uint32_t num_arguments,
                                                                    const npu_vm_value* arguments, uint32_t num_results,
                                                                    npu_vm_value* results) {
    return vmCallImpl(engine, name, num_arguments, arguments, num_results, results, "npu_vm_call_with_results");
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_infer(npu_vm_engine* engine,
                                                        npu_vm_runtime_execute_params_t* params) {
    if (!isValid(engine)) {
        NPU_VM_LOG_ERROR("Null pointer to engine instance");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    if (params == nullptr) {
        NPU_VM_LOG_ERROR("Null pointer passed for params");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    try {
        if (!engine->impl->infer(params)) {
            NPU_VM_LOG_ERROR("Inference failed for the given parameters");
            return NPU_VM_ERROR_INFERENCE_FAILED;
        }
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Error during npu_vm_infer: {}", e.what());
        return NPU_VM_ERROR_UNKNOWN;
    }
    return NPU_VM_SUCCESS;
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL
npu_vm_predict_output_shape(npu_vm_engine* engine, npu_vm_runtime_predict_output_shape_params_t2* params) {
    if (!isValid(engine)) {
        NPU_VM_LOG_ERROR("Null pointer to engine instance");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    if (params == nullptr) {
        NPU_VM_LOG_ERROR("Null pointer passed for params");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    try {
        if (!engine->impl->predictOutputShape(params)) {
            NPU_VM_LOG_ERROR("Output shape prediction failed for the given parameters");
            return NPU_VM_ERROR_OUTPUT_SHAPE_PREDICTION_FAILED;
        }
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Error during npu_vm_predict_output_shape: {}", e.what());
        return NPU_VM_ERROR_UNKNOWN;
    }
    return NPU_VM_SUCCESS;
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_infer2(npu_vm_engine* engine,
                                                         npu_vm_runtime_execute_params2_t* params) {
    if (!isValid(engine)) {
        NPU_VM_LOG_ERROR("Null pointer to engine instance");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    if (params == nullptr) {
        NPU_VM_LOG_ERROR("Null pointer passed for params");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    try {
        if (!engine->impl->infer(params)) {
            NPU_VM_LOG_ERROR("Inference failed for the given parameters");
            return NPU_VM_ERROR_INFERENCE_FAILED;
        }
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Error during npu_vm_infer: {}", e.what());
        return NPU_VM_ERROR_UNKNOWN;
    }
    return NPU_VM_SUCCESS;
}

NPU_VM_EXPORT bool NPU_VM_APICALL npu_vm_host_code_execution_required(npu_vm_engine* engine,
                                                                      npu_vm_runtime_execute_params2_t* params) {
    if (!isValid(engine)) {
        NPU_VM_LOG_ERROR("Null pointer to engine instance");
        return true;
    }

    if (params == nullptr) {
        NPU_VM_LOG_ERROR("Null pointer passed for params");
        return true;
    }

    try {
        return engine->impl->hostCodeExecutionRequired(params);
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Error during npu_vm_host_code_execution_required: {}", e.what());
        return true;
    }
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL
npu_vm_submit_recorded_command_lists(npu_vm_engine* engine, npu_vm_runtime_execute_params2_t* params) {
    if (!isValid(engine)) {
        NPU_VM_LOG_ERROR("Null pointer to engine instance");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }
    if (params == nullptr) {
        NPU_VM_LOG_ERROR("Null pointer passed for params");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }

    try {
        if (!engine->impl->submitRecordedCmdLists(params)) {
            return NPU_VM_ERROR_UNKNOWN;
        }
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Error during npu_vm_host_code_execution_required: {}", e.what());
        return NPU_VM_ERROR_UNKNOWN;
    }
    return NPU_VM_SUCCESS;
}

NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_host_sync(npu_vm_engine* engine) {
    if (!isValid(engine)) {
        NPU_VM_LOG_ERROR("Null pointer to engine instance");
        return NPU_VM_ERROR_INVALID_NULL_POINTER;
    }

    try {
        if (!engine->impl->hostSync()) {
            return NPU_VM_ERROR_UNKNOWN;
        }
    } catch (const std::exception& e) {
        NPU_VM_LOG_ERROR("Error during npu_vm_host_code_execution_required: {}", e.what());
        return NPU_VM_ERROR_UNKNOWN;
    }
    return NPU_VM_SUCCESS;
}

// NOLINTEND(readability-identifier-naming)
