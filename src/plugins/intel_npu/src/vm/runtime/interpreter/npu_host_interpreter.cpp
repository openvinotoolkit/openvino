//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/bytecode/bytecode.hpp"
#include "intel_npu/runtime/npu_mlir_runtime.hpp"

#include <variant>
#include "intel_npu/network_metadata.hpp"
// #include "intel_npu/utils/zero/zero_utils.hpp"
#include "level_zero_wrapper.h"
#include "intel_npu/utils/logger/logger.hpp"

#include <array>

#include <chrono>
#include <thread>

using namespace intel_npu;

class DebugTrace {
public:
    DebugTrace(std::string funcName, Logger logger = Logger::global())
            : _funcName(funcName), _logger(logger) {
        // _logger.info("%s starts", _funcName);
    }
    ~DebugTrace() {
        // _logger.info("%s ends", _funcName);
    }

private:
    std::string _funcName;
    Logger _logger;
};

struct MemRefNDRef {
    constexpr static size_t headerSize = 3;  // allocatedPtr, alignedPtr, offset

    int64_t* bufferPtr;
    int64_t dimCount;
    MemRefNDRef(int64_t* buffer, int64_t dim_count): bufferPtr(buffer), dimCount(dim_count) {
    }

    MemRefNDRef(const MemRefNDRef&) = delete;
    MemRefNDRef& operator=(const MemRefNDRef&) = delete;

    void setAllocated(const void* buf) {
        bufferPtr[0] = reinterpret_cast<int64_t>(buf);
    }

    void setAligned(const void* buf) {
        bufferPtr[1] = reinterpret_cast<int64_t>(buf);
    }

    void setOffset(int64_t offset) {
        bufferPtr[2] = static_cast<int64_t>(offset);
    }

    template <typename T>
    void setSizes(T* size, int64_t dimCount) {
        int64_t* ptr = bufferPtr + headerSize;
        for (int64_t i = 0; i < dimCount; ++i) {
            ptr[i] = static_cast<int64_t>(size[i]);
        }
    }

    template <typename T>
    void setStrides(T* strides, int64_t dimCount) {
        int64_t* ptr = bufferPtr + headerSize + dimCount;
        for (int64_t i = 0; i < dimCount; ++i) {
            ptr[i] = static_cast<int64_t>(strides[i]);
        }
    }

    void* getAllocated() {
        return reinterpret_cast<void*>(bufferPtr[0]);
    }

    void* getAligned() {
        return reinterpret_cast<void*>(bufferPtr[1]);
    }

    int64_t getOffset() {
        return bufferPtr[2];
    }

    int64_t* getSizes() {
        return reinterpret_cast<int64_t*>(bufferPtr + headerSize);
    }

    int64_t* getStrides() {
        return reinterpret_cast<int64_t*>(bufferPtr + headerSize + dimCount);
    }
};

struct MemRefHandle {
    int64_t* memRefBufferPtr;
    int64_t dimCount;

    MemRefHandle(int64_t dim_count): memRefBufferPtr(nullptr), dimCount(dim_count) {
        int64_t numElements = MemRefNDRef::headerSize + dimCount * 2;
        memRefBufferPtr = new int64_t[numElements];
        for (int64_t i = 0; i < numElements; ++i) {
            memRefBufferPtr[i] = 0;
        }
    }
    ~MemRefHandle() {
        if (memRefBufferPtr != nullptr) {
            delete[] memRefBufferPtr;
            memRefBufferPtr = nullptr;
        }
    }

    int64_t getMemRefBufferNumElements() {
        return MemRefNDRef::headerSize + dimCount * 2;
    }

    int64_t getMemRefBufferByteSize() {
        return getMemRefBufferNumElements() * sizeof(uint64_t);
    }

    void parseMemRef(const void** pBasePtr, const void** pData, int64_t* pOffset, int64_t* pSizes, int64_t* pStrides,
                     int64_t* pDimsCount) {
        MemRefNDRef ref(memRefBufferPtr, dimCount);
        *pBasePtr = ref.getAllocated();
        *pData = ref.getAligned();
        *pOffset = static_cast<int64_t>(ref.getOffset());
        int64_t* sizes = ref.getSizes();
        int64_t* strides = ref.getStrides();
        for (uint64_t i = 0; i < dimCount; ++i) {
            pSizes[i] = sizes[i];
            pStrides[i] = strides[i];
        }
        *pDimsCount = static_cast<uint32_t>(dimCount);
    }

    std::string toString() {
        std::string result = "MemRefHandle(dimCount=" + std::to_string(dimCount) + ", buffer=[";
        for (int64_t i = 0; i < getMemRefBufferNumElements(); ++i) {
            result += std::to_string(memRefBufferPtr[i]);
            if (i < getMemRefBufferNumElements() - 1) {
                result += ", ";
            }
        }
        result += "])";
        return result;
    }
};

struct Frame {
    std::vector<int64_t> i64;
    std::vector<vpux::MemRefDesc> m;

    uint32_t funcId = 0;
    uint32_t pc = 0;
};

struct CallFrame {
    uint32_t funcId = 0;
    uint32_t returnPc = 0;
};

struct VirtualMachine {
    Logger _logger = Logger("VirtualMachine", Logger::global().level());
    //
    // Compute subview(base, offsets, sizes) with step=1.
    // Data pointer offset = sum(offset[d]*base.strides[d]) * elemBytes
    void applySubview(const vpux::MemRefDesc& base, const vpux::SubviewTemplate& t, const std::vector<int64_t>& i64regs,
                      vpux::MemRefDesc& result) {
        if (!base.data) {
            OPENVINO_THROW("subview of null base");
        }
        // if (t.rank != base.rank) throw std::runtime_error("Subview rank mismatch");
        // if (base.elemBytes == 0) throw std::runtime_error("base.elemBytes==0");

        int64_t offElems[4] = {0, 0, 0, 0};
        for (uint32_t d = 0; d < t.rank; ++d) {
            int64_t o = 0;
            if (t.offsets[d].kind == vpux::OffsetKind::Imm) {
                o = t.offsets[d].imm;
            } else {
                if (t.offsets[d].reg >= i64regs.size()) {
                    OPENVINO_THROW("Subview offset reg OOB");
                }
                o = i64regs[t.offsets[d].reg];
            }
            // Basic bounds check (optional for prototype)
            if (o < 0 || o > base.sizes[d]) {
                OPENVINO_THROW("Subview offset out of bounds");
            }
            offElems[d] = o;
        }

        // Compute pointer advance
        int64_t linearElemOffset = 0;
        for (uint32_t d = 0; d < t.rank; ++d) {
            if (base.strides[d] == 0) {
                OPENVINO_THROW("base.strides[d]==0");
            }
            linearElemOffset += offElems[d] * base.strides[d];
        }

        result.dimCount = base.dimCount;
        result.data = static_cast<void*>(static_cast<uint8_t*>(base.data) +
                                         linearElemOffset * static_cast<int64_t>(base.elementByteSize));
        result.elementByteSize = base.elementByteSize;

        for (uint32_t d = 0; d < t.rank; ++d) {
            result.sizes[d] = t.sizes[d];
            result.strides[d] = base.strides[d];
        }
    }

    void runKernel(vpux::Kernel& kernel, std::vector<vpux::MemRefDesc>& args,
                   npu_mlir_runtime_execute_params_t* pParams) {
        DebugTrace dbg("VirtualMachine::runKernel", _logger);
        vpux::MemRefDesc* input = &args[0];
        vpux::MemRefDesc* output = &args[1];
        int32_t numInputs = 1, numOutputs = 1;

        _logger.debug("Adding inference job");
        auto result = npu_level_zero_execute_graph((void**)input, numInputs, (void**)output, numOutputs,
                                                   kernel.symbolName.data(), kernel.binaryData.data(),
                                                   kernel.binaryData.size(), pParams->ctx, pParams->device,
                                                   pParams->graphDdiTableExt, (void**)pParams->commandLists);
        if (result != 0) {
            char* pError = nullptr;
            npu_level_zero_get_last_error(&pError);
            std::string errorMsg = (pError != nullptr) ? pError : "Unknown error";
            // _logger.error("Cannot add an inference job to the command list. Code: {0} {1}", result, errorMsg);
        }
    }

    void run(vpux::ExecutionPlan& plan, npu_mlir_runtime_execute_params_t* pParams) {
        DebugTrace dbg("VirtualMachine::run", _logger);
        if (plan.entryFuncId >= plan.functions.size()) {
            OPENVINO_THROW("Bad entryFuncId: ", plan.entryFuncId);
        }
        const auto& entryFn = plan.functions[plan.entryFuncId];

        Frame fr;
        fr.funcId = plan.entryFuncId;
        fr.pc = 0;
        fr.i64.assign(entryFn.numI64, 0);
        fr.m.assign(entryFn.numMem, {});

        _logger.info("Starting execution at funcId={0}", fr.funcId);
        _logger.info("Initial frame: i64.size={0}, memref.size={1}", fr.i64.size(), fr.m.size());

        // binds operands to memref registers
        for (size_t i = 0; i < plan.operands.size() && i < fr.m.size() && i < 2; ++i) {
            auto handle = i == 0 ? reinterpret_cast<MemRefHandle*>(pParams->pInputs[i])
                                 : reinterpret_cast<MemRefHandle*>(pParams->pOutputs[i - 1]);
            int64_t dummyOffset = 0;
            handle->parseMemRef((const void**)&fr.m[i].data, (const void**)&fr.m[i].data, &dummyOffset,
                                (int64_t*)fr.m[i].sizes, (int64_t*)fr.m[i].strides, (int64_t*)&fr.m[i].dimCount);
            // MemRefHandle does not store elementByteSize, so we get it from plan.operands
            fr.m[i].elementByteSize = plan.operands[i].elementByteSize;
        }
        std::vector<CallFrame> callStack;
        auto loadFunc = [&](uint32_t funcId) -> const vpux::Function& {
            if (funcId >= plan.functions.size()) {
                OPENVINO_THROW("Bad funcId");
            }
            return plan.functions[funcId];
        };

        while (true) {
            const auto& fn = loadFunc(fr.funcId);
            if (fr.pc >= fn.code.size()) {
                OPENVINO_THROW("PC out of range");
            }
            const auto& instr = fn.code[fr.pc];
            _logger.info("Executing funcId={0} pc={1} op={2}", fr.funcId, fr.pc, static_cast<int>(instr.op));
            switch (instr.op) {
            case vpux::OpCode::I64_CONST: {
                DebugTrace dbgConst("OpCode::I64_CONST", _logger);
                // a=dstReg, imm=value
                if (instr.a >= fr.i64.size()) {
                    OPENVINO_THROW("I64_CONST: bad i64 reg");
                }
                fr.i64[instr.a] = instr.imm;
                fr.pc++;
                _logger.info("I64_CONST: Set i64[{0}] = {1}", instr.a, instr.imm);
                break;
            }
            case vpux::OpCode::I64_MEMREF_DIM: {
                DebugTrace dbgDim("OpCode::I64_MEMREF_DIM", _logger);
                // a=dstReg, b=memSlot, imm=dimIndex
                if (instr.a >= fr.i64.size()) {
                    OPENVINO_THROW("I64_MEMREF_DIM: bad i64 reg");
                }
                if (instr.b >= fr.m.size()) {
                    OPENVINO_THROW("I64_MEMREF_DIM: bad memref slot");
                }
                const vpux::MemRefDesc& memRef = fr.m[instr.b];
                if (instr.imm >= memRef.dimCount) {
                    OPENVINO_THROW("I64_MEMREF_DIM: bad dim index");
                }
                fr.i64[instr.a] = static_cast<int64_t>(memRef.sizes[instr.imm]);
                _logger.info("I64_MEMREF_DIM: Set i64[{0}] = memref[{1}].sizes[{2}] = {3}", instr.a, instr.b, instr.imm,
                             memRef.sizes[instr.imm]);
                fr.pc++;
                break;
            }
            case vpux::OpCode::I64_MOV: {
                DebugTrace dbgMov("OpCode::I64_MOV", _logger);
                // a=dstReg, b=srcReg
                if (instr.a >= fr.i64.size() || instr.b >= fr.i64.size()) {
                    OPENVINO_THROW("I64_MOV: bad i64 reg");
                }
                fr.i64[instr.a] = fr.i64[instr.b];
                _logger.info("I64_MOV: Set i64[{0}] = i64[{1}] = {2}", instr.a, instr.b, fr.i64[instr.a]);
                fr.pc++;
                break;
            }
            case vpux::OpCode::I64_CMP: {
                DebugTrace dbgCmp("OpCode::I64_CMP_LT", _logger);
                // a=boolReg, b=lhsReg, c=rhsReg
                if (instr.a >= fr.i64.size() || instr.b >= fr.i64.size() || instr.c >= fr.i64.size()) {
                    _logger.error("I64_CMP: a={0} b={1} c={2} i64.size={3}", instr.a, instr.b, instr.c, fr.i64.size());
                    OPENVINO_THROW("I64_CMP: bad reg");
                }
                switch (static_cast<vpux::CmpPred>(instr.imm)) {
                case vpux::CmpPred::LT:
                    fr.i64[instr.a] = (fr.i64[instr.b] < fr.i64[instr.c]) ? 1 : 0;
                    break;
                case vpux::CmpPred::LE:
                    fr.i64[instr.a] = (fr.i64[instr.b] <= fr.i64[instr.c]) ? 1 : 0;
                    break;
                case vpux::CmpPred::GT:
                    fr.i64[instr.a] = (fr.i64[instr.b] > fr.i64[instr.c]) ? 1 : 0;
                    break;
                case vpux::CmpPred::GE:
                    fr.i64[instr.a] = (fr.i64[instr.b] >= fr.i64[instr.c]) ? 1 : 0;
                    break;
                case vpux::CmpPred::EQ:
                    fr.i64[instr.a] = (fr.i64[instr.b] == fr.i64[instr.c]) ? 1 : 0;
                    break;
                case vpux::CmpPred::NE:
                    fr.i64[instr.a] = (fr.i64[instr.b] != fr.i64[instr.c]) ? 1 : 0;
                    break;
                default:
                    OPENVINO_THROW("I64_CMP: invalid comparison predicate");
                }
                fr.pc++;
                break;
            }
            case vpux::OpCode::I64_SELECT: {
                DebugTrace dbgSelect("OpCode::I64_SELECT", _logger);
                // a=dstReg, b=condition, c=trueReg, d=falseReg
                if (instr.a >= fr.i64.size() || instr.b >= fr.i64.size() || instr.c >= fr.i64.size() ||
                    instr.d >= fr.i64.size()) {
                    OPENVINO_THROW("I64_SELECT: bad i64 reg");
                }
                fr.i64[instr.a] = (fr.i64[instr.b] != 0) ? fr.i64[instr.c] : fr.i64[instr.d];
                _logger.info("I64_SELECT: Set i64[{0}] = (i64[{1}] != 0) ? i64[{2}] : i64[{3}] = {4}", instr.a, instr.b,
                             instr.c, instr.d, fr.i64[instr.a]);
                fr.pc++;
                break;
            }
            case vpux::OpCode::I64_ADD: {
                DebugTrace dbgAdd("OpCode::I64_ADD", _logger);
                // a=dstReg, b=src1Reg, c=src2Reg
                if (instr.a >= fr.i64.size() || instr.b >= fr.i64.size() || instr.c >= fr.i64.size()) {
                    OPENVINO_THROW("I64_ADD: bad i64 reg");
                }
                fr.i64[instr.a] = fr.i64[instr.b] + fr.i64[instr.c];
                _logger.info("I64_ADD: Set i64[{0}] = i64[{1}] + i64[{2}] = {3}", instr.a, instr.b, instr.c,
                             fr.i64[instr.a]);
                fr.pc++;
                break;
            }
            case vpux::OpCode::I64_SUB: {
                DebugTrace dbgSub("OpCode::I64_SUB", _logger);
                // a=dstReg, b=src1Reg, c=src2Reg
                if (instr.a >= fr.i64.size() || instr.b >= fr.i64.size() || instr.c >= fr.i64.size()) {
                    OPENVINO_THROW("I64_SUB: bad i64 reg");
                }
                fr.i64[instr.a] = fr.i64[instr.b] - fr.i64[instr.c];
                _logger.info("I64_SUB: Set i64[{0}] = i64[{1}] - i64[{2}] = {3}", instr.a, instr.b, instr.c,
                             fr.i64[instr.a]);
                fr.pc++;
                break;
            }
            case vpux::OpCode::I64_MUL: {
                DebugTrace dbgMul("OpCode::I64_MUL", _logger);
                // a=dstReg, b=src1Reg, c=src2Reg
                if (instr.a >= fr.i64.size() || instr.b >= fr.i64.size() || instr.c >= fr.i64.size()) {
                    OPENVINO_THROW("I64_MUL: bad i64 reg");
                }
                fr.i64[instr.a] = fr.i64[instr.b] * fr.i64[instr.c];
                _logger.info("I64_MUL: Set i64[{0}] = i64[{1}] * i64[{2}] = {3}", instr.a, instr.b, instr.c,
                             fr.i64[instr.a]);
                fr.pc++;
                break;
            }
            case vpux::OpCode::I64_DIV: {
                DebugTrace dbgDiv("OpCode::I64_DIV", _logger);
                // a=dstReg, b=src1Reg, c=src2Reg
                if (instr.a >= fr.i64.size() || instr.b >= fr.i64.size() || instr.c >= fr.i64.size()) {
                    OPENVINO_THROW("I64_DIV: bad i64 reg");
                }
                if (fr.i64[instr.c] == 0) {
                    OPENVINO_THROW("I64_DIV: division by zero");
                }
                fr.i64[instr.a] = fr.i64[instr.b] / fr.i64[instr.c];
                _logger.info("I64_DIV: Set i64[{0}] = i64[{1}] / i64[{2}] = {3}", instr.a, instr.b, instr.c,
                             fr.i64[instr.a]);
                fr.pc++;
                break;
            }
            case vpux::OpCode::I64_REM: {
                DebugTrace dbgRem("OpCode::I64_REM", _logger);
                // a=dstReg, b=src1Reg, c=src2Reg
                if (instr.a >= fr.i64.size() || instr.b >= fr.i64.size() || instr.c >= fr.i64.size()) {
                    OPENVINO_THROW("I64_REM: bad i64 reg");
                }
                if (fr.i64[instr.c] == 0) {
                    OPENVINO_THROW("I64_REM: division by zero");
                }
                fr.i64[instr.a] = fr.i64[instr.b] % fr.i64[instr.c];
                _logger.info("I64_REM: Set i64[{0}] = i64[{1}] % i64[{2}] = {3}", instr.a, instr.b, instr.c,
                             fr.i64[instr.a]);
                fr.pc++;
                break;
            }
            case vpux::OpCode::I64_MIN: {
                DebugTrace dbgMin("OpCode::I64_MIN", _logger);
                // a=dstReg, b=src1Reg, c=src2Reg
                if (instr.a >= fr.i64.size() || instr.b >= fr.i64.size() || instr.c >= fr.i64.size()) {
                    OPENVINO_THROW("I64_MIN: bad i64 reg");
                }
                fr.i64[instr.a] = std::min(fr.i64[instr.b], fr.i64[instr.c]);
                _logger.info("I64_MIN: Set i64[{0}] = min(i64[{1}], i64[{2}]) = {3}", instr.a, instr.b, instr.c,
                             fr.i64[instr.a]);
                fr.pc++;
                break;
            }
            case vpux::OpCode::JMP: {
                DebugTrace dbgJmp("OpCode::JMP", _logger);
                // a=targetPc
                fr.pc = instr.a;
                _logger.info("JMP: Set pc = {0}", fr.pc);
                break;
            }
            case vpux::OpCode::BR_IF: {
                DebugTrace dbgBrIf("OpCode::BR_IF", _logger);
                // a=boolReg, b=pcTrue, c=pcFalse
                if (instr.a >= fr.i64.size()) {
                    OPENVINO_THROW("BR_IF: bad bool reg");
                }
                if (fr.i64[instr.a]) {
                    fr.pc = instr.b;
                } else {
                    fr.pc = instr.c;
                }
                _logger.info("BR_IF: Set pc = {0}", fr.pc);
                break;
            }
            case vpux::OpCode::RET: {
                DebugTrace dbgRet("OpCode::RET", _logger);
                if (callStack.empty()) {
                    _logger.info("Program completed");
                    return;
                }
                const auto& callFrame = callStack.back();
                fr.funcId = callFrame.funcId;
                fr.pc = callFrame.returnPc;
                callStack.pop_back();
                break;
            }
            case vpux::OpCode::ASSERT: {
                DebugTrace dbgAssert("OpCode::ASSERT", _logger);
                // a=boolReg
                if (instr.a >= fr.i64.size()) {
                    OPENVINO_THROW("ASSERT: bad bool reg");
                }
                if (fr.i64[instr.a] == 0) {
                    OPENVINO_THROW("ASSERT failed at funcId=", fr.funcId, " pc=", fr.pc);
                }
                fr.pc++;
                _logger.info("ASSERT: i64[{0}] != 0 passed", instr.a);
                break;
            }
            case vpux::OpCode::CALL: {
                DebugTrace dbgCall("OpCode::CALL", _logger);
                callStack.push_back({fr.funcId, fr.pc + 1});

                const vpux::Function& callee = loadFunc(instr.a);
                // Shared-frame prototype: grow storage to meet callee needs.
                if (fr.i64.size() < callee.numI64) {
                    fr.i64.resize(callee.numI64, 0);
                }
                if (fr.m.size() < callee.numMem) {
                    fr.m.resize(callee.numMem, {});
                }

                fr.funcId = instr.a;
                fr.pc = 0;
                break;
            }
            case vpux::OpCode::MEMREF_SUBVIEW: {
                DebugTrace dbgSubview("OpCode::MEMREF_SUBVIEW", _logger);
                // a=dstMemSlot, b=baseMemSlot, imm=subviewTemplateId
                if (instr.a >= fr.m.size() || instr.b >= fr.m.size()) {
                    OPENVINO_THROW("MEMREF_SUBVIEW: bad memref slot");
                }
                int64_t tid = instr.imm;
                if (tid >= plan.subviews.size()) {
                    OPENVINO_THROW("MEMREF_SUBVIEW: bad subview template id");
                }
                applySubview(fr.m[instr.b], plan.subviews[tid], fr.i64, fr.m[instr.a]);
                fr.pc++;
                _logger.info("MEMREF_SUBVIEW: Set memref[{0}] = subview(memref[{1}])", instr.a, instr.b);
                break;
            }
            case vpux::OpCode::CALL_KERNEL: {
                DebugTrace dbgRunStage("OpCode::CALL_KERNEL", _logger);
                int64_t descId = instr.imm;
                const vpux::KernelDesc& kernelDesc = plan.kernelDescs[descId];

                std::vector<vpux::MemRefDesc> kernelArgs;
                kernelArgs.reserve(kernelDesc.argSlots.size());
                for (auto memSlot : kernelDesc.argSlots) {
                    if (memSlot >= fr.m.size()) {
                        OPENVINO_THROW("CALL_KERNEL: bad memref slot");
                    }
                    kernelArgs.push_back(fr.m[memSlot]);
                    _logger.info("CALL_KERNEL: arg memref[{0}] data={1} dimCount={2}", memSlot, fr.m[memSlot].data,
                                 fr.m[memSlot].dimCount);
                }
                vpux::Kernel& kernel = plan.kernelMap.at(kernelDesc.kernelId);
                // _logger.info("CALL_KERNEL: Running kernel {0}", kernel.symbolName);
                runKernel(kernel, kernelArgs, pParams);
                _logger.info("CALL_KERNEL: Executed kernel {0}", kernelDesc.kernelId);
                fr.pc++;
                break;
            }
            case vpux::OpCode::SUBMIT: {
                DebugTrace dbgSubmit("OpCode::SUBMIT", _logger);
                npu_level_zero_submit_commandlist((void**)pParams->commandLists, pParams->commandQueue,
                                                  pParams->inferenceFence, pParams->event);
                fr.pc++;
                break;
            }
            default:
                OPENVINO_THROW("Unsupported opcode: ", static_cast<int>(instr.op));
            }
        }
    }
};

class NPUMLIRRuntime {
public:
    NPUMLIRRuntime(const npu_mlir_runtime_blob_desc_t* desc, npu_mlir_runtime_properties_t* pProperties);
    ~NPUMLIRRuntime();

    void createExecutionEngine(const npu_mlir_runtime_blob_desc_t* blob);

    void parseMetadata();

    void getArgumentProperties(uint32_t argIndex, ze_graph_argument_properties_3_t* pGraphArgumentProperties,
                               ze_graph_argument_metadata_t* pGraphArgumentMetadata);

    void execute(npu_mlir_runtime_execute_params_t* pParams);

    void predictOutputShape(npu_mlir_runtime_predict_output_shape_params_t* pParams);

private:
    vpux::ExecutionPlan _execPlan;
    intel_npu::NetworkMetadata _metadata;
    // Seems new version use ze_graph_argument_properties_3_t instead of ArgumentDescriptor in metadata function
    std::vector<ArgumentDescriptor> _inputs;
    std::vector<ArgumentDescriptor> _outputs;
    uint32_t _numOfSubgraphs = 0;
    uint32_t _numOfArgs = 0;
    Logger _logger = Logger("NPUMLIRRuntime", Logger::global().level());
    std::vector<vpux::MemRefDesc> inputTileMemRefDescs;
    std::vector<vpux::MemRefDesc> outputTileMemRefDescs;
    VirtualMachine _vm;
};

class MemoryStreamBuf : public std::streambuf {
public:
    MemoryStreamBuf(const void* data, std::size_t size) {
        char* begin = static_cast<char*>(const_cast<void*>(data));
        setg(begin, begin, begin + size);
    }
};

void NPUMLIRRuntime::createExecutionEngine(const npu_mlir_runtime_blob_desc_t* desc) {
    _logger.debug("Creating execution engine from blob at {0} of size {1}", desc->pInput, desc->inputSize);
    _logger.debug("Created ExecutionEngine");
}

void NPUMLIRRuntime::parseMetadata() {
    _logger.debug("Parsing metadata");
    auto result = npu_level_zero_get_network_metadata(_execPlan.binaryMetadata.data(), _execPlan.binaryMetadata.size(),
                                                      &_metadata, &_inputs, &_outputs);

    if (result) {
        OPENVINO_THROW("Error invoking main: ", result);
    }
    _logger.debug("num of subgraphs: {0} inputs: {1} outputs: {2}", _numOfSubgraphs, _inputs.size(), _outputs.size());
    _metadata.bindRelatedDescriptors();
    _numOfArgs = static_cast<uint32_t>(_inputs.size() + _outputs.size());
    for (size_t i = 0; i < _inputs.size(); i++) {
        _metadata.inputs[i].indexUsedByDriver = _inputs[i].idx;
    }

    for (size_t i = 0; i < _outputs.size(); i++) {
        _metadata.outputs[i].indexUsedByDriver = _outputs[i].idx;
    }
    _logger.debug("Parsed metadata");
}

NPUMLIRRuntime::NPUMLIRRuntime(const npu_mlir_runtime_blob_desc_t* desc, npu_mlir_runtime_properties_t* pProperties) {
    _logger.debug("Constructor");
    npu_level_zero_init();

    MemoryStreamBuf buf(desc->pInput, desc->inputSize);
    std::istream is(&buf);
    _execPlan.deserialize(is);
    _execPlan.print();

    parseMetadata();
    _numOfArgs = _execPlan.operands.size();
    // TODO: only one subgraph is supported for now
    _numOfSubgraphs = 1;

    pProperties->numOfSubGraphs = _numOfSubgraphs;
    pProperties->numOfGraphArgs = _numOfArgs;

    // TODO: only one input and output is supported for now
    inputTileMemRefDescs.resize(1);
    outputTileMemRefDescs.resize(1);

    _logger.debug("Constructor - done");
}

NPUMLIRRuntime::~NPUMLIRRuntime() {
    _logger.debug("Destructor");
    npu_level_zero_destroy();
    _logger.debug("Destructor - done");
}

void NPUMLIRRuntime::getArgumentProperties(uint32_t argIndex,
                                           ze_graph_argument_properties_3_t* pGraphArgumentProperties,
                                           ze_graph_argument_metadata_t* pGraphArgumentMetadata) {
    _logger.debug("Getting argument properties for index {0}", argIndex);
    if (argIndex >= _numOfArgs) {
        OPENVINO_THROW("Invalid argument index {0}", argIndex);
    }

    const ArgumentDescriptor* argDesc = nullptr;
    IODescriptor desc;
    if (argIndex < _inputs.size()) {
        argDesc = &_inputs[argIndex];
        desc = _metadata.inputs[argIndex];
    } else {
        argDesc = &_outputs[argIndex - _inputs.size()];
        desc = _metadata.outputs[argIndex - _inputs.size()];
    }

    // Define new struct to hold metadata
    *pGraphArgumentProperties = argDesc->info;

    // Fill in metadata struct
    pGraphArgumentMetadata->stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_METADATA;
    pGraphArgumentMetadata->pNext = nullptr;
    pGraphArgumentMetadata->type = argDesc->info.type;
    std::strncpy(pGraphArgumentMetadata->friendly_name, argDesc->info.name, ZE_MAX_GRAPH_ARGUMENT_NAME);
    pGraphArgumentMetadata->data_type = ZE_GRAPH_METADATA_TYPE_UNDEFINED;

    if (desc.shapeFromIRModel.has_value()) {
        // Only care about shape, this is shapeFromIRModel
        for (size_t i = 0; i < desc.shapeFromIRModel->size() && i < ZE_MAX_GRAPH_TENSOR_REF_DIMS; ++i) {
            auto val = desc.shapeFromIRModel.value()[i];
            pGraphArgumentMetadata->shape[i] =
                    val.is_dynamic() ? std::numeric_limits<uint64_t>::max() : val.get_length();
        }
    } else {
        // Use shapeFromCompiler
        std::copy(std::begin(desc.shapeFromCompiler.get_shape()), std::end(desc.shapeFromCompiler.get_shape()),
                  std::begin(pGraphArgumentMetadata->shape));
    }
    pGraphArgumentMetadata->shape_size = argDesc->info.dims_count;
    pGraphArgumentMetadata->tensor_names_count = 0;  // Not used
    std::strncpy(pGraphArgumentMetadata->input_name, argDesc->info.name, ZE_MAX_GRAPH_ARGUMENT_NAME);

    // Dump argDesc info
    _logger.debug("Argument Descriptor Info:");
    _logger.debug("  Name: {0}", argDesc->info.name);
    _logger.debug("  Type: {0}", argDesc->info.type);
    _logger.debug("  Dimensions: {0}", argDesc->info.dims_count);
    for (size_t i = 0; i < argDesc->info.dims_count; ++i) {
        _logger.debug("    Dim[{0}]: {1}", i, argDesc->info.dims[i]);
    }

    // Dump metadata info
    _logger.debug("Graph Argument Metadata Info:");
    _logger.debug("  Input Name: {0}", pGraphArgumentMetadata->input_name);
    _logger.debug("  Shape Size: {0}", pGraphArgumentMetadata->shape_size);
    for (size_t i = 0; i < pGraphArgumentMetadata->shape_size; ++i) {
        _logger.debug("    Shape[{0}]: {1}", i, pGraphArgumentMetadata->shape[i]);
    }
    _logger.debug("Get properties done");
}

void NPUMLIRRuntime::execute(npu_mlir_runtime_execute_params_t* pParams) {
    _logger.debug("Executing with {0} inputs and {1} outputs", pParams->numOfInputs, pParams->numOfOutputs);
    if (pParams->numOfInputs != 1 || pParams->numOfOutputs != 1) {
        _logger.error("Host runtime support only single input and output for now");
        return;
    }
    // parse inputs
    MemRefHandle* input = reinterpret_cast<MemRefHandle*>(pParams->pInputs[0]);
    MemRefHandle* output = reinterpret_cast<MemRefHandle*>(pParams->pOutputs[0]);
    MemRefNDRef inputMemref(input->memRefBufferPtr, input->dimCount);
    MemRefNDRef outputMemref(output->memRefBufferPtr, output->dimCount);

    // _logger.debug("Input MemRef: {0}", input->toString());
    // _logger.debug("Output MemRef: {0}", output->toString());

    _vm.run(_execPlan, pParams);
}

void NPUMLIRRuntime::predictOutputShape(npu_mlir_runtime_predict_output_shape_params_t* params) {
    _logger.debug("Predict output shape from input");
    _logger.debug("Executing with {0} inputs and {1} outputs", params->numOfInputs, params->numOfOutputs);
    // for (uint32_t i = 0; i < params->numOfInputs; i++) {
        // MemRefHandle* input = reinterpret_cast<MemRefHandle*>(params->pInputs[i]);
        // _logger.debug("Input : {0}, info: {1}", i, input->toString());
    // }

    for (uint32_t i = 0; i < params->numOfOutputs; i++) {
        MemRefHandle* output = reinterpret_cast<MemRefHandle*>(params->pOutputs[i]);
        MemRefNDRef outputMemref(output->memRefBufferPtr, output->dimCount);

        MemRefHandle* input = reinterpret_cast<MemRefHandle*>(params->pInputs[i]);
        MemRefNDRef inputMemref(input->memRefBufferPtr, input->dimCount);
        outputMemref.setSizes<int64_t>(inputMemref.getSizes(), inputMemref.dimCount);
        outputMemref.setStrides<int64_t>(inputMemref.getStrides(), inputMemref.dimCount);
        // _logger.debug("Output : {0}, info: {1}", i, output->toString());
    }
    _logger.debug("PredictOutputShape done");
    return;
}

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __attribute__((visibility("default")))
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Get API version
DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL
npuMLIRRuntimeGetAPIVersion(npu_mlir_runtime_version_t* pVersion) {
    if (pVersion == nullptr) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    *pVersion = NPU_MLIR_RUNTIME_VERSION_CURRENT;
    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Init MLIR runtime instance and return handle
DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL
npuMLIRRuntimeCreate(const npu_mlir_runtime_blob_desc_t* desc, npu_mlir_runtime_handle_t* phRuntime,
                     npu_mlir_runtime_properties_t* pProperties) {
    if (phRuntime == nullptr || desc == nullptr) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = new NPUMLIRRuntime(desc, pProperties);
        *phRuntime = reinterpret_cast<npu_mlir_runtime_handle_t>(runtime);
    } catch (const std::exception& e) {
        Logger::global().error("npuMLIRRuntimeCreate - Error creating MLIR runtime: {0}", e.what());
        return NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Destroy MLIR runtime instance
DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL npuMLIRRuntimeDestroy(npu_mlir_runtime_handle_t hRuntime) {
    if (hRuntime == nullptr) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        delete runtime;
    } catch (const std::exception& e) {
        Logger::global().error("npuMLIRRuntimeDestroy - Error destroying MLIR runtime: {0}", e.what());
        return NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
/// @brief Get metadata from MLIR runtime instance
DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL
npuMLIRRuntimeGetMetadata(npu_mlir_runtime_handle_t hRuntime, uint32_t argIndex,
                          ze_graph_argument_properties_3_t* pGraphArgumentProperties,
                          ze_graph_argument_metadata_t* pGraphArgumentMetadata, int64_t* upperBound) {
    if (hRuntime == nullptr || pGraphArgumentProperties == nullptr || pGraphArgumentMetadata == nullptr) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        runtime->getArgumentProperties(argIndex, pGraphArgumentProperties, pGraphArgumentMetadata);
    } catch (const std::exception& e) {
        Logger::global().error("Error getting argument properties: {0}", e.what());
        return NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL
npuMLIRRuntimeExecute(npu_mlir_runtime_handle_t hRuntime, npu_mlir_runtime_execute_params_t* pParams) {
    if (hRuntime == nullptr || pParams == nullptr) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        runtime->execute(pParams);
    } catch (const std::exception& e) {
        Logger::global().error("npuMLIRRuntimeExecute - Error executing MLIR runtime: {0}", e.what());
        return NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL npuMLIRRuntimePredictOutputShape(
        npu_mlir_runtime_handle_t hRuntime, npu_mlir_runtime_predict_output_shape_params_t* pParams) {
    if (hRuntime == nullptr || pParams == nullptr || pParams->pInputs == nullptr || pParams->pOutputs == nullptr) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        NPUMLIRRuntime* runtime = reinterpret_cast<NPUMLIRRuntime*>(hRuntime);
        runtime->predictOutputShape(pParams);
    } catch (const std::exception& e) {
        Logger::global().error("npuMLIRRuntimePredictOutputShape - Error executing MLIR runtime: {0}", e.what());
        return NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL
npuMLIRRuntimeCreateMemRef(int64_t dimsCount, npu_mlir_runtime_mem_ref_handle_t* phMemRef) {
    if (phMemRef == nullptr || dimsCount == 0) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    try {
        // Now just support up to 5 since ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE is 5
        MemRefHandle* memRef = new MemRefHandle(dimsCount);
        *phMemRef = reinterpret_cast<npu_mlir_runtime_mem_ref_handle_t>(memRef);
    } catch (const std::exception& e) {
        Logger::global().error("npuMLIRRuntimeCreateMemRef - Error creating MemRef: {0}", e.what());
        return NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL
npuMLIRRuntimeDestroyMemRef(npu_mlir_runtime_mem_ref_handle_t hMemRef) {
    if (hMemRef == nullptr) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    try {
        MemRefHandle* memRef = reinterpret_cast<MemRefHandle*>(hMemRef);
        delete memRef;
    } catch (const std::exception& e) {
        Logger::global().error("npuMLIRRuntimeDestroyMemRef - Error destroying MemRef: {0}", e.what());
        return NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL
npuMLIRRuntimeSetMemRef(npu_mlir_runtime_mem_ref_handle_t hMemRef, const void* basePtr, const void* data,
                        int64_t offset, int64_t* pSizes, int64_t* pStrides, int64_t dimsCount) {
    if (hMemRef == nullptr || pSizes == nullptr || pStrides == nullptr) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    try {
        // cout params
        MemRefHandle* memRef = reinterpret_cast<MemRefHandle*>(hMemRef);
        MemRefNDRef ref(memRef->memRefBufferPtr, dimsCount);
        ref.setAllocated(basePtr);
        ref.setAligned(data);
        ref.setOffset(offset);
        ref.setSizes(pSizes, dimsCount);
        ref.setStrides(pStrides, dimsCount);
    } catch (const std::exception& e) {
        Logger::global().error("npuMLIRRuntimeSetMemRef - Error setting MemRef: {0}", e.what());
        return NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

DLLEXPORT npu_mlir_runtime_result_t NPU_MLIR_RUNTIME_APICALL
npuMLIRRuntimeParseMemRef(npu_mlir_runtime_mem_ref_handle_t hMemRef, const void** pBasePtr, const void** pData,
                          int64_t* pOffset, int64_t* pSizes, int64_t* pStrides, int64_t* pDimsCount) {
    if (hMemRef == nullptr || pBasePtr == nullptr || pData == nullptr || pOffset == nullptr || pSizes == nullptr ||
        pStrides == nullptr || pDimsCount == nullptr) {
        return NPU_MLIR_RUNTIME_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    try {
        MemRefHandle* memRef = reinterpret_cast<MemRefHandle*>(hMemRef);
        memRef->parseMemRef(pBasePtr, pData, pOffset, pSizes, pStrides, pDimsCount);
    } catch (const std::exception& e) {
        Logger::global().error("npuMLIRRuntimeSetMemRef - Error parsing MemRef: {0}", e.what());
        return NPU_MLIR_RUNTIME_RESULT_ERROR_UNKNOWN;
    }

    return NPU_MLIR_RUNTIME_RESULT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
