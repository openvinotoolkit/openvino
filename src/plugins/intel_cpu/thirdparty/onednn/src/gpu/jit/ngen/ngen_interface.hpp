/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef NGEN_INTERFACE_HPP
#define NGEN_INTERFACE_HPP


#include "ngen_core.hpp"
#include <sstream>


namespace ngen {

template <HW hw> class OpenCLCodeGenerator;
template <HW hw> class L0CodeGenerator;

// Exceptions.
#ifdef NGEN_SAFE
class unknown_argument_exception : public std::runtime_error {
public:
    unknown_argument_exception() : std::runtime_error("Argument not found") {}
};

class bad_argument_type_exception : public std::runtime_error {
public:
    bad_argument_type_exception() : std::runtime_error("Bad argument type") {}
};

class interface_not_finalized : public std::runtime_error {
public:
    interface_not_finalized() : std::runtime_error("Interface has not been finalized") {}
};

class use_simd1_local_id_exception : public std::runtime_error {
public:
    use_simd1_local_id_exception() : std::runtime_error("Use getSIMD1LocalID for SIMD1 kernels") {}
};
#endif

enum class ExternalArgumentType { Scalar, GlobalPtr, LocalPtr, Hidden };
enum class GlobalAccessType { None = 0, Stateless = 1, Surface = 2, All = 3 };

static inline GlobalAccessType operator|(GlobalAccessType access1, GlobalAccessType access2)
{
    return static_cast<GlobalAccessType>(static_cast<int>(access1) | static_cast<int>(access2));
}

class InterfaceHandler
{
    template <HW hw> friend class OpenCLCodeGenerator;
    template <HW hw> friend class L0CodeGenerator;

public:
    InterfaceHandler(HW hw_) : hw(hw_), simd(GRF::bytes(hw_) >> 2) {}

    inline void externalName(const std::string &name)   { kernelName = name; }

    template <typename DT>
    inline void newArgument(std::string name)           { newArgument(name, getDataType<DT>()); }
    inline void newArgument(std::string name, DataType type, ExternalArgumentType exttype = ExternalArgumentType::Scalar, GlobalAccessType access = GlobalAccessType::All);
    inline void newArgument(std::string name, ExternalArgumentType exttype, GlobalAccessType access = GlobalAccessType::All);

    inline Subregister getArgument(const std::string &name) const;
    inline Subregister getArgumentIfExists(const std::string &name) const;
    inline int getArgumentSurface(const std::string &name) const;
    inline int getArgumentSurfaceIfExists(const std::string &name) const;
    inline GRF getLocalID(int dim) const;
    inline RegData getSIMD1LocalID(int dim) const;
    inline Subregister getLocalSize(int dim) const;

    const std::string &getExternalName() const           { return kernelName; }
    int getSIMD() const                                  { return simd; }
    int getGRFCount() const                              { return needGRF; }
    size_t getSLMSize() const                            { return slmSize; }

    void require32BitBuffers()                           { allow64BitBuffers = false; }
    void requireBarrier()                                { barrierCount = 1; }
    void requireDPAS()                                   { needDPAS = true; }
    void requireGlobalAtomics()                          { needGlobalAtomics = true; }
    void requireGRF(int grfs)                            { needGRF = grfs; }
    void requireLocalID(int dimensions)                  { needLocalID = dimensions; }
    void requireLocalSize()                              { needLocalSize = true; }
    void requireNonuniformWGs()                          { needNonuniformWGs = true; }
    void requireNoPreemption()                           { needNoPreemption = true; }
    void requireScratch(size_t bytes = 1)                { scratchSize = bytes; }
    void requireSIMD(int simd_)                          { simd = simd_; }
    void requireSLM(size_t bytes)                        { slmSize = bytes; }
    void requireStatelessWrites(bool req = true)         { needStatelessWrites = req; }
    inline void requireType(DataType type);
    template <typename T> void requireType()             { requireType(getDataType<T>()); }
    void requireWalkOrder(int o1, int o2)                { walkOrder[0] = o1; walkOrder[1] = o2; walkOrder[2] = -1; }
    void requireWalkOrder(int o1, int o2, int o3)        { walkOrder[0] = o1; walkOrder[1] = o2; walkOrder[2] = o3; }
    void requireWorkgroup(size_t x, size_t y = 1,
                          size_t z = 1)                  { wg[0] = x; wg[1] = y; wg[2] = z; }

    void setSkipPerThreadOffset(int32_t offset)          { offsetSkipPerThread = offset; }
    void setSkipCrossThreadOffset(int32_t offset)        { offsetSkipCrossThread = offset; }

    inline void finalize();

    template <typename CodeGenerator>
    inline void generatePrologue(CodeGenerator &generator, const GRF &temp = GRF(127)) const;

    inline void generateDummyCL(std::ostream &stream) const;
    inline std::string generateZeInfo() const;

#ifdef NGEN_ASM
    inline void dumpAssignments(std::ostream &stream) const;
#endif

    static constexpr int noSurface = 0x80;        // Returned by getArgumentSurfaceIfExists in case of no surface assignment

protected:
    struct Assignment {
        std::string name;
        DataType type;
        ExternalArgumentType exttype;
        GlobalAccessType access;
        Subregister reg;
        int surface;
        int index;

        bool globalSurfaceAccess()   const { return (static_cast<int>(access) & static_cast<int>(GlobalAccessType::Surface)); }
        bool globalStatelessAccess() const { return (static_cast<int>(access) & static_cast<int>(GlobalAccessType::Stateless)); }
    };

    HW hw;

    std::vector<Assignment> assignments;
    std::string kernelName = "default_kernel";

    int nextArgIndex = 0;
    bool finalized = false;

    bool allow64BitBuffers = 0;
    int barrierCount = 0;
    bool needDPAS = false;
    bool needGlobalAtomics = false;
    int32_t needGRF = 128;
    int needLocalID = 0;
    bool needLocalSize = false;
    bool needNonuniformWGs = false;
    bool needNoPreemption = false;
    bool needHalf = false;
    bool needDouble = false;
    bool needStatelessWrites = true;
    int32_t offsetSkipPerThread = 0;
    int32_t offsetSkipCrossThread = 0;
    size_t scratchSize = 0;
    int simd = 8;
    size_t slmSize = 0;
    int walkOrder[3] = {-1, -1, -1};
    size_t wg[3] = {0, 0, 0};

    int crossthreadGRFs = 0;
    inline int getCrossthreadGRFs() const;
    inline GRF getCrossthreadBase(bool effective = true) const;
    int grfsPerLID() const { return (simd > 16 && GRF::bytes(hw) < 64) ? 2 : 1; }
};

using NEOInterfaceHandler = InterfaceHandler;

void InterfaceHandler::newArgument(std::string name, DataType type, ExternalArgumentType exttype, GlobalAccessType access)
{
    if (exttype != ExternalArgumentType::GlobalPtr)
        access = GlobalAccessType::None;
    assignments.push_back({name, type, exttype, access, Subregister{}, noSurface, nextArgIndex++});
}

void InterfaceHandler::newArgument(std::string name, ExternalArgumentType exttype, GlobalAccessType access)
{
    DataType type = DataType::invalid;

    switch (exttype) {
        case ExternalArgumentType::GlobalPtr: type = DataType::uq; break;
        case ExternalArgumentType::LocalPtr:  type = DataType::ud; break;
        default:
#ifdef NGEN_SAFE
            throw bad_argument_type_exception();
#else
        break;
#endif
    }

    newArgument(name, type, exttype, access);
}

Subregister InterfaceHandler::getArgumentIfExists(const std::string &name) const
{
    for (auto &assignment : assignments) {
        if (assignment.name == name)
            return assignment.reg;
    }

    return Subregister{};
}

Subregister InterfaceHandler::getArgument(const std::string &name) const
{
    Subregister arg = getArgumentIfExists(name);

#ifdef NGEN_SAFE
    if (arg.isInvalid())
        throw unknown_argument_exception();
#endif

    return arg;
}

int InterfaceHandler::getArgumentSurfaceIfExists(const std::string &name) const
{
    for (auto &assignment : assignments)
        if (assignment.name == name)
            return assignment.surface;
    return noSurface;
}

int InterfaceHandler::getArgumentSurface(const std::string &name) const
{
    int surface = getArgumentSurfaceIfExists(name);

#ifdef NGEN_SAFE
    if (surface == noSurface)
        throw unknown_argument_exception();
#endif
    return surface;
}

RegData InterfaceHandler::getSIMD1LocalID(int dim) const
{
#ifdef NGEN_SAFE
    if (dim > needLocalID || simd != 1) throw unknown_argument_exception();
#endif

    return GRF(1).uw(dim);
}

GRF InterfaceHandler::getLocalID(int dim) const
{
#ifdef NGEN_SAFE
    if (dim > needLocalID) throw unknown_argument_exception();
    if (simd == 1) throw use_simd1_local_id_exception();
#endif

    if (simd == 1)
        return GRF(1).uw();
    else
        return GRF(1 + dim * grfsPerLID()).uw();
}

void InterfaceHandler::requireType(DataType type)
{
    switch (type) {
        case DataType::hf: needHalf = true;   break;
        case DataType::df: needDouble = true; break;
        default: break;
    }
}

static inline const char *getCLDataType(DataType type)
{
    static const char *names[16] = {"uint", "int", "ushort", "short", "uchar", "char", "double", "float", "ulong", "long", "half", "ushort", "INVALID", "INVALID", "INVALID", "INVALID"};
    return names[static_cast<uint8_t>(type) & 0xF];
}

void InterfaceHandler::generateDummyCL(std::ostream &stream) const
{
#ifdef NGEN_SAFE
    if (!finalized) throw interface_not_finalized();
#endif
    const char *dpasDummy = "    int __builtin_IB_sub_group_idpas_s8_s8_8_1(int, int, int8) __attribute__((const));\n"
                            "    int z = __builtin_IB_sub_group_idpas_s8_s8_8_1(0, ____[0], 1);\n"
                            "    for (int i = 0; i < z; i++) (void) ____[0];\n";

    if (needHalf)   stream << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    if (needDouble) stream << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

    if (wg[0] > 0 && wg[1] > 0 && wg[2] > 0)
        stream << "__attribute__((reqd_work_group_size(" << wg[0] << ',' << wg[1] << ',' << wg[2] << ")))\n";
    if (walkOrder[0] >= 0) {
        stream << "__attribute__((intel_reqd_workgroup_walk_order(" << walkOrder[0] << ',' << walkOrder[1];
        if (walkOrder[2] >= 0)
            stream << ',' << walkOrder[2];
        stream << ")))\n";
    }
    stream << "__attribute__((intel_reqd_sub_group_size(" << simd << ")))\n";
    stream << "kernel void " << kernelName << '(';

    bool firstArg = true;
    for (const auto &assignment : assignments) {
        if (assignment.exttype == ExternalArgumentType::Hidden) continue;

        if (!firstArg) stream << ", ";

        switch (assignment.exttype) {
            case ExternalArgumentType::GlobalPtr: stream << "global void *"; break;
            case ExternalArgumentType::LocalPtr: stream << "local void *"; break;
            case ExternalArgumentType::Scalar: stream << getCLDataType(assignment.type) << ' '; break;
            default: break;
        }

        stream << assignment.name;
        firstArg = false;
    }
    stream << ") {\n";
    stream << "    global volatile int *____;\n";

    if (needLocalID)        stream << "    (void) ____[get_local_id(0)];\n";
    if (needLocalSize)      stream << "    (void) ____[get_enqueued_local_size(0)];\n";
    if (barrierCount > 0)   stream << "    barrier(CLK_GLOBAL_MEM_FENCE);\n";
    if (needDPAS)           stream << dpasDummy;
    if (needGlobalAtomics)  stream << "    atomic_inc(____);\n";
    if (scratchSize > 0)    stream << "    volatile char scratch[" << scratchSize << "] = {0};\n";
    if (slmSize > 0)        stream << "    volatile local char slm[" << slmSize << "]; slm[0]++;\n";
    if (needNoPreemption) {
        if (hw == HW::Gen9)
            stream << "    volatile double *__df; *__df = 1.1 / *__df;\n"; // IEEE macro causes IGC to disable MTP.
        /* To do: Gen11 */
    }

    if (hw >= HW::XeHP) for (const auto &assignment : assignments) {
        // Force IGC to assume stateless accesses could occur if necessary.
        if (assignment.exttype == ExternalArgumentType::GlobalPtr && assignment.globalStatelessAccess())
            stream << "    __asm__ volatile(\"\" :: \"rw.u\"(" << assignment.name << "));\n";
    }

    stream << "}\n";
}

inline Subregister InterfaceHandler::getLocalSize(int dim) const
{
    static const std::string localSizeArgs[3] = {"__local_size0", "__local_size1", "__local_size2"};
    return getArgument(localSizeArgs[dim]);
}

void InterfaceHandler::finalize()
{
    // Make assignments, following NEO rules:
    //  - all inputs are naturally aligned
    //  - all sub-DWord inputs are DWord-aligned
    //  - first register is
    //      r3 (no local IDs)
    //      r5 (SIMD8/16, local IDs)
    //      r8 (SIMD32, local IDs)
    // [- assign local ptr arguments left-to-right? not checked]
    //  - assign global pointer arguments left-to-right
    //  - assign scalar arguments left-to-right
    //  - assign surface indices left-to-right for global pointers
    //  - no arguments can cross a GRF boundary. Arrays like work size count
    //     as 1 argument for this rule.

    static const std::string localSizeArgs[3] = {"__local_size0", "__local_size1", "__local_size2"};
    static const std::string scratchSizeArg = "__scratch_size";

    GRF base = getCrossthreadBase();
    int offset = 32;
    int nextSurface = 0;
    const int grfSize = GRF::bytes(hw);

    auto assignArgsOfType = [&](ExternalArgumentType exttype) {
        for (auto &assignment : assignments) {
            if (assignment.exttype != exttype) continue;

            auto bytes = getBytes(assignment.type);
            auto size = getDwords(assignment.type) << 2;

            if (assignment.name == localSizeArgs[0]) {
                // Move to next GRF if local size arguments won't fit in this one.
                if (offset > grfSize - (3 * 4)) {
                    offset = 0;
                    base++;
                }
            }

            offset = (offset + size - 1) & -size;
            if (offset >= grfSize) {
                offset = 0;
                base++;
            }

            assignment.reg = base.sub(offset / bytes, assignment.type);

            if (assignment.exttype == ExternalArgumentType::GlobalPtr) {
                if (!assignment.globalStatelessAccess())
                    assignment.reg = Subregister{};
                if (assignment.globalSurfaceAccess())
                    assignment.surface = nextSurface;
                nextSurface++;
            }
            else if (assignment.exttype == ExternalArgumentType::Scalar)
                requireType(assignment.type);

            offset += size;
        }
    };

    assignArgsOfType(ExternalArgumentType::LocalPtr);
    assignArgsOfType(ExternalArgumentType::GlobalPtr);
    assignArgsOfType(ExternalArgumentType::Scalar);

    // Add private memory size arguments.
    if (scratchSize > 0)
        newArgument(scratchSizeArg, DataType::uq, ExternalArgumentType::Hidden);

    // Add enqueued local size arguments.
    if (needLocalSize && needNonuniformWGs)
        for (int dim = 0; dim < 3; dim++)
            newArgument(localSizeArgs[dim], DataType::ud, ExternalArgumentType::Hidden);

    assignArgsOfType(ExternalArgumentType::Hidden);

    crossthreadGRFs = base.getBase() - getCrossthreadBase().getBase() + 1;

    // Manually add regular local size arguments.
    if (needLocalSize && !needNonuniformWGs)
        for (int dim = 0; dim < 3; dim++)
            assignments.push_back({localSizeArgs[dim], DataType::ud, ExternalArgumentType::Hidden,
                                   GlobalAccessType::None, GRF(getCrossthreadBase()).ud(dim + 3), noSurface, -1});

    finalized = true;
}

GRF InterfaceHandler::getCrossthreadBase(bool effective) const
{
    if (!needLocalID)
        return GRF((!effective || (hw >= HW::XeHP)) ? 1 : 2);
    else if (simd == 1)
        return GRF(2);
    else
        return GRF(1 + 3 * grfsPerLID());
}

int InterfaceHandler::getCrossthreadGRFs() const
{
#ifdef NGEN_SAFE
    if (!finalized) throw interface_not_finalized();
#endif
    return crossthreadGRFs;
}

template <typename CodeGenerator>
void InterfaceHandler::generatePrologue(CodeGenerator &generator, const GRF &temp) const
{
#ifdef NGEN_INTERFACE_OLD_PROLOGUE
    if (needLocalID)
        generator.loadlid(getCrossthreadGRFs(), needLocalID, simd, temp, 8*16);
    if (getCrossthreadGRFs() > 1)
        generator.loadargs(getCrossthreadBase(), getCrossthreadGRFs(), temp);
#else
    if (needLocalID)
        generator.loadlid(getCrossthreadGRFs(), needLocalID, simd, temp, 12*16);
    if (getCrossthreadGRFs() > 1)
        generator.loadargs(getCrossthreadBase().advance(1), getCrossthreadGRFs() - 1, temp);
#endif
}

std::string InterfaceHandler::generateZeInfo() const
{
#ifdef NGEN_SAFE
    if (!finalized) throw interface_not_finalized();
#endif

    std::stringstream md;

    md << "version: 1.0\n"
          "kernels: \n"
          "  - name: \"" << kernelName << "\"\n"
          "    execution_env: \n"
          "      grf_count: " << needGRF << "\n"
          "      simd_size: " << simd << "\n";
    if (simd > 1)
        md << "      required_sub_group_size: " << simd << "\n";
    if (wg[0] > 0 && wg[1] > 0 && wg[2] > 0) {
        md << "      required_work_group_size:\n"
           << "        - " << wg[0] << "\n"
           << "        - " << wg[1] << "\n"
           << "        - " << wg[2] << "\n";
    }
    if (walkOrder[0] >= 0) {
        md << "      work_group_walk_order_dimensions:\n"
           << "        - " << walkOrder[0] << "\n"
           << "        - " << walkOrder[1] << "\n"
           << "        - " << std::max(walkOrder[2], 0) << "\n";
    }
    md << "      actual_kernel_start_offset: " << offsetSkipCrossThread << '\n';
    if (offsetSkipPerThread > 0)
        md << "      offset_to_skip_per_thread_data_load: " << offsetSkipPerThread << '\n';
    if (barrierCount > 0)
        md << "      barrier_count: " << barrierCount << '\n';
    if (allow64BitBuffers)
        md << "      has_4gb_buffers: true\n";
    if (needDPAS)
        md << "      has_dpas: true\n";
    if (needGlobalAtomics)
        md << "      has_global_atomics: true\n";
    if (slmSize > 0)
        md << "      slm_size: " << slmSize << '\n';
    if (!needStatelessWrites)
        md << "      has_no_stateless_write: true\n";
    if (needNoPreemption)
        md << "      disable_mid_thread_preemption: true\n";
    md << "\n";
    md << "    payload_arguments: \n";
    for (auto &assignment : assignments) {
        uint32_t size = 0;
        bool skipArg = false;
        bool explicitArg = true;

        if (assignment.globalSurfaceAccess()) {
            md << "      - arg_type: arg_bypointer\n"
                  "        arg_index: " << assignment.index << "\n"
                  "        offset: 0\n"
                  "        size: 0\n"
                  "        addrmode: stateful\n"
                  "        addrspace: global\n"
                  "        access_type: readwrite\n"
                  "\n";
        }

        switch (assignment.exttype) {
            case ExternalArgumentType::Scalar:
                md << "      - arg_type: arg_byvalue\n";
                size = (assignment.reg.getDwords() << 2);
                break;
            case ExternalArgumentType::LocalPtr:
            case ExternalArgumentType::GlobalPtr:
                if (!assignment.globalStatelessAccess())
                    skipArg = true;
                else {
                    md << "      - arg_type: arg_bypointer\n";
                    size = (assignment.reg.getDwords() << 2);
                }
                break;
            case ExternalArgumentType::Hidden: {
                explicitArg = false;
                if (assignment.name == "__local_size0") {
                    // from Zebin spec : local_size Argument size : int32x3
                    // may need refining to allow
                    // either int32x1, int32x2, int32x3 (x, xy, xyz)
                    // or fine grain : local_size_x, local_size_y, local_size_z
                    md << "      - arg_type: "
                       << (needNonuniformWGs ? "enqueued_local_size\n" : "local_size\n");
                    size = (assignment.reg.getDwords() << 2) * 3;
                } else
                    skipArg = true;
                break;
            }
        }
        if (skipArg)
            continue;

        auto offset = (assignment.reg.getBase() - getCrossthreadBase().getBase()) * GRF::bytes(hw) + assignment.reg.getByteOffset();
        if (explicitArg)
            md << "        arg_index: " << assignment.index << "\n";
        md << "        offset: " << offset << "\n"
              "        size: " << size << '\n';

        if (assignment.globalStatelessAccess()) {
            md << "        addrmode: stateless\n"
                  "        addrspace: global\n"
                  "        access_type: readwrite\n";
        } else if (assignment.exttype == ExternalArgumentType::LocalPtr) {
            md << "        addrmode: slm\n"
                  "        addrspace: local\n"
                  "        access_type: readwrite\n";
        }
        md << "\n";
    }

    md << "\n";
    md << "    binding_table_indices: \n";

    for (auto &assignment : assignments) {
        if (assignment.globalSurfaceAccess()) {
            md << "      - bti_value: " << assignment.surface << "\n"
                  "        arg_index: " << assignment.index << "\n"
                  " \n";
        }
    }

    md << "\n";
    md << "    per_thread_payload_arguments: \n";

    if (needLocalID) {
        if (simd == 1) {
            md << "      - arg_type: packed_local_ids\n"
                  "        offset: 0\n"
                  "        size: 6\n"
                  "  \n";
        } else {
            auto localIDBytes = grfsPerLID() * 32;
            localIDBytes *= 3; // runtime currently supports 0 or 3 localId channels in per thread data
            md << "      - arg_type: local_id\n"
                  "        offset: 0\n"
                  "        size: " << localIDBytes << "\n"
                  "  \n";
        }
    }

    md << "\n"; // ensure file ends with newline

#ifdef NGEN_DUMP_ZE_INFO
    std::cerr << md.str();
#endif

    return md.str();
}

#ifdef NGEN_ASM
void InterfaceHandler::dumpAssignments(std::ostream &stream) const
{
    LabelManager manager;

    for (auto &assignment : assignments) {
        stream << "//  ";
        if (assignment.reg.isValid())
            assignment.reg.outputText(stream, PrintDetail::sub, manager);
        else
            stream << "(none)";
        stream << '\t' << assignment.name;
        if (assignment.surface != noSurface)
            stream << "\t(BTI " << assignment.surface << ')';
        stream << std::endl;
    }
}
#endif

} /* namespace ngen */

#endif /* header guard */
