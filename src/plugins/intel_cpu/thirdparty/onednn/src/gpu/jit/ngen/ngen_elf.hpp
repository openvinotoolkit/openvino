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

#ifndef NGEN_ELF_HPP
#define NGEN_ELF_HPP

#include "ngen.hpp"
#include "ngen_interface.hpp"

#include "npack/neo_packager.hpp"

namespace ngen {

// ELF binary format generator class.
template <HW hw>
class ELFCodeGenerator : public BinaryCodeGenerator<hw>
{
public:
    inline std::vector<uint8_t> getBinary();
    static inline void getHWInfo(const std::vector<uint8_t> &binary, HW &_hw, int &steppingID);

protected:
    NEOInterfaceHandler interface_{hw};

    void externalName(const std::string &name)                           { interface_.externalName(name); }

    const std::string &getExternalName() const                           { return interface_.getExternalName(); }
    int getSIMD() const                                                  { return interface_.getSIMD(); }
    int getGRFCount() const                                              { return interface_.getGRFCount(); }
    size_t getSLMSize() const                                            { return interface_.getSLMSize(); }

    void require32BitBuffers()                                           { interface_.require32BitBuffers(); }
    void requireBarrier()                                                { interface_.requireBarrier(); }
    void requireDPAS()                                                   { interface_.requireDPAS(); }
    void requireGlobalAtomics()                                          { interface_.requireGlobalAtomics(); }
    void requireGRF(int grfs)                                            { interface_.requireGRF(grfs); }
    void requireLocalID(int dimensions)                                  { interface_.requireLocalID(dimensions); }
    void requireLocalSize()                                              { interface_.requireLocalSize(); }
    void requireNonuniformWGs()                                          { interface_.requireNonuniformWGs(); }
    void requireNoPreemption()                                           { interface_.requireNoPreemption(); }
    void requireScratch(size_t bytes = 1)                                { interface_.requireScratch(bytes); }
    void requireSIMD(int simd_)                                          { interface_.requireSIMD(simd_); }
    void requireSLM(size_t bytes)                                        { interface_.requireSLM(bytes); }
    void requireStatelessWrites(bool req = true)                         { interface_.requireStatelessWrites(req); }
    inline void requireType(DataType type)                               { interface_.requireType(type); }
    template <typename T> void requireType()                             { interface_.requireType<T>(); }
    void requireWalkOrder(int o1, int o2)                                { interface_.requireWalkOrder(o1, o2); }
    void requireWalkOrder(int o1, int o2, int o3)                        { interface_.requireWalkOrder(o1, o2, o3); }
    void requireWorkgroup(size_t x, size_t y = 1, size_t z = 1)          { interface_.requireWorkgroup(x, y, z); }

    void finalizeInterface()                                             { interface_.finalize(); }

    template <typename DT>
    void newArgument(std::string name)                                   { interface_.newArgument<DT>(name); }
    void newArgument(std::string name, DataType type,
                     ExternalArgumentType exttype = ExternalArgumentType::Scalar,
                     GlobalAccessType access = GlobalAccessType::All)
    {
        interface_.newArgument(name, type, exttype, access);
    }
    void newArgument(std::string name, ExternalArgumentType exttype,
                     GlobalAccessType access = GlobalAccessType::All)
    {
        interface_.newArgument(name, exttype, access);
    }

    Subregister getArgument(const std::string &name) const               { return interface_.getArgument(name); }
    Subregister getArgumentIfExists(const std::string &name) const       { return interface_.getArgumentIfExists(name); }
    int getArgumentSurface(const std::string &name) const                { return interface_.getArgumentSurface(name); }
    int getArgumentSurfaceIfExists(const std::string &name) const        { return interface_.getArgumentSurfaceIfExists(name); }
    GRF getLocalID(int dim) const                                        { return interface_.getLocalID(dim); }
    RegData getSIMD1LocalID(int dim) const                               { return interface_.getSIMD1LocalID(dim); }
    Subregister getLocalSize(int dim) const                              { return interface_.getLocalSize(dim); }

    void prologue()                                                      { interface_.generatePrologue(*this); }
    void epilogue(RegData r0_info = RegData())
    {
        if (r0_info.isInvalid()) r0_info = this->r0;
        int GRFCount = interface_.getGRFCount();
        bool hasSLM = (interface_.getSLMSize() > 0);
        BinaryCodeGenerator<hw>::epilogue(GRFCount, hasSLM, r0_info);
    }

private:
    using BinaryCodeGenerator<hw>::labelManager;
    using BinaryCodeGenerator<hw>::rootStream;

    struct ZebinELF {
        enum {
            ELFMagic = 0x464C457F,             // '\x7FELF'
            ELFClass64 = 2,
            ELFLittleEndian = 1,
            ELFVersion1 = 1,
        };
        enum {
            ZebinExec = 0xFF12
        };
        union ZebinFlags {
            uint32_t all;
            struct {
                unsigned genFlags : 8;
                unsigned minHWRevision : 5;
                unsigned validateRevision : 1;
                unsigned disableExtValidation : 1;
                unsigned useGfxCoreFamily : 1;
                unsigned maxHWRevision : 5;
                unsigned generator : 3;
                unsigned reserved : 8;
            } parts;
        };
        struct FileHeader {
            uint32_t magic = ELFMagic;
            uint8_t elfClass = ELFClass64;
            uint8_t endian = ELFLittleEndian;
            uint8_t version = ELFVersion1;
            uint8_t osABI = 0;
            uint64_t pad = 0;
            uint16_t type = ZebinExec;
            uint16_t machine;
            uint32_t version2 = 1;
            uint64_t entrypoint = 0;
            uint64_t programHeaderOff = 0;
            uint64_t sectionTableOff;
            ZebinFlags flags;
            uint16_t size;
            uint16_t programHeaderSize = 0;
            uint16_t programTableEntries = 0;
            uint16_t sectionHeaderSize;
            uint16_t sectionCount;
            uint16_t strTableIndex = 1;
        } fileHeader;
        struct SectionHeader {
            uint32_t name;
            enum Type : uint32_t {
                Null = 0, Program = 1, SymbolTable = 2, StringTable = 3, ZeInfo = 0xFF000011
            } type;
            uint64_t flags = 0;
            uint64_t addr = 0;
            uint64_t offset;
            uint64_t size;
            uint32_t link = 0;
            uint32_t info = 0;
            uint64_t align = 0x10;
            uint64_t entrySize = 0;
        } sectionHeaders[4];
        struct StringTable {
            const char zero = '\0';
            const char snStrTable[10] = ".shstrtab";
            const char snMetadata[9] = ".ze_info";
            const char snText[6] = {'.', 't', 'e', 'x', 't', '.'};
        } stringTable;

        static size_t align(size_t sz) {
            return (sz + 0xF) & ~0xF;
        }

        ZebinELF(size_t szKernelName, size_t szMetadata, size_t szKernel) {
            fileHeader.size = sizeof(fileHeader);
            fileHeader.sectionHeaderSize = sizeof(SectionHeader);
            fileHeader.sectionTableOff = offsetof(ZebinELF, sectionHeaders);
            fileHeader.sectionCount = sizeof(sectionHeaders) / sizeof(SectionHeader);

            fileHeader.flags.all = 0;
            fileHeader.flags.parts.useGfxCoreFamily = 1;
            fileHeader.machine = static_cast<uint16_t>(npack::encodeGfxCoreFamily(hw));
            if (hw == HW::XeHPG) {
                fileHeader.flags.parts.useGfxCoreFamily = 0;
                fileHeader.machine = static_cast<uint16_t>(npack::ProductFamily::DG2);
            }

            sectionHeaders[0].name = 0;
            sectionHeaders[0].type = SectionHeader::Type::Null;
            sectionHeaders[0].offset = 0;
            sectionHeaders[0].size = 0;

            sectionHeaders[1].name = offsetof(StringTable, snStrTable);
            sectionHeaders[1].type = SectionHeader::Type::StringTable;
            sectionHeaders[1].offset = offsetof(ZebinELF, stringTable);
            sectionHeaders[1].size = sizeof(stringTable);

            sectionHeaders[2].name = offsetof(StringTable, snMetadata);
            sectionHeaders[2].type = SectionHeader::Type::ZeInfo;
            sectionHeaders[2].offset = align(sizeof(ZebinELF) + szKernelName);
            sectionHeaders[2].size = szMetadata;

            sectionHeaders[3].name = offsetof(StringTable, snText);
            sectionHeaders[3].type = SectionHeader::Type::Program;
            sectionHeaders[3].offset = sectionHeaders[2].offset + align(szMetadata);
            sectionHeaders[3].size = szKernel;
        }

        static size_t kernelNameOffset() {
            return offsetof(ZebinELF, stringTable.snText) + sizeof(stringTable.snText);
        }

        bool valid() const {
            if (fileHeader.magic != ELFMagic || fileHeader.elfClass != ELFClass64
                    || fileHeader.endian != ELFLittleEndian || fileHeader.sectionHeaderSize != sizeof(SectionHeader)
                    || (fileHeader.version != 0 && fileHeader.version != ELFVersion1)
                    || fileHeader.type != ZebinExec)
                return false;
            auto *base = reinterpret_cast<const uint8_t *>(&fileHeader);
            auto *sheader = reinterpret_cast<const SectionHeader *>(base + fileHeader.sectionTableOff);
            for (int s = 0; s < fileHeader.sectionCount; s++, sheader++)
                if (sheader->type == SectionHeader::Type::ZeInfo)
                    return true;
            return false;
        }
    };
};

#define NGEN_FORWARD_ELF(hw) NGEN_FORWARD(hw) \
template <typename... Targs> void externalName(Targs&&... args) { ngen::ELFCodeGenerator<hw>::externalName(std::forward<Targs>(args)...); } \
const std::string &getExternalName() const { return ngen::ELFCodeGenerator<hw>::getExternalName(); } \
int getSIMD() const { return ngen::ELFCodeGenerator<hw>::getSIMD(); } \
int getGRFCount() const { return ngen::ELFCodeGenerator<hw>::getGRFCount(); } \
size_t getSLMSize() const { return ngen::ELFCodeGenerator<hw>::getSLMSize(); } \
template <typename... Targs> void require32BitBuffers(Targs&&... args) { ngen::ELFCodeGenerator<hw>::require32BitBuffers(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireBarrier(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireBarrier(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireGlobalAtomics(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireGlobalAtomics(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireGRF(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireGRF(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireLocalID(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireLocalID(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireLocalSize(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireLocalSize(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireNonuniformWGs(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireNonuniformWGs(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireNoPreemption(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireNoPreemption(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireScratch(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireScratch(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireSIMD(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireSIMD(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireSLM(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireSLM(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireStatelessWrites(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireStatelessWrites(std::forward<Targs>(args)...); } \
void requireType(ngen::DataType type) { ngen::ELFCodeGenerator<hw>::requireType(type); } \
template <typename DT = void> void requireType() { ngen::BinaryCodeGenerator<hw>::template requireType<DT>(); } \
template <typename... Targs> void requireWalkOrder(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireWalkOrder(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireWorkgroup(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireWorkgroup(std::forward<Targs>(args)...); } \
template <typename... Targs> void finalizeInterface(Targs&&... args) { ngen::ELFCodeGenerator<hw>::finalizeInterface(std::forward<Targs>(args)...); } \
template <typename... Targs> void newArgument(Targs&&... args) { ngen::ELFCodeGenerator<hw>::newArgument(std::forward<Targs>(args)...); } \
template <typename... Targs> ngen::Subregister getArgument(Targs&&... args) { return ngen::ELFCodeGenerator<hw>::getArgument(std::forward<Targs>(args)...); } \
template <typename... Targs> ngen::Subregister getArgumentIfExists(Targs&&... args) { return ngen::ELFCodeGenerator<hw>::getArgumentIfExists(std::forward<Targs>(args)...); } \
template <typename... Targs> int getArgumentSurface(Targs&&... args) { return ngen::ELFCodeGenerator<hw>::getArgumentSurface(std::forward<Targs>(args)...); } \
template <typename... Targs> int getArgumentSurfaceIfExists(Targs&&... args) { return ngen::ELFCodeGenerator<hw>::getArgumentSurfaceIfExists(std::forward<Targs>(args)...); } \
template <typename... Targs> ngen::GRF getLocalID(Targs&&... args) { return ngen::ELFCodeGenerator<hw>::getLocalID(std::forward<Targs>(args)...); } \
template <typename... Targs> ngen::RegData getSIMD1LocalID(Targs&&... args) { return ngen::ELFCodeGenerator<hw>::getSIMD1LocalID(std::forward<Targs>(args)...); } \
template <typename... Targs> ngen::Subregister getLocalSize(Targs&&... args) { return ngen::ELFCodeGenerator<hw>::getLocalSize(std::forward<Targs>(args)...); } \
void epilogue(const ngen::RegData &r0_info = ngen::RegData()) { ngen::ELFCodeGenerator<hw>::epilogue(r0_info); } \
NGEN_FORWARD_ELF_EXTRA \
NGEN_FORWARD_ELF_EXTRA2

#define NGEN_FORWARD_ELF_EXTRA \
template <typename... Targs> void requireDPAS(Targs&&... args) { ngen::ELFCodeGenerator<hw>::requireDPAS(std::forward<Targs>(args)...); } \
void prologue() { ngen::ELFCodeGenerator<hw>::prologue(); }

#define NGEN_FORWARD_ELF_EXTRA2


template <HW hw>
std::vector<uint8_t> ELFCodeGenerator<hw>::getBinary()
{
    using super = BinaryCodeGenerator<hw>;
    std::vector<uint8_t> binary;
    std::string metadata;

    // Get raw kernel, also fixing up labels.
    auto kernel = super::getCode();

    // Locate entrypoints for XeHP+.
    if (hw >= HW::XeHP) {
        auto idPerThread = super::_labelLocalIDsLoaded.getID(labelManager);
        auto idCrossThread = super::_labelArgsLoaded.getID(labelManager);

        if (labelManager.hasTarget(idPerThread))
            interface_.setSkipPerThreadOffset(labelManager.getTarget(idPerThread));
        if (labelManager.hasTarget(idCrossThread))
            interface_.setSkipCrossThreadOffset(labelManager.getTarget(idCrossThread));
    }

    // Generate metadata.
    metadata = interface_.generateZeInfo();

    // Construct ELF.
    size_t szKernelName = interface_.getExternalName().length();
    size_t szELF = ZebinELF::align(sizeof(ZebinELF) + szKernelName);
    size_t szMetadata = ZebinELF::align(metadata.size());
    size_t szKernel = ZebinELF::align(kernel.size());

    binary.resize(szELF + szMetadata + szKernel);

    (void) new(binary.data()) ZebinELF(szKernelName, szMetadata, szKernel);
    std::memcpy(&binary[ZebinELF::kernelNameOffset()], interface_.getExternalName().c_str(), szKernelName);
    std::memcpy(&binary[szELF], metadata.c_str(), metadata.size());
    std::memcpy(&binary[szELF + szMetadata], kernel.data(), kernel.size());

    return binary;
}

template <HW hw>
void ELFCodeGenerator<hw>::getHWInfo(const std::vector<uint8_t> &binary, HW &_hw, int &steppingID)
{
    auto zebinELF = reinterpret_cast<const ZebinELF *>(binary.data());
    if (zebinELF->valid()) {
        if (zebinELF->fileHeader.flags.parts.useGfxCoreFamily)
            _hw = npack::decodeGfxCoreFamily(static_cast<npack::GfxCoreFamily>(zebinELF->fileHeader.machine));
        else
            _hw = npack::decodeProductFamily(static_cast<npack::ProductFamily>(zebinELF->fileHeader.machine));
        // XXX: Report minHWRevision when minHWRevision != maxHWRevision: it's
        // important to reliably detect early steppings to apply potential WAs.
        steppingID = zebinELF->fileHeader.flags.parts.minHWRevision;
    } else {
        npack::getHWInfo(binary, _hw, steppingID);
    }
}

} /* namespace ngen */

#endif /* NGEN_ELF_HPP */
