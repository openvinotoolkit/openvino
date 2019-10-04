// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/custom_layer.hpp>

#include <climits>

#include <map>
#include <fstream>
#include <streambuf>
#include <tuple>
#include <utility>
#include <memory>
#include <string>
#include <vector>

#if defined(__linux__) || defined (__APPLE__)
# include <dlfcn.h>
#endif

#ifdef _WIN32
# include <windows.h>
#endif

#include <bitset>

#include <description_buffer.hpp>
#include <xml_parse_utils.h>
#include <details/caseless.hpp>

#include <vpu/utils/simple_math.hpp>
#include <vpu/utils/extra.hpp>

namespace vpu {

namespace {

VPU_PACKED(Elf32Ehdr {
    uint8_t  offs1[28];
    uint32_t ePhoff;        // Program header offset
    uint32_t eShoff;        // Section header offset
    uint8_t  offs2[12];
    uint16_t eShnum;        // Number of sections
    uint16_t offs3;
};)

VPU_PACKED(Elf32Section {
    uint32_t shName;
    uint32_t shType;
    uint32_t shFlags;
    uint32_t shAddr;
    uint32_t shOffset;
    uint32_t shSize;
    uint32_t shLink;
    uint32_t shInfo;
    uint32_t shAddralign;
    uint32_t shEntsize;
};)

VPU_PACKED(Elf32Phdr {
    uint32_t pType;       // Identifies program segment type
    uint32_t pOffset;     // Segment file offset
    uint32_t pVaddr;      // Segment virtual address
    uint32_t pPaddr;      // Segment physical address
    uint32_t pFilesz;     // Segment size in file
    uint32_t pMemsz;      // Segment size in memory
    uint32_t pFlags;      // Flags position from ELF standard spec
    uint32_t pAlign;      // Segment alignment, file & memory
};)

VPU_PACKED(Elf32Sym {
    uint32_t stName;
    uint32_t stValue;
    uint32_t stSize;
    uint8_t  stInfo;
    uint8_t  stOther;
    uint16_t stShndx;
};)

VPU_PACKED(KernelHdr {
    uint32_t address;       // Kernel address
    uint32_t flags;         // Should be 0 for now
    uint32_t sectionSize;   // Section size, offset to the next kernel
    uint32_t argOffset;     // offset to arguments
    uint32_t stackSize;     // Size of the stack required for kernel
    uint32_t stackSizeWI;     // Size of the stack required for kernel per WI
};)

VPU_PACKED(KernelArgHdr {
    uint32_t stringOffset;
    uint32_t addressSpace;
    uint32_t typeOffset;
    uint32_t size;
    uint32_t laneSize;
};)

enum Flags {
  CL_Vecz          = 0x01,
  CL_Unrolled      = 0x02,
  CL_Predicated    = 0x04,
  CL_Dma           = 0x08,
  CL_VeczDma       = 0x10
};

std::pair<const Elf32Section*, const Elf32Section*> findSymbolTable(
        const char* ELFData) {
    const uint32_t SYMTAB = 2;  // Link editing symbol table
    const uint32_t STRTAB = 3;  // A string table

    IE_ASSERT(ELFData != nullptr);

    auto ehdr = reinterpret_cast<const Elf32Ehdr*>(ELFData);
    auto shdr = reinterpret_cast<const Elf32Section*>(ELFData + ehdr->eShoff);

    const Elf32Section* strShdr = nullptr;
    const Elf32Section* symShdr = nullptr;
    for (size_t i = 0; i < ehdr->eShnum; i++) {
        if (shdr[i].shType == STRTAB && strShdr == nullptr) {
            strShdr = &shdr[i];
        } else if (shdr[i].shType == SYMTAB && symShdr == nullptr) {
            symShdr = &shdr[i];
        }

        if (symShdr != nullptr && strShdr != nullptr)
            break;
    }
    IE_ASSERT(symShdr != nullptr && strShdr != nullptr);

    return std::make_pair(strShdr, symShdr);
}

uint32_t getKernelEntry(const char* ELFData, const std::string& kernelName) {
    ie::details::CaselessEq<std::string> cmp;

    IE_ASSERT(ELFData != nullptr);

    auto ehdr = reinterpret_cast<const Elf32Ehdr*>(ELFData);
    auto phdr = reinterpret_cast<const Elf32Phdr*>(ELFData + ehdr->ePhoff);

    const Elf32Section* strShdr = nullptr;
    const Elf32Section* symShdr = nullptr;
    std::tie(strShdr, symShdr) = findSymbolTable(ELFData);
    IE_ASSERT(symShdr != nullptr && strShdr != nullptr);

    auto numSymEntries = symShdr->shSize / symShdr->shEntsize;
    auto sym = reinterpret_cast<const Elf32Sym*>(ELFData + symShdr->shOffset);
    auto firstStr = ELFData + strShdr->shOffset;

    for (size_t i = 0; i < numSymEntries; i++) {
        if (cmp(firstStr + sym[i].stName, kernelName)) {
            return sym[i].stValue - phdr->pVaddr;
        }
    }

    VPU_THROW_EXCEPTION << "Cannot find kernel entry point for custom kernel " << kernelName;
}

SmallVector<std::string> deduceKernelParameters(
        const char* ELFData,
        uint32_t kernelAddress) {
    ie::details::CaselessEq<std::string> cmp;

    IE_ASSERT(ELFData != nullptr);

    auto ehdr = reinterpret_cast<const Elf32Ehdr*>(ELFData);
    auto phdr = reinterpret_cast<const Elf32Phdr*>(ELFData + ehdr->ePhoff);
    auto shdr = reinterpret_cast<const Elf32Section*>(ELFData + ehdr->eShoff);

    const Elf32Section* strShdr = nullptr;
    const Elf32Section* symShdr = nullptr;
    std::tie(strShdr, symShdr) = findSymbolTable(ELFData);
    IE_ASSERT(symShdr != nullptr && strShdr != nullptr);

    auto numSymEntries = symShdr->shSize / symShdr->shEntsize;
    auto sym = reinterpret_cast<const Elf32Sym*>(ELFData + symShdr->shOffset);
    auto firstStr = ELFData + strShdr->shOffset;

    const char* kernelArgStrings = nullptr;
    for (size_t i = 0; i < numSymEntries; i++) {
        if (cmp(firstStr + sym[i].stName, "opencl.kernelArgs.strings")) {
            kernelArgStrings = ELFData + shdr[sym[i].stShndx].shOffset;
            break;
        }
    }
    IE_ASSERT(kernelArgStrings != nullptr);

    SmallVector<std::string> parameters;
    for (size_t i = 0; i < numSymEntries; i++) {
        if (cmp(firstStr + sym[i].stName, "opencl.kernelArgs.info")) {
            auto ptr = ELFData + shdr[sym[i].stShndx].shOffset;
            auto numKernels = *reinterpret_cast<const int*>(ptr);

            auto metaOffset = sizeof(int);
            for (int k = 0; k < numKernels; k++) {
                auto kHdr = reinterpret_cast<const KernelHdr*>(ptr + metaOffset);

                if (kHdr->address-phdr->pVaddr == kernelAddress) {
                    auto aHdr = reinterpret_cast<const KernelArgHdr*>(
                        reinterpret_cast<const char*>(&(kHdr->argOffset)) + sizeof(kHdr->argOffset) + kHdr->argOffset);

                    std::bitset<5> optBits(kHdr->flags);
                    auto numArgs = reinterpret_cast<const int*>(kHdr + 1)[optBits.count()*2];

                    for (int n = 0; n < numArgs; n++, aHdr++) {
                        parameters.push_back(kernelArgStrings + aHdr->stringOffset);
                    }

                    break;
                }

                metaOffset += kHdr->sectionSize + sizeof(kHdr->address) + sizeof(kHdr->flags);
            }
        }
    }

    return parameters;
}

std::pair<uint32_t, uint32_t> deduceVectorized(
        const char* ELFData,
        uint32_t kernelAddress) {
    ie::details::CaselessEq<std::string> cmp;

    IE_ASSERT(ELFData != nullptr);

    auto ehdr = reinterpret_cast<const Elf32Ehdr*>(ELFData);
    auto phdr = reinterpret_cast<const Elf32Phdr*>(ELFData + ehdr->ePhoff);
    auto shdr = reinterpret_cast<const Elf32Section*>(ELFData + ehdr->eShoff);

    const Elf32Section* strShdr = nullptr;
    const Elf32Section* symShdr = nullptr;
    std::tie(strShdr, symShdr) = findSymbolTable(ELFData);
    IE_ASSERT(symShdr != nullptr && strShdr != nullptr);

    auto numSymEntries = symShdr->shSize / symShdr->shEntsize;
    auto sym = reinterpret_cast<const Elf32Sym*>(ELFData + symShdr->shOffset);
    auto firstStr = ELFData + strShdr->shOffset;

    const char* kernelArgStrings = nullptr;
    for (size_t i = 0; i < numSymEntries; i++) {
        if (cmp(firstStr + sym[i].stName, "opencl.kernelArgs.strings")) {
            kernelArgStrings = ELFData + shdr[sym[i].stShndx].shOffset;
            break;
        }
    }
    IE_ASSERT(kernelArgStrings != nullptr);

    for (size_t i = 0; i < numSymEntries; i++) {
        if (cmp(firstStr + sym[i].stName, "opencl.kernelArgs.info")) {
            auto ptr = ELFData + shdr[sym[i].stShndx].shOffset;
            auto numKernels = *reinterpret_cast<const int*>(ptr);

            auto metaOffset = sizeof(int);
            for (int k = 0; k < numKernels; k++) {
                auto kHdr = reinterpret_cast<const KernelHdr*>(ptr + metaOffset);

                if (kHdr->address-phdr->pVaddr == kernelAddress && kHdr->flags == 1) {
                    auto vecInfo = reinterpret_cast<const uint32_t*>(kHdr + 1);
                    return std::make_pair(vecInfo[1], vecInfo[0]-phdr->pVaddr);
                }

                metaOffset += kHdr->sectionSize + sizeof(kHdr->address) + sizeof(kHdr->flags);
            }
        }
    }

    return std::make_pair(0, 0);
}

}  // namespace

ie::details::caseless_map<std::string, std::vector<CustomLayer::Ptr>> CustomLayer::loadFromFile(
        const std::string& configFile,
        bool canBeMissed) {
    ie::details::caseless_map<std::string, std::vector<CustomLayer::Ptr>> out;

    pugi::xml_document xmlDoc;
    pugi::xml_parse_result res = xmlDoc.load_file(configFile.c_str());

    if (res.status != pugi::status_ok) {
        if (canBeMissed) {
            // Config file might not exist - like global config, for example.
            return out;
        } else {
            VPU_THROW_EXCEPTION
                << "Failed to load custom layer configuration file " << configFile
                << " : " << res.description()
                << " at offset " << res.offset;
        }
    }

#ifdef _WIN32
    char path[MAX_PATH];
    auto abs_path_ptr = _fullpath(path, configFile.c_str(), MAX_PATH);
#elif defined(__linux__) || defined(__APPLE__)
    char path[PATH_MAX];
    auto abs_path_ptr = realpath(configFile.c_str(), path);
#endif

    if (abs_path_ptr == nullptr) {
        VPU_THROW_EXCEPTION
            << "Failed to load custom layer configuration file " << configFile
            << " : can't get canonicalized absolute path";
    }

    std::string abs_file_name(path);

    // Try extracting directory from config path.
    auto dir_split_pos = abs_file_name.find_last_of("/\\");
    auto colon_pos = abs_file_name.find_first_of(":");
    auto first_slash_pos = abs_file_name.find_first_of("/");

    // If path is absolute.
    std::string dir_path;
    if (dir_split_pos != std::string::npos && (colon_pos != std::string::npos || first_slash_pos == 0)) {
        dir_path = abs_file_name.substr(0, dir_split_pos);
    } else {
        VPU_THROW_EXCEPTION
            << "Failed to load custom layer configuration file " << configFile
            << " : path is not valid";
    }

    for (auto r = xmlDoc.document_element(); r; r = r.next_sibling()) {
        CustomLayer::Ptr layer(new CustomLayer(dir_path));

        layer->loadSingleLayer(r);

        out[layer->_layerName].push_back(layer);
    }

    return out;
}

int CustomLayer::maxShaves() const {
    return _maxShaves;
}

void CustomLayer::setStageNumInputs(int id) {
    _stageNumInputs = id;
}

int CustomLayer::stageNumInputs() const {
    return _stageNumInputs;
}

int CustomLayer::kernelAddress(int idx) const {
    for (const auto& x : _kernelAddress) {
        if ((idx % x.first) == 0) {
            return x.second;
        }
    }

    auto it = _kernelAddress.find(1);
    IE_ASSERT(it != _kernelAddress.end());

    return it->second;
}

void CustomLayer::loadSingleLayer(const pugi::xml_node& node) {
    ie::details::CaselessEq<std::string> cmp;

    std::string nodeName(node.name());
    if (!cmp(nodeName, "CustomLayer")) {
        VPU_THROW_EXCEPTION << "Wrong custom layer XML : Node is not CustomLayer, but " << nodeName;
    }

    auto nodeType = XMLParseUtils::GetStrAttr(node, "type", "");
    if (!cmp(nodeType, "MVCL")) {
        VPU_THROW_EXCEPTION << "Wrong custom layer XML : Type is not MVCL, but " << nodeType;
    }

    auto version = XMLParseUtils::GetIntAttr(node, "version", -1);
    IE_ASSERT(version == 1);

    auto layerStage = XMLParseUtils::GetStrAttr(node, "stage", "");
    auto layerName = XMLParseUtils::GetStrAttr(node, "name", "");
    if (layerName.empty()) {
        VPU_THROW_EXCEPTION << "Missing Layer name in CustomLayer";
    }
    _layerName = layerStage.empty() ? layerName : layerName + "@stage_" + layerStage;

    _maxShaves = XMLParseUtils::GetIntAttr(node, "max-shaves", 0);

    processWhere(node.child("Where"));

    processKernelNode(node.child("Kernel"));

    processParametersNode(node.child("Parameters"));

    processWorkSizesNode(node.child("WorkSizes"));
}

void CustomLayer::processWhere(const pugi::xml_node& node) {
    for (auto child : node.attributes()) {
        _whereParams[child.name()] = child.value();
    }
}

void CustomLayer::processKernelNode(const pugi::xml_node& node) {
    ie::details::CaselessEq<std::string> cmp;

    std::string nodeName(node.name());
    if (!cmp(nodeName, "Kernel")) {
        VPU_THROW_EXCEPTION << "Wrong node, expected Kernel found " << nodeName;
    }

    if (!_kernelBinary.empty()) {
        VPU_THROW_EXCEPTION << "Multiple definition of Kernel";
    }

    _kernelEntry = XMLParseUtils::GetStrAttr(node, "entry", "");
    if (_kernelEntry.empty()) {
        VPU_THROW_EXCEPTION << "No Kernel entry in custom layer";
    }

    _kernelBinary.clear();
    for (auto sourceNode = node.child("Source"); !sourceNode.empty(); sourceNode = sourceNode.next_sibling("Source")) {
        auto fileName = _configDir + "/" + XMLParseUtils::GetStrAttr(sourceNode, "filename", "");

        std::ifstream inputFile(fileName, std::ios::binary);
        if (!inputFile.is_open()) {
            VPU_THROW_EXCEPTION << "Couldn't open kernel file " << fileName;
        }

        std::ostringstream contentStream;
        contentStream << inputFile.rdbuf();
        _kernelBinary.append(contentStream.str());

        if (_kernelBinary.size() >= 32*1024) {
            VPU_THROW_EXCEPTION << "Kernel binary exceeds 32KB." << fileName;
        }
    }

    _kernelAddress[1] = getKernelEntry(&_kernelBinary[0], _kernelEntry);
    _parameters = deduceKernelParameters(&_kernelBinary[0], _kernelAddress[1]);

    auto vecInfo = deduceVectorized(&_kernelBinary[0], _kernelAddress[1]);
    if (vecInfo.first != 0) {
        _kernelAddress[vecInfo.first] = vecInfo.second;
    }
}

void CustomLayer::processParametersNode(const pugi::xml_node& node) {
    ie::details::CaselessEq<std::string> cmp;

    std::string nodeName(node.name());
    if (!cmp(nodeName, "Parameters")) {
        VPU_THROW_EXCEPTION << "Wrong node, expected Parameters found " << nodeName;
    }

    for (auto tensorNode = node.child("Tensor"); !tensorNode.empty(); tensorNode = tensorNode.next_sibling("Tensor")) {
        KernelParam kp;

        auto typeStr = XMLParseUtils::GetStrAttr(tensorNode, "type");
        if (cmp(typeStr, "input")) {
            kp.type = CustomParamType::Input;
        } else if (cmp(typeStr, "output")) {
            kp.type = CustomParamType::Output;
        } else if (cmp(typeStr, "input_buffer")) {
            kp.type = CustomParamType::InputBuffer;
        } else if (cmp(typeStr, "output_buffer")) {
            kp.type = CustomParamType::OutputBuffer;
        } else if (cmp(typeStr, "data")) {
            kp.type = CustomParamType::Data;
        } else {
            VPU_THROW_EXCEPTION << "Tensor node has an invalid type " << typeStr;
        }

        kp.format = formatFromString(XMLParseUtils::GetStrAttr(tensorNode, "format", "BFYX"));
        if (kp.format == CustomDataFormat::None) {
            VPU_THROW_EXCEPTION << "Tensor node has an invalid format " << kp.format;
        }

        kp.argName = XMLParseUtils::GetStrAttr(tensorNode, "arg-name");
        if (kp.argName.empty()) {
            VPU_THROW_EXCEPTION << "Tensor node has no arg-name";
        }

        kp.portIndex = XMLParseUtils::GetIntAttr(tensorNode, "port-index", -1);
        if (kp.portIndex == -1) {
            VPU_THROW_EXCEPTION << "Tensor node has no port-index";
        }

        if (kp.type == CustomParamType::InputBuffer || kp.type == CustomParamType::OutputBuffer) {
            std::string bufferSize(XMLParseUtils::GetStrAttr(tensorNode, "size", ""));
            while (!bufferSize.empty()) {
                auto pos = bufferSize.find_first_of(',');
                auto rule = bufferSize.substr(0, pos);
                if (!isLegalSizeRule(rule)) {
                    VPU_THROW_EXCEPTION << "Invalid BufferSize " << rule;
                }

                kp.bufferSizeRules.emplace_back(std::move(rule));

                if (pos == std::string::npos) {
                    bufferSize.clear();
                } else {
                    bufferSize = bufferSize.substr(pos + 1, std::string::npos);
                }
            }

            kp.dimIdx = -1;
            std::string dim_src_string(XMLParseUtils::GetStrAttr(tensorNode, "dim", ""));
            if (!dim_src_string.empty()) {
                // Try to locate index separator.
                auto pos = dim_src_string.find_first_of(',');
                auto flag = dim_src_string.substr(0, pos);
                if (cmp(flag, "input")) {
                    kp.dimSource = CustomDimSource::Input;
                } else if (cmp(flag, "output")) {
                    kp.dimSource = CustomDimSource::Output;
                } else {
                    VPU_THROW_EXCEPTION << "Invalid WG dim source " << flag;
                }

                int idx = 0;
                if (pos != std::string::npos) {
                    // User explicitly set input index in config.
                    auto input_idx_string = dim_src_string.substr(pos + 1, std::string::npos);
                    idx = std::stoi(input_idx_string);
                }
                if (idx < 0) {
                    VPU_THROW_EXCEPTION << "Invalid tensor index " << idx;
                }

                kp.dimIdx = idx;
            }
        }

        kp.irSource.clear();

        _kernelParams.emplace_back(std::move(kp));
    }

    for (auto dataNode = node.child("Data"); !dataNode.empty(); dataNode = dataNode.next_sibling("Data")) {
        KernelParam kp;

        kp.type = CustomParamType::Data;
        kp.format = CustomDataFormat::Any;

        kp.argName = XMLParseUtils::GetStrAttr(dataNode, "arg-name");
        if (kp.argName.empty()) {
            VPU_THROW_EXCEPTION << "Data node has no arg-name";
        }

        kp.portIndex = -1;

        kp.irSource = XMLParseUtils::GetStrAttr(dataNode, "source", "");
        if (kp.irSource.empty()) {
            VPU_THROW_EXCEPTION << "Data node has no source";
        }

        _kernelParams.emplace_back(std::move(kp));
    }

    for (auto scalarNode = node.child("Scalar"); !scalarNode.empty(); scalarNode = scalarNode.next_sibling("Scalar")) {
        KernelParam kp;

        std::string typeStr = XMLParseUtils::GetStrAttr(scalarNode, "type");
        if (cmp(typeStr, "int")) {
            kp.type = CustomParamType::Int;
        } else if (cmp(typeStr, "float")) {
            kp.type = CustomParamType::Float;
        } else {
            VPU_THROW_EXCEPTION << "Scalar node has an invalid type " << typeStr;
        }

        kp.format = CustomDataFormat::Any;

        kp.argName = XMLParseUtils::GetStrAttr(scalarNode, "arg-name");
        if (kp.argName.empty()) {
            VPU_THROW_EXCEPTION << "Scalar node has no arg-name";
        }

        kp.portIndex = XMLParseUtils::GetIntAttr(scalarNode, "port-index", 0);

        kp.irSource = XMLParseUtils::GetStrAttr(scalarNode, "source", "");
        if (kp.irSource.empty()) {
            VPU_THROW_EXCEPTION << "Scalar node has no source";
        }

        _kernelParams.emplace_back(std::move(kp));
    }
}

void CustomLayer::processWorkSizesNode(const pugi::xml_node & node) {
    ie::details::CaselessEq<std::string> cmp;

    std::string nodeName(node.name());
    if (!cmp(node.name(), "WorkSizes")) {
        VPU_THROW_EXCEPTION << "Wrong node, expected WorkSizes found " << nodeName;
    }

    _wgDimIdx = -1;
    std::string dim_src_string(node.attribute("dim").as_string(""));
    if (!dim_src_string.empty()) {
        // Try to locate index separator.
        auto pos = dim_src_string.find_first_of(',');
        auto flag = dim_src_string.substr(0, pos);
        if (cmp(flag, "input")) {
            _wgDimSource = CustomDimSource::Input;
        } else if (cmp(flag, "output")) {
            _wgDimSource = CustomDimSource::Output;
        } else {
            VPU_THROW_EXCEPTION << "Invalid WG dim source " << flag;
        }

        int idx = 0;
        if (pos != std::string::npos) {
            // User explicitly set input index in config.
            auto input_idx_string = dim_src_string.substr(pos + 1, std::string::npos);
            idx = std::stoi(input_idx_string);
        }
        if (idx < 0) {
            VPU_THROW_EXCEPTION << "Invalid tensor index " << idx;
        }

        _wgDimIdx = idx;
    }

    std::string gws(node.attribute("global").as_string(""));
    while (!gws.empty()) {
        auto pos = gws.find_first_of(',');
        auto rule = gws.substr(0, pos);
        if (!isLegalSizeRule(rule)) {
            VPU_THROW_EXCEPTION << "Invalid WorkSize " << rule;
        }

        _globalSizeRules.emplace_back(std::move(rule));

        if (pos == std::string::npos) {
            gws.clear();
        } else {
            gws = gws.substr(pos + 1, std::string::npos);
        }
    }

    std::string lws(node.attribute("local").as_string(""));
    while (!lws.empty()) {
        auto pos = lws.find_first_of(',');
        auto rule = lws.substr(0, pos);
        if (!isLegalSizeRule(rule)) {
            VPU_THROW_EXCEPTION << "Invalid WorkSize " << rule;
        }

        _localSizeRules.emplace_back(std::move(rule));

        if (pos == std::string::npos) {
            lws.clear();
        } else {
            lws = lws.substr(pos + 1, std::string::npos);
        }
    }
}

bool CustomLayer::isLegalSizeRule(const std::string& rule) {
    SimpleMathExpression expr;

    expr.setVariables({
        { 'b', 1 }, { 'B', 1 },
        { 'f', 1 }, { 'F', 1 },
        { 'y', 1 }, { 'Y', 1 },
        { 'x', 1 }, { 'X', 1 },
    });

    try {
        expr.parse(rule);
    } catch (...) {
        return false;
    }

    return true;
}

CustomDataFormat CustomLayer::formatFromString(const std::string & str) {
    static const ie::details::caseless_map<std::string, CustomDataFormat> FormatNameToType = {
        { "BFYX" , CustomDataFormat::BFYX },
        { "BYXF" , CustomDataFormat::BYXF },
        { "FYX" , CustomDataFormat::FYX },
        { "YXF" , CustomDataFormat::YXF },
        { "ANY"  , CustomDataFormat::Any },
    };

    auto it = FormatNameToType.find(str);
    if (it != FormatNameToType.end()) {
        return it->second;
    }

    return CustomDataFormat::None;
}

}  // namespace vpu
