//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "metadata_helper.hpp"

#include <cstdint>
#include <iostream>

namespace {

static constexpr uint8_t MAX_PRINTABLE_VECTOR_ELEMENTS = 6;  // 1, 2, 3, ..., 7, 8, 9

template <typename T>
struct PrintHelper {
    PrintHelper(T& tmp) : _tmp(tmp) {}

    template <typename, typename = void>
    static constexpr bool has_to_string = false;

    friend std::ostream& operator<<(std::ostream& ostream, const PrintHelper<T>& other) {
        if constexpr (has_to_string<T>) {
            ostream << other._tmp.to_string();
        } else {
            ostream << other._tmp;
        }
        return ostream;
    }

    T& _tmp;
};

template <typename T>
template <typename U>
constexpr bool PrintHelper<T>::has_to_string<U, std::void_t<decltype(std::declval<U&>().to_string())>> = true;

template <typename T>
std::ostream& operator<<(std::ostream& ostream, const std::vector<T>& vec) {
    ostream << "(size = " << vec.size() << "): [";
    if (vec.size() <= MAX_PRINTABLE_VECTOR_ELEMENTS) {
        for (auto it = vec.begin(); it != vec.end() - 1; ++it) {
            ostream << PrintHelper(*it) << ", ";
        }
    } else {
        for (auto it = vec.begin(); it != vec.begin() + MAX_PRINTABLE_VECTOR_ELEMENTS / 2; ++it) {
            ostream << PrintHelper(*it) << ", ";
        }
        ostream << "..., ";
        for (auto it = vec.end() - MAX_PRINTABLE_VECTOR_ELEMENTS / 2; it != vec.end() - 1; ++it) {
            ostream << PrintHelper(*it) << ", ";
        }
    }
    ostream << PrintHelper(*(vec.end() - 1)) << "]";
    return ostream;
}

}  // namespace

namespace npu {
namespace utils {

std::pair<uint32_t, std::unique_ptr<intel_npu::MetadataBase>> parseNPUMetadata(std::istream& stream) {
    size_t magicBytesSize = intel_npu::MAGIC_BYTES.size();
    std::string blobMagicBytes;
    blobMagicBytes.resize(magicBytesSize);

    stream.read(blobMagicBytes.data(), magicBytesSize);
    if (intel_npu::MAGIC_BYTES != blobMagicBytes) {
        OPENVINO_THROW("Blob is missing NPU metadata!");
    }

    uint32_t metaVersion;
    stream.read(reinterpret_cast<char*>(&metaVersion), sizeof(metaVersion));

    std::unique_ptr<intel_npu::MetadataBase> storedMeta;
    storedMeta = intel_npu::create_metadata(metaVersion);
    storedMeta->read(stream);

    return {metaVersion, std::move(storedMeta)};
}

void printNPUMetadata(uint32_t version, const intel_npu::MetadataBase* metadataPtr) {
    std::cout << "NPU metadata version ........... " << intel_npu::MetadataBase::get_major(version) << "."
              << intel_npu::MetadataBase::get_minor(version) << std::endl;
    ;

    auto initSizesOpt = metadataPtr->get_init_sizes();
    if (initSizesOpt.has_value()) {
        std::cout << "Init sizes ........... " << initSizesOpt.value() << std::endl;
    }

    auto batchSizeOpt = metadataPtr->get_batch_size();
    if (batchSizeOpt.has_value()) {
        std::cout << "Batch size ........... " << batchSizeOpt.value() << std::endl;
    }

    auto inputLayoutsOpt = metadataPtr->get_input_layouts();
    if (inputLayoutsOpt.has_value()) {
        std::cout << "Input layouts ........... " << inputLayoutsOpt.value() << std::endl;
    }

    auto ouputLayoutsOpt = metadataPtr->get_input_layouts();
    if (ouputLayoutsOpt.has_value()) {
        std::cout << "Output layouts ........... " << ouputLayoutsOpt.value() << std::endl;
    }
}

}  // namespace utils
}  // namespace npu
