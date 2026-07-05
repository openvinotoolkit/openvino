// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_format_handlers.hpp"

#include "intel_npu/utils/utils.hpp"
#include "metadata.hpp"
#include "openvino/runtime/allocator.hpp"

namespace {

ov::Tensor allocate_aligned_tensor(const size_t blobSize) {
    ov::Allocator customAllocator{intel_npu::utils::AlignedAllocator{intel_npu::utils::STANDARD_PAGE_SIZE}};
    ov::Tensor tensor(ov::element::u8, ov::Shape{blobSize}, customAllocator);
    if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
        OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
    }

    return tensor;
}

}  // namespace

namespace intel_npu {

IBlobFormatHandler::IBlobFormatHandler(const std::shared_ptr<ov::Model>& original_model,
                                       const FilteredConfig& config,
                                       const Logger& logger)
    : m_original_model(original_model),
      m_config(config),
      m_logger(logger) {}

RawBlobHandler::RawBlobHandler(std::istream& compiler_main_schedule,
                               const std::shared_ptr<ov::Model>& original_model,
                               const FilteredConfig& config)
    : IBlobFormatHandler(original_model, config, Logger("RawBlobHandler", config.get<LOG_LEVEL>())) {
    const size_t blob_size = MetadataBase::getFileSize(compiler_main_schedule);
    OPENVINO_ASSERT(blob_size > 0, "The blob provided for import is empty");

    m_main_schedule = allocate_aligned_tensor(blob_size);
    compiler_main_schedule.read(m_main_schedule.data<char>(), static_cast<std::streamsize>(blob_size));
}

RawBlobHandler::RawBlobHandler(const ov::Tensor& compiler_main_schedule,
                               const std::shared_ptr<ov::Model>& original_model,
                               const FilteredConfig& config)
    : IBlobFormatHandler(original_model, config, Logger("RawBlobHandler", config.get<LOG_LEVEL>())) {
    const size_t blob_size = compiler_main_schedule.get_byte_size();
    OPENVINO_ASSERT(blob_size > 0, "The blob provided for import is empty");

    m_main_schedule = ov::Tensor(compiler_main_schedule, ov::Coordinate{0}, ov::Coordinate{blob_size});
}

ov::Tensor RawBlobHandler::extract_main_schedule() const {
    return m_main_schedule;
}

std::optional<std::vector<ov::Tensor>> RawBlobHandler::extract_init_schedules() const {
    return std::nullopt;
}

std::optional<int> RawBlobHandler::extract_batch_size() const {
    return std::nullopt;
}

std::optional<std::pair<std::vector<ov::Layout>>> RawBlobHandler::extract_layouts() const {
    return std::nullopt;
}

BlobFormatV1Handler::BlobFormatV1Handler(std::istream& npu_formatted_blob,
                                         const std::shared_ptr<ov::Model>& original_model,
                                         const FilteredConfig& config)
    : IBlobFormatHandler(original_model, config, Logger("BlobFormatV1Handler", config.get<LOG_LEVEL>())) {
    // Read only the metadata from the stream and check if the blob is compatible. Load the blob into memory only if
    // it passes the compatibility checks.
    m_metadata = read_metadata_from(npu_formatted_blob);

    const size_t blob_size = m_metadata->get_blob_size();
    OPENVINO_ASSERT(blob_size > 0, "The blob provided for import doesn't have any compiler payload");

    m_compiler_payload = allocate_aligned_tensor(blob_size);
    npu_formatted_blob.read(m_compiler_payload.data<char>(), static_cast<std::streamsize>(blob_size));
}

BlobFormatV1Handler::BlobFormatV1Handler(const ov::Tensor& npu_formatted_blob,
                                         const std::shared_ptr<ov::Model>& original_model,
                                         const FilteredConfig& config)
    : IBlobFormatHandler(original_model, config, Logger("BlobFormatV1Handler", config.get<LOG_LEVEL>())) {
    m_metadata = read_metadata_from(npu_formatted_blob);

    const size_t blob_size = m_metadata->get_blob_size();
    OPENVINO_ASSERT(blob_size > 0, "The blob provided for import doesn't have any compiler payload");

    // ROI tensor to skip the NPU plugin metadata
    m_compiler_payload = ov::Tensor(npu_formatted_blob, ov::Coordinate{0}, ov::Coordinate{blob_size});
}

namespace blob_format_handler_factory {

std::shared_ptr<IBlobFormatHandler> create(std::istream& npu_formatted_blob,
                                           const bool is_raw_blob,
                                           const std::shared_ptr<ov::Model>& original_model,
                                           const FilteredConfig& config) {
    const Logger logger("blob_format_handler_factory", config.get<LOG_LEVEL>());
    if (is_raw_blob) {
        logger.info("Blob compatibility check skipped.");

        return std::make_shared<RawBlobHandler>(npu_formatted_blob, original_model, config);
    }

    // The V1 format is identified by some magic bytes at the end of the input
    size_t magic_bytes_size = MAGIC_BYTES.size();
    std::string blob_magic_bytes;
    blob_magic_bytes.resize(magic_bytes_size);

    std::streampos compiler_payload_beggining = npu_formatted_blob.tellg();
    npu_formatted_blob.seekg(-std::streampos(magic_bytes_size), std::ios::end);
    npu_formatted_blob.read(blob_magic_bytes.data(), magic_bytes_size);
    if (MAGIC_BYTES != blob_magic_bytes) {
        OPENVINO_THROW("The blob is missing the NPU metadata!");
    }

    npu_formatted_blob.seekg(compiler_payload_beggining, std::ios::beg);

    return std::make_shared<BlobFormatV1Handler>(npu_formatted_blob, original_model, config);
}

std::shared_ptr<IBlobFormatHandler> create(const ov::Tensor& npu_formatted_blob,
                                           const bool is_raw_blob,
                                           const std::shared_ptr<ov::Model>& original_model,
                                           const FilteredConfig& config) {
    const Logger logger("blob_format_handler_factory", config.get<LOG_LEVEL>());
    if (is_raw_blob) {
        logger.info("Blob compatibility check skipped.");  // TODO string views

        return std::make_shared<RawBlobHandler>(npu_formatted_blob, original_model, config);
    }

    size_t magic_bytes_size = MAGIC_BYTES.size();
    std::string_view blob_magic_bytes(
        npu_formatted_blob.data<const char>() + npu_formatted_blob.get_byte_size() - magic_bytes_size,
        magic_bytes_size);

    if (MAGIC_BYTES != blob_magic_bytes) {
        OPENVINO_THROW("The blob is missing the NPU metadata!");
    }

    return std::make_shared<BlobFormatV1Handler>(npu_formatted_blob, original_model, config);
}

}  // namespace blob_format_handler_factory

}  // namespace intel_npu
