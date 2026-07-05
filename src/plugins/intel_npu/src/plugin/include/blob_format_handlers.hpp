#pragma once

#include <map>
#include <string>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/igraph.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

class IBlobFormatHandler {
public:
    IBlobFormatHandler(const std::shared_ptr<ov::Model>& original_model,
                       const FilteredConfig& config,
                       const Logger& logger);

    virtual std::shared_ptr<ov::Model> create_dummy_model() = 0;

    virtual std::shared_ptr<IGraph> create_graph() = 0;

    virtual ~IBlobFormatHandler() = default;

private:
    virtual ov::Tensor extract_main_schedule() = 0;

    virtual ov::Tensor extract_init_schedules() = 0;

    virtual ov::Tensor decrypt_schedules() = 0;

    virtual ov::Tensor create_weights_map() = 0;

    std::shared_ptr<ov::Model> m_original_model;
    FilteredConfig m_config;
    Logger m_logger;
};

class RawBlobHandler : public IBlobFormatHandler {
public:
    explicit RawBlobHandler(std::istream& compiler_main_schedule,
                            const std::shared_ptr<ov::Model>& original_model,
                            const FilteredConfig& config);

    explicit RawBlobHandler(const ov::Tensor& compiler_main_schedule,
                            const std::shared_ptr<ov::Model>& original_model,
                            const FilteredConfig& config);

private:
    ov::Tensor m_compiler_main_schedule;
};

class BlobFormatV1Handler : public IBlobFormatHandler {
public:
    explicit BlobFormatV1Handler(std::istream& npu_formatted_blob,
                                 const std::shared_ptr<ov::Model>& original_model,
                                 const FilteredConfig& config);

    explicit BlobFormatV1Handler(const ov::Tensor& npu_formatted_blob,
                                 const std::shared_ptr<ov::Model>& original_model,
                                 const FilteredConfig& config);

private:
    ov::Tensor m_compiler_payload;
    std::unique_ptr<MetadataBase> m_metadata;
};

namespace blob_format_handler_factory {

std::shared_ptr<IBlobFormatHandler> create(std::istream& npu_formatted_blob,
                                           const bool raw_blob,
                                           const std::shared_ptr<ov::Model>& original_model,
                                           const FilteredConfig& config);

std::shared_ptr<IBlobFormatHandler> create(const ov::Tensor& npu_formatted_blob,
                                           const bool raw_blob,
                                           const std::shared_ptr<ov::Model>& original_model,
                                           const FilteredConfig& config);

}  // namespace blob_format_handler_factory

}  // namespace intel_npu
