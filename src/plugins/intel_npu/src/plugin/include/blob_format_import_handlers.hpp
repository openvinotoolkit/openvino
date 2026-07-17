#pragma once

#include <map>
#include <string>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/igraph.hpp"
#include "intel_npu/common/npu.hpp"
#include "metadata.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

class ICore;

}

namespace intel_npu {

class IBlobFormatImportHandler {
public:
    IBlobFormatImportHandler(const std::shared_ptr<const ov::Model>& original_model,
                             const FilteredConfig& config,
                             const Logger& logger);

    std::shared_ptr<IGraph> create_graph(const ov::SoPtr<IEngineBackend>& backend,
                                         const std::string_view network_name,
                                         const std::string_view device_name,
                                         const std::shared_ptr<ov::ICore>& core);

    std::shared_ptr<ov::Model> create_dummy_model() const;

    FilteredConfig get_config() const;

    virtual ~IBlobFormatImportHandler() = default;

protected:
    FilteredConfig m_config;
    Logger m_logger;

private:
    virtual void decrypt_schedules() = 0;

    virtual ov::Tensor extract_main_schedule() const = 0;

    virtual std::optional<std::vector<ov::Tensor>> extract_init_schedules() const = 0;

    virtual std::optional<int> extract_batch_size() const = 0;

    virtual std::optional<std::pair<std::vector<ov::Layout>, std::vector<ov::Layout>>> extract_layouts() const = 0;

    virtual std::optional<std::string> extract_compiler_compatibility_descriptor() const = 0;

    void log_contents(const std::optional<std::string>& compatibility_descriptor);

    std::optional<std::shared_ptr<const ov::Model>> m_original_model;

    ov::Tensor m_main_schedule;
    std::optional<std::vector<ov::Tensor>> m_init_schedules;
    std::optional<int> m_batch_size;

    std::shared_ptr<IGraph> m_graph;
};

class RawBlobImportHandler : public IBlobFormatImportHandler {
public:
    explicit RawBlobImportHandler(std::istream& compiler_main_schedule,
                                  const std::shared_ptr<const ov::Model>& original_model,
                                  const FilteredConfig& config);

    explicit RawBlobImportHandler(const ov::Tensor& compiler_main_schedule,
                                  const std::shared_ptr<const ov::Model>& original_model,
                                  const FilteredConfig& config);

private:
    void decrypt_schedules() override;

    ov::Tensor extract_main_schedule() const override;

    std::optional<std::vector<ov::Tensor>> extract_init_schedules() const override;

    std::optional<int> extract_batch_size() const override;

    std::optional<std::pair<std::vector<ov::Layout>, std::vector<ov::Layout>>> extract_layouts() const override;

    std::optional<std::string> extract_compiler_compatibility_descriptor() const override;

    ov::Tensor m_compiler_payload;
};

class BlobFormatV1ImportHandler : public IBlobFormatImportHandler {
public:
    explicit BlobFormatV1ImportHandler(std::istream& npu_formatted_blob,
                                       const std::shared_ptr<const ov::Model>& original_model,
                                       const FilteredConfig& config);

    explicit BlobFormatV1ImportHandler(const ov::Tensor& npu_formatted_blob,
                                       const std::shared_ptr<const ov::Model>& original_model,
                                       const FilteredConfig& config);

private:
    void decrypt_schedules() override;

    ov::Tensor extract_main_schedule() const override;

    std::optional<std::vector<ov::Tensor>> extract_init_schedules() const override;

    std::optional<int> extract_batch_size() const override;

    std::optional<std::pair<std::vector<ov::Layout>, std::vector<ov::Layout>>> extract_layouts() const override;

    std::optional<std::string> extract_compiler_compatibility_descriptor() const override;

    void register_compiler_version();

    ov::Tensor m_compiler_payload;
    std::unique_ptr<MetadataBase> m_metadata;
};

namespace blob_format_import_handler_factory {

std::unique_ptr<IBlobFormatImportHandler> create(std::istream& npu_formatted_blob,
                                                 const bool is_raw_blob,
                                                 const std::shared_ptr<const ov::Model>& original_model,
                                                 const FilteredConfig& config);

std::unique_ptr<IBlobFormatImportHandler> create(const ov::Tensor& npu_formatted_blob,
                                                 const bool is_raw_blob,
                                                 const std::shared_ptr<const ov::Model>& original_model,
                                                 const FilteredConfig& config);

}  // namespace blob_format_import_handler_factory

}  // namespace intel_npu
