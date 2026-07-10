#pragma once

#include <map>
#include <string>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/igraph.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

class IBlobFormatHandler {
public:
    IBlobFormatHandler(const std::shared_ptr<const ov::Model>& original_model,
                       const FilteredConfig& config,
                       const Logger& logger);

    std::shared_ptr<ov::Model> create_dummy_model() const;

    std::shared_ptr<IGraph> create_graph(const std::shared_ptr<ZeroInitStructsHolder>& zero_init_structs) const;

    virtual ~IBlobFormatHandler() = default;

private:
    virtual ov::Tensor extract_main_schedule() const = 0;

    void decrypt_schedules() = 0;

    virtual std::optional<std::vector<ov::Tensor>> extract_init_schedules() const = 0;

    virtual std::optional<int> extract_batch_size() const = 0;

    virtual std::optional<std::pair<std::vector<ov::Layout>>> extract_layouts() const = 0;

    virtual std::optional<std::string> extract_compiler_compatibility_descriptor() const = 0;

    std::unordered_map<size_t, ov::Constant> create_weights_map() const;

    std::optional<std::shared_ptr<const ov::Model>> m_original_model;
    FilteredConfig m_config;
    Logger m_logger;

    ov::Tensor m_main_schedule;
    std::optional<std::vector<ov::Tensor>> m_init_schedules;
    std::optional<int> m_batch_size;
    std::optional<std::vector<ov::Layout>> m_input_layouts;
    std::optional<std::vector<ov::Layout>> m_output_layouts;

    std::shared_ptr<IGraph> m_graph;
};

class RawBlobHandler : public IBlobFormatHandler {
public:
    explicit RawBlobHandler(std::istream& compiler_main_schedule,
                            const std::shared_ptr<const ov::Model>& original_model,
                            const FilteredConfig& config);

    explicit RawBlobHandler(const ov::Tensor& compiler_main_schedule,
                            const std::shared_ptr<const ov::Model>& original_model,
                            const FilteredConfig& config);

private:
    void decrypt_schedules() override;

    ov::Tensor extract_main_schedule() const override;

    std::optional<std::vector<ov::Tensor>> extract_init_schedules() const override;

    std::optional<int> extract_batch_size() const override;

    std::optional<std::pair<std::vector<ov::Layout>>> extract_layouts() const override;

    std::optional<std::string> extract_compiler_compatibility_descriptor() const override;
};

class BlobFormatV1Handler : public IBlobFormatHandler {
public:
    explicit BlobFormatV1Handler(std::istream& npu_formatted_blob,
                                 const std::shared_ptr<const ov::Model>& original_model,
                                 const FilteredConfig& config);

    explicit BlobFormatV1Handler(const ov::Tensor& npu_formatted_blob,
                                 const std::shared_ptr<const ov::Model>& original_model,
                                 const FilteredConfig& config);

private:
    void decrypt_schedules() override;

    ov::Tensor extract_main_schedule() const override;

    std::optional<std::vector<ov::Tensor>> extract_init_schedules() const override;

    std::optional<int> extract_batch_size() const override;

    std::optional<std::pair<std::vector<ov::Layout>>> extract_layouts() const override;

    std::optional<std::string> extract_compiler_compatibility_descriptor() const override;

    ov::Tensor m_compiler_payload;
    std::unique_ptr<MetadataBase> m_metadata;
};

namespace blob_format_handler_factory {

std::shared_ptr<IBlobFormatHandler> create(std::istream& npu_formatted_blob,
                                           const bool is_raw_blob,
                                           const std::shared_ptr<const ov::Model>& original_model,
                                           const FilteredConfig& config);

std::shared_ptr<IBlobFormatHandler> create(const ov::Tensor& npu_formatted_blob,
                                           const bool is_raw_blob,
                                           const std::shared_ptr<const ov::Model>& original_model,
                                           const FilteredConfig& config);

}  // namespace blob_format_handler_factory

}  // namespace intel_npu
