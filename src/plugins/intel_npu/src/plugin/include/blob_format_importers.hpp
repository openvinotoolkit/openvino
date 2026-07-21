// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

/**
 * @brief Abstract class used as the base for importing blobs of different formats.
 */
class IBlobFormatImporter {
public:
    /**
     * @param original_model The model used for compiling the target blob. This parameter may be used as a source of
     * weights only if the blob was compiled using the weights separation feature. Can be "nullptr", in which case the
     * "weights path" from the configuration object may be used as the source of weights instead.
     * @param config Used for multiple purposes such as: extracting the log level, weights path, decryption callbacks.
     * @param logger A logger that should already use the name of the subclass.
     */
    IBlobFormatImporter(const std::shared_ptr<const ov::Model>& original_model,
                        const FilteredConfig& config,
                        const Logger& logger);

    /**
     * @brief Parses the current blob to create a graph type of object.
     * @details This function calls multiple virtual methods in order to extract from the current format all information
     * it needs (e.g. the compiler main schedule) to build the graph object.
     *
     * @param backend The current backend in use, required for creating and using a parser.
     * @param network_name The resulted graph object will use this network name.
     * @param device_name The name of the device. Used to update the compiler type if "perf count" was enabled via the
     * configuration.
     * @param core The OV core. The weights separation feature may depend on this object to extract the original
     * weights.
     * @return A graph object. The type of graph depends on the content of the blob. E.g. a "weightless" graph will be
     * returned if the weights separation feature was detected.
     */
    std::shared_ptr<IGraph> create_graph(const ov::SoPtr<IEngineBackend>& backend,
                                         const std::string_view network_name,
                                         const std::string_view device_name,
                                         const std::shared_ptr<ov::ICore>& core);

    /**
     * @brief Uses the metadata of the resulted graph object to build a minimalistic OV model.
     * @note This should be called only after "create_graph".
     * @throws ov::AssertFailure if called before "create_graph" (invalid state).
     *
     * @return A model object containing only "Parameter" and "Result" nodes, according to the metadata stored in the
     * graph object.
     */
    std::shared_ptr<ov::Model> create_dummy_model() const;

    FilteredConfig get_config() const;

    virtual ~IBlobFormatImporter() = default;

protected:
    FilteredConfig m_config;
    Logger m_logger;

private:
    /**
     * @brief Decrypts all schedules (main and inits if applicable) if all conditions are met (e.g. a decryption
     * callback was provided).
     */
    virtual void decrypt_schedules() = 0;

    /**
     * @brief Extracts the compiler main schedule as a tensor.
     */
    virtual ov::Tensor extract_main_schedule() const = 0;

    /**
     * @brief If weights separation was used, the init schedules will be extracted as tensors.
     */
    virtual std::optional<std::vector<ov::Tensor>> extract_init_schedules() const = 0;

    /**
     * @brief If a batch size a stored, this method will be used to extract it.
     */
    virtual std::optional<int64_t> extract_batch_size() const = 0;

    /**
     * @brief If input/output layouts were stored, this method will be used to extract them.
     * @note The layouts are only used as optional information within the "dummy" model.
     */
    virtual std::optional<std::pair<std::vector<ov::Layout>, std::vector<ov::Layout>>> extract_layouts() const = 0;

    /**
     * @brief If a compiler compatibility string was stored, this method will be used to extract it.
     */
    virtual std::optional<std::string> extract_compiler_compatibility_descriptor() const = 0;

    void log_contents(const std::optional<std::string>& compatibility_descriptor);

    /**
     * @brief A potential source of weights for weights separation. Can be `nullptr`.
     */
    std::shared_ptr<const ov::Model> m_original_model;
    std::optional<int64_t> m_batch_size;
    std::shared_ptr<IGraph> m_graph;
};

/**
 * @brief Class used to import a blob that contains only the compiler main schedule.
 */
class RawBlobImporter : public IBlobFormatImporter {
public:
    explicit RawBlobImporter(std::istream& compiler_main_schedule,
                             const std::shared_ptr<const ov::Model>& original_model,
                             const FilteredConfig& config);

    explicit RawBlobImporter(const ov::Tensor& compiler_main_schedule,
                             const std::shared_ptr<const ov::Model>& original_model,
                             const FilteredConfig& config);

private:
    /**
     * @brief Decrypts the compiler main schedule if a decryption callback was received.
     */
    void decrypt_schedules() override;

    ov::Tensor extract_main_schedule() const override;

    /**
     * @note N/A
     * @return Always std::nullopt
     */
    std::optional<std::vector<ov::Tensor>> extract_init_schedules() const override;

    /**
     * @note N/A
     * @return Always std::nullopt
     */
    std::optional<int64_t> extract_batch_size() const override;

    /**
     * @note N/A
     * @return Always std::nullopt
     */
    std::optional<std::pair<std::vector<ov::Layout>, std::vector<ov::Layout>>> extract_layouts() const override;

    /**
     * @note N/A
     * @return Always std::nullopt
     */
    std::optional<std::string> extract_compiler_compatibility_descriptor() const override;

    /**
     * @brief The compiler main schedule, that is also the whole blob received to be imported.
     */
    ov::Tensor m_main_schedule;
};

/**
 * @brief Class used to import a blob that follows the "V1" format: compiler payload + some (non-TLV) metadata
 */
class BlobFormatV1Importer : public IBlobFormatImporter {
public:
    explicit BlobFormatV1Importer(std::istream& npu_formatted_blob,
                                  const std::shared_ptr<const ov::Model>& original_model,
                                  const FilteredConfig& config);

    explicit BlobFormatV1Importer(const ov::Tensor& npu_formatted_blob,
                                  const std::shared_ptr<const ov::Model>& original_model,
                                  const FilteredConfig& config);

private:
    /**
     * @brief Decrypts the whole compiler payload (main schedule + init schedules if applicable) if:
     *   1. A decryption callback was provided and
     *   2. The metadata indicates the blob was encrypted.
     * @throws ov::AssertFailure if the blob was encrypted but no decryption callback was provided.
     */
    void decrypt_schedules() override;

    ov::Tensor extract_main_schedule() const override;

    std::optional<std::vector<ov::Tensor>> extract_init_schedules() const override;

    std::optional<int64_t> extract_batch_size() const override;

    std::optional<std::pair<std::vector<ov::Layout>, std::vector<ov::Layout>>> extract_layouts() const override;

    std::optional<std::string> extract_compiler_compatibility_descriptor() const override;

    /**
     * @brief Registers the compiler version inside the configuration attribute if the version is found within the
     * metadata.
     */
    void register_compiler_version();

    /**
     * @brief The whole compiler payload. Init schedules include if weights separation was used.
     */
    ov::Tensor m_compiler_payload;
    std::unique_ptr<MetadataBase> m_metadata;
};

namespace blob_format_importer_factory {

/**
 * @brief Identifies the blob format used for the given blob and creates the corresponding importer for it.
 *
 * @param npu_formatted_blob The target blob.
 * @param is_raw_blob Flag indicating whether or not the whole blob is just a compiler main schedule.
 * @param original_model A potential source of weights for the weights separation feature if necessary. Can be
 * `nullptr`.
 * @param config Will be held by the newly created importer and used for multiple purposes, such as: extracting the log
 * level, weights path, decryption callbacks.
 * @return An importer object of the type that corresponds to the format of the blob.
 */
std::unique_ptr<IBlobFormatImporter> create(std::istream& npu_formatted_blob,
                                            const bool is_raw_blob,
                                            const std::shared_ptr<const ov::Model>& original_model,
                                            const FilteredConfig& config);

/**
 * @brief Identifies the blob format used for the given blob and creates the corresponding importer for it.
 *
 * @param npu_formatted_blob The target blob.
 * @param is_raw_blob Flag indicating whether or not the whole blob is just a compiler main schedule.
 * @param original_model A potential source of weights for the weights separation feature if necessary. Can be
 * `nullptr`.
 * @param config Will be held by the newly created importer and used for multiple purposes, such as: extracting the log
 * level, weights path, decryption callbacks.
 * @return An importer object of the type that corresponds to the format of the blob.
 */
std::unique_ptr<IBlobFormatImporter> create(const ov::Tensor& npu_formatted_blob,
                                            const bool is_raw_blob,
                                            const std::shared_ptr<const ov::Model>& original_model,
                                            const FilteredConfig& config);

}  // namespace blob_format_importer_factory

}  // namespace intel_npu
