// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "dynamic_graph.hpp"
#include "graph.hpp"
#include "intel_npu/common/isection.hpp"
#include "openvino/runtime/properties.hpp"
#include "weightless_graph.hpp"

namespace intel_npu {

class ELFMainScheduleSection final : public ISection {
public:
    ELFMainScheduleSection(const std::shared_ptr<Graph>& graph,
                           const std::optional<ov::EncryptionCallbacks>& encryption_callbacks = std::nullopt,
                           const ov::log::Level log_level = ov::log::Level::WARNING);

    ELFMainScheduleSection(ov::Tensor&& main_schedule,
                           const std::optional<ov::EncryptionCallbacks>& encryption_callbacks = std::nullopt,
                           const ov::log::Level log_level = ov::log::Level::WARNING);

    std::vector<CREToken> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    /**
     * @note The compiler payload is encrypted before writing if an encryption callback is available.
     */
    void write(BlobWriterInterface& writer) override;

    void set_graph(const std::shared_ptr<Graph>& graph);

    ov::Tensor get_schedule() const;

    static std::shared_ptr<ISection> read(BlobReaderInterface& blob_reader);

    void decrypt(const ov::EncryptionCallbacks& encryption_callbacks);

    std::optional<std::string> get_inidividual_compatibility_requirements() const override;

private:
    std::variant<std::shared_ptr<Graph>, ov::Tensor> m_graph_or_schedule;
    std::optional<ov::EncryptionCallbacks> m_encryption_callbacks;

    Logger m_logger;
};

class ELFInitSchedulesSection final : public ISection {
public:
    ELFInitSchedulesSection(const std::shared_ptr<WeightlessGraph>& weightless_graph,
                            const std::optional<ov::EncryptionCallbacks>& encryption_callbacks = std::nullopt,
                            const ov::log::Level log_level = ov::log::Level::WARNING);

    ELFInitSchedulesSection(std::vector<ov::Tensor>&& init_schedules,
                            const std::optional<ov::EncryptionCallbacks>& encryption_callbacks = std::nullopt,
                            const ov::log::Level log_level = ov::log::Level::WARNING);

    std::vector<CREToken> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    /**
     * @note The compiler payload is encrypted before writing if an encryption callback is available.
     */
    void write(BlobWriterInterface& writer) override;

    void set_graph(const std::shared_ptr<WeightlessGraph>& weightless_graph);

    std::vector<ov::Tensor> get_schedules() const;

    static std::shared_ptr<ISection> read(BlobReaderInterface& blob_reader);

    void decrypt(const ov::EncryptionCallbacks& encryption_callbacks);

private:
    std::variant<std::shared_ptr<WeightlessGraph>, std::vector<ov::Tensor>> m_graph_or_schedules;
    std::optional<ov::EncryptionCallbacks> m_encryption_callbacks;

    Logger m_logger;
};

class DynamicScheduleSection final : public ISection {
public:
    DynamicScheduleSection(const std::shared_ptr<DynamicGraph>& graph,
                           const std::optional<ov::EncryptionCallbacks>& encryption_callbacks = std::nullopt,
                           const ov::log::Level log_level = ov::log::Level::WARNING);

    DynamicScheduleSection(ov::Tensor&& main_schedule,
                           const BlobType blob_type,
                           const std::optional<ov::EncryptionCallbacks>& encryption_callbacks = std::nullopt,
                           const ov::log::Level log_level = ov::log::Level::WARNING);

    std::vector<CREToken> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    /**
     * @note The compiler payload is encrypted before writing if an encryption callback is available.
     */
    void write(BlobWriterInterface& writer) override;

    void set_graph(const std::shared_ptr<DynamicGraph>& graph);

    // TODO consider moving the tensor to free the schedule earlier
    ov::Tensor get_schedule() const;

    BlobType get_blob_type() const;

    static std::shared_ptr<ISection> read(BlobReaderInterface& blob_reader);

    void decrypt(const ov::EncryptionCallbacks& encryption_callbacks);

    std::optional<std::string> get_inidividual_compatibility_requirements() const override;

private:
    ELFMainScheduleSection m_impl;
    BlobType m_blob_type;

    Logger m_logger;
};

}  // namespace intel_npu
