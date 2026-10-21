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

/**
 * @brief Class handling the writing, parsing and encryption/decryption of a compiler ELF main schedule.
 */
class ELFMainScheduleSection final : public ISection {
public:
    ELFMainScheduleSection(const std::shared_ptr<Graph>& graph,
                           const std::function<std::string(const std::string&)>& encryption_callback = nullptr,
                           const ov::log::Level log_level = ov::log::Level::WARNING);

    ELFMainScheduleSection(ov::Tensor&& main_schedule,
                           const std::function<std::string(const std::string&)>& encryption_callback = nullptr,
                           const ov::log::Level log_level = ov::log::Level::WARNING);

    std::vector<std::shared_ptr<CREToken>> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    /**
     * @note The compiler payload is encrypted before writing if an encryption callback is available.
     */
    void encode(const ov::runtime::hsm::SectionSink& sink) const override;

    /**
     * @brief Stores the given graph. The previously stored graph or schedule (as tensor) will be discarded.
     */
    void set_graph(const std::shared_ptr<Graph>& graph);

    void set_encryption_callback(const std::function<std::string(const std::string&)> encryption_callback);

    ov::Tensor get_schedule() const;

    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

    // TODO can't this happen during `read`, by getting the callbacks from the BlobReader?
    void decrypt(const std::function<std::string(const std::string&)>& decryption_callback);

    std::optional<std::string> get_individual_compatibility_requirements() const override;

private:
    /**
     * @brief Where the compiler schedule is found. Either the graph or the current class handles ownership.
     */
    std::variant<std::shared_ptr<Graph>, ov::Tensor> m_graph_or_schedule;
    /**
     * @brief Used to encrypt the schedule before writing it if available.
     * @note Encryption will happen only if this attribute is not null.
     */
    std::function<std::string(const std::string&)> m_encryption_callback;

    Logger m_logger;
};

/**
 * @brief Class handling the writing, parsing and encryption/decryption of compiler ELF init schedules.
 * @note This section should be deployed only when weights separation is involved.
 */
class ELFInitSchedulesSection final : public ISection {
public:
    ELFInitSchedulesSection(const std::shared_ptr<WeightlessGraph>& weightless_graph,
                            const std::function<std::string(const std::string&)>& encryption_callback = nullptr,
                            const ov::log::Level log_level = ov::log::Level::WARNING);

    ELFInitSchedulesSection(std::vector<ov::Tensor>&& init_schedules,
                            const std::function<std::string(const std::string&)>& encryption_callback = nullptr,
                            const ov::log::Level log_level = ov::log::Level::WARNING);

    std::vector<std::shared_ptr<CREToken>> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    /**
     * @note The compiler payload is encrypted before writing if an encryption callback is available.
     */
    void encode(const ov::runtime::hsm::SectionSink& sink) const override;

    void set_graph(const std::shared_ptr<WeightlessGraph>& weightless_graph);

    void set_encryption_callback(const std::function<std::string(const std::string&)> encryption_callback);

    std::vector<ov::Tensor> get_schedules() const;

    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

    void decrypt(const std::function<std::string(const std::string&)>& decryption_callback);

private:
    /**
     * @brief Where the compiler schedules are found. Either the graph or the current class handles ownership.
     */
    std::variant<std::shared_ptr<WeightlessGraph>, std::vector<ov::Tensor>> m_graph_or_schedules;
    /**
     * @brief Used to encrypt the schedules before writing it if available.
     * @note Encryption will happen only if this attribute is not null.
     */
    std::function<std::string(const std::string&)> m_encryption_callback;

    std::variant<std::shared_ptr<Graph>, ov::Tensor> m_graph_or_schedule;

    Logger m_logger;
};

/**
 * @brief Class handling the writing, parsing and encryption/decryption of a compiler dynamic schedule.
 */
class DynamicScheduleSection final : public ISection {
public:
    DynamicScheduleSection(const std::shared_ptr<DynamicGraph>& graph,
                           const std::function<std::string(const std::string&)>& encryption_callback = nullptr,
                           const ov::log::Level log_level = ov::log::Level::WARNING);

    DynamicScheduleSection(ov::Tensor&& main_schedule,
                           const std::function<std::string(const std::string&)>& encryption_callback = nullptr,
                           const ov::log::Level log_level = ov::log::Level::WARNING);

    std::vector<std::shared_ptr<CREToken>> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    /**
     * @note The compiler payload is encrypted before writing if an encryption callback is available.
     */
    void encode(const ov::runtime::hsm::SectionSink& sink) const override;

    void set_graph(const std::shared_ptr<DynamicGraph>& graph);

    void set_encryption_callback(const std::function<std::string(const std::string&)> encryption_callback);

    // TODO consider moving the tensor to free the schedule earlier
    ov::Tensor get_schedule() const;

    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

    void decrypt(const std::function<std::string(const std::string&)>& decryption_callback);

    std::optional<std::string> get_individual_compatibility_requirements() const override;

private:
    /**
     * @note The dynamic schedule is handled almost the same as the ELFMainSchedule. Thus this attribute.
     */
    ELFMainScheduleSection m_impl;

    Logger m_logger;
};

}  // namespace intel_npu
