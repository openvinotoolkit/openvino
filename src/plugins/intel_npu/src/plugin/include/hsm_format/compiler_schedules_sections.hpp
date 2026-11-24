// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "hsm_format/isection.hpp"
#include "intel_npu/common/igraph.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

/**
 * @brief Class handling the writing, parsing and encryption/decryption of a compiler ELF main schedule.
 */
class ELFMainScheduleSection final : public ISection {
public:
    /**
     * @note Intended to be used on the compilation path. The graph object is required to conveniently export the
     * schedule.
     * @note A page alignment value is stored as attribute. This value will be handed over to the HSM writer, and this
     * writer will take care of adding the necessary padding.
     */
    ELFMainScheduleSection(const std::shared_ptr<IGraph>& graph,
                           const std::function<std::string(const std::string&)>& encryption_callback = nullptr);

    /**
     * @note Intended to be used on the import path. A "graph" object is not immediately available after parsing the
     * section. Building a "graph" requires backend components, that are part of a different module.
     * @note A page alignment value is stored as attribute. This value will be handed over to the HSM writer, and this
     * writer will take care of adding the necessary padding.
     */
    ELFMainScheduleSection(ov::Tensor&& main_schedule,
                           const std::function<std::string(const std::string&)>& encryption_callback = nullptr);

    /**
     * @details The plugin importer needs to know how to handle ELF schedules to properly work with the content of this
     * section. Each schedule has individual requirements that need to be interpreted by the compiler, thus the need to
     * add the id token as well.
     * @return The section type and id as CRE tokens
     */
    std::vector<std::shared_ptr<CREToken>> get_cre_tokens(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    /**
     * @brief Exports the schedule being held by the graph, either in plain format or encrypted.
     * @note The compiler schedule is encrypted before writing only if an encryption callback is available.
     */
    void encode(const ov::runtime::hsm::SectionSink& sink) const override;

    /**
     * @brief Stores the given graph. The previously stored schedule (as tensor) will be discarded.
     * @note This method is intended to be called once the blob ownership responsability is transferred to the "graph"
     * object.
     */
    void set_graph(const std::shared_ptr<IGraph>& graph);

    /**
     * @brief Sets the encryption callback to be used once serialization via @ref encode is requested.
     * @details A `nullptr` callback means no encryption will be performed. Otherwise, encryption will be performed
     * using the provided callback.
     */
    void set_encryption_callback(const std::function<std::string(const std::string&)> encryption_callback);

    /**
     * @throws ov::Exception if in its current state the object holds a "graph", not a tensor.
     */
    ov::Tensor get_schedule() const;

    /**
     * @brief Extracts the schedule from the HSM section, and creates a new @ref ELFMainScheduleSection to hold it.
     * @note No decryption is performed at this stage. Decryption should be requested separately using the @ref decrypt
     * call.
     */
    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

    /**
     * @brief Decrypts the ov::Tensor held by this object.
     * @note A check should be done prior to calling this to verify whether or not the schedule was encrypted in the
     * first place.
     */
    void decrypt(const std::function<std::string(const std::string&)>& decryption_callback);

    /**
     * @brief Extracts the compiler requirements for the current schedule
     */
    std::optional<std::string> get_individual_runtime_requirements() const override;

private:
    /**
     * @brief Where the compiler schedule is found. Either the graph or the current class handles ownership.
     */
    std::variant<std::shared_ptr<IGraph>, ov::Tensor> m_graph_or_schedule;
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
    /**
     * @note Intended to be used on the compilation path. The graph object is required to conveniently export the
     * schedules.
     * @note A page alignment value is stored as attribute. This value will be handed over to the HSM writer, and this
     * writer will take care of adding the necessary padding.
     */
    ELFInitSchedulesSection(const std::shared_ptr<IGraph>& weightless_graph,
                            const std::function<std::string(const std::string&)>& encryption_callback = nullptr);

    /**
     * @note Intended to be used on the import path. A "graph" object is not immediately available after parsing the
     * section. Building a "graph" requires backend components, that are part of a different module.
     * @note A page alignment value is stored as attribute. This value will be handed over to the HSM writer, and this
     * writer will take care of adding the necessary padding.
     */
    ELFInitSchedulesSection(std::vector<ov::Tensor>&& init_schedules,
                            const std::function<std::string(const std::string&)>& encryption_callback = nullptr);

    /**
     * @details The plugin importer needs to know how to handle ELF init schedules to properly work with the content of
     * this section. Unlike the main schedule, init schedules do not have individual requirements that need to go
     * through the compiler for validation.
     * @return The section type as a CRE token
     */
    std::vector<std::shared_ptr<CREToken>> get_cre_tokens(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    /**
     * @brief Exports the schedules being held by the graph, either in plain format or encrypted. The following format
     * is used:
     * <schedule 1 content> ... <schedule n content> <schedule 1 size> ... <schedule n size> <number of schedules>
     * @note The compiler schedules are encrypted before writing only if an encryption callback is available.
     */
    void encode(const ov::runtime::hsm::SectionSink& sink) const override;

    /**
     * @brief Stores the given graph. The previously stored schedules (as tensors) will be discarded.
     * @note This method is intended to be called once the blob ownership responsability is transferred to the "graph"
     * object.
     */
    void set_graph(const std::shared_ptr<IGraph>& weightless_graph);

    /**
     * @brief Sets the encryption callback to be used once serialization via @ref encode is requested.
     * @details A `nullptr` callback means no encryption will be performed. Otherwise, all schedules will be encrypted
     * using the provided callback.
     */
    void set_encryption_callback(const std::function<std::string(const std::string&)> encryption_callback);

    /**
     * @throws ov::Exception if in its current state the object holds "graphs", not tensors.
     */
    std::vector<ov::Tensor> get_schedules() const;

    /**
     * @brief Extracts the schedules from the HSM section, and creates a new @ref ELFInitSchedulesSection to hold them.
     * The following format is expected:
     * <schedule 1 content> ... <schedule n content> <schedule 1 size> ... <schedule n size> <number of schedules>
     * @note No decryption is performed at this stage. Decryption should be requested separately using the @ref decrypt
     * call.
     */
    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

    /**
     * @brief Decrypts the tensors held by this object.
     * @note A check should be done prior to calling this to verify whether or not the schedules were encrypted in the
     * first place.
     */
    void decrypt(const std::function<std::string(const std::string&)>& decryption_callback);

private:
    /**
     * @brief Where the compiler schedules are found. Either the graph or the current class handles ownership.
     */
    std::variant<std::shared_ptr<IGraph>, std::vector<ov::Tensor>> m_graph_or_schedules;
    /**
     * @brief Used to encrypt the schedules before writing it if available.
     * @note Encryption will happen only if this attribute is not null.
     */
    std::function<std::string(const std::string&)> m_encryption_callback;

    Logger m_logger;
};

/**
 * @brief Class handling the writing, parsing and encryption/decryption of a compiler dynamic schedule.
 */
class DynamicScheduleSection final : public ISection {
public:
    /**
     * @note Intended to be used on the compilation path. The graph object is required to conveniently export the
     * schedule.
     * @note A page alignment value is stored as attribute. This value will be handed over to the HSM writer, and this
     * writer will take care of adding the necessary padding.
     */
    DynamicScheduleSection(const std::shared_ptr<IGraph>& graph,
                           const std::function<std::string(const std::string&)>& encryption_callback = nullptr);

    /**
     * @note Intended to be used on the import path. A "graph" object is not immediately available after parsing the
     * section. Building a "graph" requires backend components, that are part of a different module.
     * @note A page alignment value is stored as attribute. This value will be handed over to the HSM writer, and this
     * writer will take care of adding the necessary padding.
     */
    DynamicScheduleSection(ov::Tensor&& main_schedule,
                           const std::function<std::string(const std::string&)>& encryption_callback = nullptr);

    /**
     * @details The plugin importer needs to know how to handle dynamic schedules to properly work with the content of
     * this section. Each schedule has individual requirements that need to be interpreted by the compiler, thus the
     * need to add the id token as well.
     * @return The section type and id as CRE tokens
     */
    std::vector<std::shared_ptr<CREToken>> get_cre_tokens(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const override;

    /**
     * @brief Exports the schedule being held by the graph, either in plain format or encrypted.
     * @note The compiler schedule is encrypted before writing only if an encryption callback is available.
     */
    void encode(const ov::runtime::hsm::SectionSink& sink) const override;

    /**
     * @brief Stores the given graph. The previously stored schedule (as tensor) will be discarded.
     * @note This method is intended to be called once the blob ownership responsability is transferred to the "graph"
     * object.
     */
    void set_graph(const std::shared_ptr<IGraph>& graph);

    /**
     * @brief Sets the encryption callback to be used once serialization via @ref encode is requested.
     * @details A `nullptr` callback means no encryption will be performed. Otherwise, encryption will be performed
     * using the provided callback.
     */
    void set_encryption_callback(const std::function<std::string(const std::string&)> encryption_callback);

    /**
     * @throws ov::Exception if in its current state the object holds a "graph", not a tensor.
     */
    ov::Tensor get_schedule() const;

    /**
     * @brief Extracts the schedule from the HSM section, and creates a new @ref DynamicScheduleSection to hold it.
     * @note No decryption is performed at this stage. Decryption should be requested separately using the @ref decrypt
     * call.
     */
    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

    /**
     * @brief Decrypts the ov::Tensor held by this object.
     * @note A check should be done prior to calling this to verify whether or not the schedule was encrypted in the
     * first place.
     */
    void decrypt(const std::function<std::string(const std::string&)>& decryption_callback);

    /**
     * @brief Extracts the compiler requirements for the current schedule
     */
    std::optional<std::string> get_individual_runtime_requirements() const override;

private:
    /**
     * @note The dynamic schedule is handled almost the same as the ELFMainSchedule. Thus this attribute.
     */
    ELFMainScheduleSection m_impl;

    Logger m_logger;
};

}  // namespace intel_npu
