// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/igraph.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"

namespace intel_npu {

class ICompilerAdapter {
public:
    virtual std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model,
                                            const FilteredConfig& config) const = 0;

    /**
     * @brief Compiles the model, weights separation enabled.
     * @details The result of compilation will be a binary object that does not contain a significant portion of
     * weights. The binary object will include two types of schedules: weights initialization and the operations of the
     * main graph. In order to run inference on this weightless blob, the original weights will need to be provided as
     * inputs to the weights initialization schedule. Running this will output the processed weights, that can then be
     * fed to the main schedule and therefore enable it to run predictions.
     *
     * @param model The model that will be compiled.
     * @param config Will be passed to the compiler. Additionally, the "SEPARATE_WEIGHTS_VERSION" option will determine
     * which weights separation implementation will be used. See the weights separation specific methods within
     * "icompiler.hpp".
     * @return A "WeightlessGraph" type of object.
     */
    virtual std::shared_ptr<IGraph> compileWS(const std::shared_ptr<ov::Model>& model,
                                              const FilteredConfig& config) const = 0;

    /**
     * @brief Parses the provided binary objects and returns a wrapper over the resulted L0 handles. The model may also
     * be loaded on the device depending on the provided configuration (if "NPU_DEFER_WEIGHTS_LOAD" is off).
     *
     * @param mainBlob The core binary object to be parsed and used to build the "Graph" object.
     * @param config Used to influence the downstream flow of the implementation based on preferences.
     * @param initBlobs Optional. If provided, the "weights separation" flow is enabled and the binary objects
     * corresponding to the init schedules will be parsed as well.
     * @param model Optional, but required if "initBlobs" is provided. The "ov::Model" object is leveraged in the
     * "weights separation" implementation in order to extract the buffers of the weights.
     * @return A wrapper over the corresponding L0 graph handles (multiple only if "initBlobs" has been provided). This
     * wrapper further details the compiled model and brings it in a state closer to execution.
     */
    virtual std::shared_ptr<IGraph> parse(
        ov::Tensor mainBlob,
        const FilteredConfig& config,
        std::optional<std::vector<ov::Tensor>> initBlobs = std::nullopt,
        const std::optional<std::shared_ptr<const ov::Model>>& model = std::nullopt) const = 0;

    virtual ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model,
                                      const FilteredConfig& config) const = 0;
    virtual uint32_t get_version() const = 0;
    virtual std::vector<std::string> get_supported_options() const = 0;
    virtual bool is_option_supported(std::string optname) const = 0;

    /**
     * @brief Stores the information within the "WeightlessCacheAttribute" as runtime fields that persist upon
     * serialization.
     * @details Constant nodes (weights) may contain as medatadata the "WeightlessCacheAttribute", that is information
     * regarding the offset of the weights within the binary file, as well as the original size and precision. This
     * information is required within the "weights separation" flow, therefore this function is here to store it.
     * @note Not calling this function in the weights separation flow would lead to this information being lost upon
     * serialization. The "WeightlessCacheAttribute" information that is populated upon de-serialization would represent
     * metadata corresponding to the serialized stream, not the original weights file. Therefore the compiler would be
     * misinformed and lookups of weights offsets could fail.
     *
     * @param model Both source and target.
     */
    void storeWeightlessCacheAttribute(const std::shared_ptr<ov::Model>& model) const {
        size_t constantId = 0;
        for (auto&& node : model->get_ordered_ops()) {
            if (ov::is_type<ov::op::v0::Constant>(node)) {
                ov::RTMap& runtimeInfoMap = node->get_rt_info();
                const auto& weightlessCacheAttrIt =
                    runtimeInfoMap.find(ov::WeightlessCacheAttribute::get_type_info_static());

                const std::string constantIdString = std::to_string(constantId++);
                if (weightlessCacheAttrIt != runtimeInfoMap.end()) {
                    auto& weightlessCacheAttr = weightlessCacheAttrIt->second.as<ov::WeightlessCacheAttribute>();
                    model->set_rt_info(weightlessCacheAttr.bin_offset, "ws_bin_offset_" + constantIdString);
                    model->set_rt_info(weightlessCacheAttr.original_size, "ws_original_size_" + constantIdString);
                    model->set_rt_info(weightlessCacheAttr.original_dtype, "ws_original_dtype_" + constantIdString);
                }
            }
        }
    }

    virtual ~ICompilerAdapter() = default;
};

}  // namespace intel_npu
