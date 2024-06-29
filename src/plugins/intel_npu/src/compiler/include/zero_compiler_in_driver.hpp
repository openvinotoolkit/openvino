// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <type_traits>
#include <utility>

#include "intel_npu/al/icompiler.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"

namespace intel_npu {
namespace driverCompilerAdapter {

using SerializedIR = std::pair<size_t, std::shared_ptr<uint8_t>>;

#define NotSupportLogHandle(T) \
    (std::is_same<T, ze_graph_dditable_ext_1_2_t>::value || std::is_same<T, ze_graph_dditable_ext_1_3_t>::value)

#define NotSupportQuery(T) (std::is_same<T, ze_graph_dditable_ext_1_2_t>::value)

// ext version == 1.3 && 1.4, support API (pfnQueryNetworkCreate, pfnQueryNetworkDestroy,
// pfnQueryNetworkGetSupportedLayers)
#define SupportAPIGraphQueryNetworkV1(T) \
    (std::is_same<T, ze_graph_dditable_ext_1_3_t>::value || std::is_same<T, ze_graph_dditable_ext_1_4_t>::value)

// ext version >= 1.5, support API (pfnCreate2, pfnQueryNetworkCreate2, pfnQueryContextMemory)
#define SupportAPIGraphQueryNetworkV2(T) ((!NotSupportQuery(T) && !SupportAPIGraphQueryNetworkV1(T)))

// For ext version >= 1.5, pfnCreate2 api is avaible
#define NotSupportGraph2(T)                                                                                        \
    (std::is_same<T, ze_graph_dditable_ext_1_2_t>::value || std::is_same<T, ze_graph_dditable_ext_1_3_t>::value || \
     std::is_same<T, ze_graph_dditable_ext_1_4_t>::value)

// For ext version >= 1.6, originalShape is avaible
#define NotSupportOriginalShape(T)                                                                                 \
    (std::is_same<T, ze_graph_dditable_ext_1_2_t>::value || std::is_same<T, ze_graph_dditable_ext_1_3_t>::value || \
     std::is_same<T, ze_graph_dditable_ext_1_4_t>::value || std::is_same<T, ze_graph_dditable_ext_1_5_t>::value)

/**
 * Adapter to use CiD through ZeroAPI
 */
template <typename TableExtension>
class LevelZeroCompilerInDriver final : public ICompiler {
public:
    LevelZeroCompilerInDriver(const char* extName, ze_driver_handle_t driverHandle);
    LevelZeroCompilerInDriver(const LevelZeroCompilerInDriver&) = delete;
    LevelZeroCompilerInDriver& operator=(const LevelZeroCompilerInDriver&) = delete;
    ~LevelZeroCompilerInDriver() override;

    uint32_t getSupportedOpsetVersion() const override final;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    NetworkDescription compile(const std::shared_ptr<const ov::Model>& model,
                               const Config& config) const override final;

    NetworkMetadata parse(const void* mmapBlob, size_t mmapSize, const Config& config) const;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network,
                                                            const Config& config) const override final {
        OPENVINO_THROW("Profiling post-processing is not implemented.");
    }

    template <typename T = TableExtension, std::enable_if_t<!NotSupportQuery(T), bool> = true>
    std::unordered_set<std::string> getQueryResultFromSupportedLayers(
        ze_result_t result,
        ze_graph_query_network_handle_t& hGraphQueryNetwork) const;

    /**
     * @brief Serialize input / output information to string format.
     * @details Format:
     * --inputs_precisions="<input1Name>:<input1Precision> [<input2Name>:<input2Precision>]"
     * --inputs_layouts="<input1Name>:<input1Layout> [<input2Name>:<input2Layout>]"
     * --outputs_precisions="<output1Name>:<output1Precision>"
     * --outputs_layouts="<output1Name>:<output1Layout>"
     *
     * Since the layout information is no longer an important part of the metadata values when using the 2.0 OV
     * API, the layout fields shall be filled with default values in order to assure the backward compatibility
     * with the driver.
     */
    static std::string serializeIOInfo(const std::shared_ptr<const ov::Model>& model);

private:
    NetworkMetadata getNetworkMeta(ze_graph_handle_t graphHandle) const;

    SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                             ze_graph_compiler_version_info_t compilerVersion) const;
    std::string serializeConfig(const Config& config, ze_graph_compiler_version_info_t& compilerVersion) const;

    /**
     * @brief Extracts the layout value or the state descriptor from the given Level Zero structure.
     * @details Extracting the layout information is required only when using older driver versions which rely on
     * this legacy attribute. Since this information is not found within the parameter/result nodes, we need to
     * extract this value here.
     *
     * The state variables are also not found in the previously mentioned nodes, thus if the given Level Zero
     * parameter corresponds to an input/output, we shall extract the layout value from it. Else it represents a
     * state variable and the descriptor will be extracted and stored in an OpenVINO specific format.
     * @param parameters Holds the already extracted input node descriptors. The transposed shape attribute of the
     * corresponding entry may be updated according to the extracted layout value.
     * @param results Holds the already extracted output node descriptors. The transposed shape attribute of the
     * corresponding entry may be updated according to the extracted layout value.
     * @param states The state descriptors shall be stored here in an OpenVINO specific format.
     * @param stateNames The output location of the state variables' names in the order found within the compiled
     * model.
     * @param arg The Level Zero specific structure from which the layout value or state variable descriptor shall
     * be extracted.
     */
    template <typename T>
    void getLayoutOrStateDescriptor(IONodeDescriptorMap& parameters,
                                    IONodeDescriptorMap& results,
                                    IONodeDescriptorMap& states,
                                    std::vector<std::string>& stateNames,
                                    const T& arg) const;

    template <typename T = TableExtension, typename std::enable_if_t<NotSupportOriginalShape(T), bool> = true>
    void getMetadata(TableExtension* graphDdiTableExt,
                     ze_graph_handle_t graphHandle,
                     uint32_t index,
                     std::vector<std::string>& inputNames,
                     std::vector<std::string>& outputNames,
                     std::vector<std::string>& stateNames,
                     IONodeDescriptorMap& parameters,
                     IONodeDescriptorMap& results,
                     IONodeDescriptorMap& state) const;

    template <typename T = TableExtension, typename std::enable_if_t<!NotSupportOriginalShape(T), bool> = true>
    void getMetadata(TableExtension* graphDdiTableExt,
                     ze_graph_handle_t graphHandle,
                     uint32_t index,
                     std::vector<std::string>& inputNames,
                     std::vector<std::string>& outputNames,
                     std::vector<std::string>& stateNames,
                     IONodeDescriptorMap& parameters,
                     IONodeDescriptorMap& results,
                     IONodeDescriptorMap& state) const;

    // ext version >= 1.5, support API (pfnCreate2, pfnQueryNetworkCreate2, pfnQueryContextMemory)
    template <typename T = TableExtension, typename std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool> = true>
    std::unordered_set<std::string> queryImpl(const std::shared_ptr<const ov::Model>& model,
                                              const Config& config) const;

    // ext version == 1.3 && 1.4, support API (pfnQueryNetworkCreate, pfnQueryNetworkDestroy,
    // pfnQueryNetworkGetSupportedLayers)
    template <typename T = TableExtension, typename std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool> = true>
    std::unordered_set<std::string> queryImpl(const std::shared_ptr<const ov::Model>& model,
                                              const Config& config) const;

    // For ext version < 1.3
    template <typename T = TableExtension, typename std::enable_if_t<NotSupportQuery(T), bool> = true>
    std::unordered_set<std::string> queryImpl(const std::shared_ptr<const ov::Model>& model,
                                              const Config& config) const;

    template <typename T = TableExtension, typename std::enable_if_t<NotSupportGraph2(T), bool> = true>
    ze_result_t createGraph(const ze_graph_format_t& format,
                            const SerializedIR& serializedIR,
                            const std::string& buildFlags,
                            const uint32_t& flags,
                            ze_graph_handle_t* graph) const;

    template <typename T = TableExtension, typename std::enable_if_t<!NotSupportGraph2(T), bool> = true>
    ze_result_t createGraph(const ze_graph_format_t& format,
                            const SerializedIR& serializedIR,
                            const std::string& buildFlags,
                            const uint32_t& flags,
                            ze_graph_handle_t* graph) const;

    template <typename T = TableExtension, typename std::enable_if_t<!NotSupportLogHandle(T), bool> = true>
    std::string getLatestBuildError() const;

    template <typename T = TableExtension, typename std::enable_if_t<NotSupportLogHandle(T), bool> = true>
    std::string getLatestBuildError() const {
        return "";
    }

private:
    ze_driver_handle_t _driverHandle = nullptr;
    ze_device_handle_t _deviceHandle = nullptr;
    ze_context_handle_t _context = nullptr;

    TableExtension* _graphDdiTableExt = nullptr;
    mutable Logger _logger;
};

template <typename TableExtension>
LevelZeroCompilerInDriver<TableExtension>::LevelZeroCompilerInDriver(const char* extName,
                                                                     ze_driver_handle_t driverHandle)
    : _driverHandle(driverHandle),
      _logger("LevelZeroCompilerInDriver", Logger::global().level()) {
    // Load our graph extension
    auto result =
        zeDriverGetExtensionFunctionAddress(_driverHandle, extName, reinterpret_cast<void**>(&_graphDdiTableExt));

    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("Failed to initialize zeDriver. Error code: ", std::hex, result);
    }

    uint32_t deviceCount = 1;
    // Get our target device
    result = zeDeviceGet(_driverHandle, &deviceCount, &_deviceHandle);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("Failed to get device. Error code: ", std::hex, result);
    }

    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    result = zeContextCreate(_driverHandle, &contextDesc, &_context);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("Failed to initialize context for device. Error code: ", std::hex, result);
    }
}

}  // namespace driverCompilerAdapter
}  // namespace intel_npu
