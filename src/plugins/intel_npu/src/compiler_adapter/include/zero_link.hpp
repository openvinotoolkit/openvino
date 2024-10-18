// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <type_traits>
#include <utility>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"
#include "izero_link.hpp"

namespace intel_npu {

#define NotSupportQuery(T) (T == ZE_GRAPH_EXT_VERSION_1_2)

// ext version == 1.3 && 1.4, support API (pfnQueryNetworkCreate, pfnQueryNetworkDestroy,
// pfnQueryNetworkGetSupportedLayers)
#define SupportAPIGraphQueryNetworkV1(T) (T == ZE_GRAPH_EXT_VERSION_1_3 || T == ZE_GRAPH_EXT_VERSION_1_4)

// ext version >= 1.5, support API (pfnCreate2, pfnQueryNetworkCreate2, pfnQueryContextMemory)
#define SupportAPIGraphQueryNetworkV2(T) ((!NotSupportQuery(T) && !SupportAPIGraphQueryNetworkV1(T)))

// For ext version >= 1.5, pfnCreate2 api is avaible
#define NotSupportGraph2(T) \
    (T == ZE_GRAPH_EXT_VERSION_1_2 || T == ZE_GRAPH_EXT_VERSION_1_3 || T == ZE_GRAPH_EXT_VERSION_1_4)

// A bug inside the driver makes the "pfnGraphGetArgumentMetadata" call not safe for use prior to
// "ze_graph_dditable_ext_1_6_t".
// See: E#117498
#define NotSupportArgumentMetadata(T)                                                                   \
    (T == ZE_GRAPH_EXT_VERSION_1_2 || T == ZE_GRAPH_EXT_VERSION_1_3 || T == ZE_GRAPH_EXT_VERSION_1_4 || \
     T == ZE_GRAPH_EXT_VERSION_1_5)

#define UseCopyForNativeBinary(T)                                                                       \
    (T == ZE_GRAPH_EXT_VERSION_1_2 || T == ZE_GRAPH_EXT_VERSION_1_3 || T == ZE_GRAPH_EXT_VERSION_1_4 || \
     T == ZE_GRAPH_EXT_VERSION_1_5 || T == ZE_GRAPH_EXT_VERSION_1_6)

/**
 * Adapter to use CiD through ZeroAPI
 */
template <ze_graph_ext_version_t TableExtension>
class ZeroLink final : public IZeroLink {
public:
    ZeroLink(ze_driver_handle_t driverHandle,
             ze_device_handle_t deviceHandle,
             ze_context_handle_t zeContext,
             ze_graph_dditable_ext_curr_t& graph_ddi_table_ext,
             ze_command_queue_npu_dditable_ext_curr_t& _commandQueueDdiTable,
             uint32_t group_ordinal);
    ZeroLink(const ZeroLink&) = delete;
    ZeroLink& operator=(const ZeroLink&) = delete;
    ~ZeroLink();

    std::unordered_set<std::string> queryResultFromSupportedLayers(SerializedIR serializedIR,
                                                                   const std::string& buildFlags) const override;
    ze_graph_handle_t getGraphHandle(SerializedIR serializedIR,
                                     const std::string& buildFlags,
                                     const uint32_t& flags) const override;

    ze_graph_handle_t getGraphHandle(const std::vector<uint8_t>& network) const override;

    template <ze_graph_ext_version_t T = TableExtension, std::enable_if_t<!NotSupportQuery(T), bool> = true>
    std::unordered_set<std::string> getQueryResultFromSupportedLayers(
        ze_result_t result,
        ze_graph_query_network_handle_t& hGraphQueryNetwork) const;

    NetworkMetadata getNetworkMeta(ze_graph_handle_t graphHandle) const override;

    _ze_result_t release(ze_graph_handle_t graphHandle) override;

    CompiledNetwork getCompiledNetwork(ze_graph_handle_t graphHandle) override;

    void setArgumentValue(ze_graph_handle_t graphHandle, uint32_t argi_, const void* argv) const override;

    void graphInitialie(ze_graph_handle_t graphHandle, const Config& config) const override;

    std::tuple<std::vector<ArgumentDescriptor>, std::vector<ArgumentDescriptor>> getIODesc(
        ze_graph_handle_t graphHandle) const override;

    std::shared_ptr<CommandQueue> crateCommandQueue(const Config& config) const override;

private:
    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<NotSupportArgumentMetadata(T), bool> = true>
    void getMetadata(ze_graph_dditable_ext_curr_t& graphDdiTableExt,
                     ze_graph_handle_t graphHandle,
                     uint32_t index,
                     std::vector<IODescriptor>& inputs,
                     std::vector<IODescriptor>& outputs) const;

    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<!NotSupportArgumentMetadata(T), bool> = true>
    void getMetadata(ze_graph_dditable_ext_curr_t& graphDdiTableExt,
                     ze_graph_handle_t graphHandle,
                     uint32_t index,
                     std::vector<IODescriptor>& inputs,
                     std::vector<IODescriptor>& outputs) const;

    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<UseCopyForNativeBinary(T), bool> = true>
    void getNativeBinary(ze_graph_dditable_ext_curr_t& graphDdiTableExt,
                         ze_graph_handle_t graphHandle,
                         std::vector<uint8_t>& blob,
                         const uint8_t*& blobPtr,
                         size_t& blobSize) const;

    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<!UseCopyForNativeBinary(T), bool> = true>
    void getNativeBinary(ze_graph_dditable_ext_curr_t& graphDdiTableExt,
                         ze_graph_handle_t graphHandle,
                         std::vector<uint8_t>& /* unusedBlob */,
                         const uint8_t*& blobPtr,
                         size_t& blobSize) const;

    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool> = true>
    ze_result_t seriazlideIRModelAndQueryNetworkCreateV2(SerializedIR serializedIR,
                                                         const std::string& buildFlags,
                                                         const ze_device_handle_t& _deviceHandle,
                                                         ze_graph_query_network_handle_t& hGraphQueryNetwork) const;

    // ext version >= 1.5, support API (pfnCreate2, pfnQueryNetworkCreate2, pfnQueryContextMemory)
    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool> = true>
    std::unordered_set<std::string> queryImpl(SerializedIR serializedIR, const std::string& buildFlags) const;

    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool> = true>
    ze_result_t seriazlideIRModelAndQueryNetworkCreateV1(SerializedIR serializedIR,
                                                         const std::string& buildFlags,
                                                         const ze_device_handle_t& _deviceHandle,
                                                         ze_graph_query_network_handle_t& hGraphQueryNetwork) const;

    // ext version == 1.3 && 1.4, support API (pfnQueryNetworkCreate, pfnQueryNetworkDestroy,
    // pfnQueryNetworkGetSupportedLayers)
    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool> = true>
    std::unordered_set<std::string> queryImpl(SerializedIR serializedIR, const std::string& buildFlags) const;

    // For ext version < 1.3
    template <ze_graph_ext_version_t T = TableExtension, typename std::enable_if_t<NotSupportQuery(T), bool> = true>
    std::unordered_set<std::string> queryImpl(SerializedIR serializedIR, const std::string& buildFlags) const;

    template <ze_graph_ext_version_t T = TableExtension, typename std::enable_if_t<NotSupportGraph2(T), bool> = true>
    ze_result_t createGraph(SerializedIR serializedIR,
                            const std::string& buildFlags,
                            const uint32_t& flags,
                            ze_graph_handle_t* graph) const;

    template <ze_graph_ext_version_t T = TableExtension, typename std::enable_if_t<!NotSupportGraph2(T), bool> = true>
    ze_result_t createGraph(SerializedIR serializedIR,
                            const std::string& buildFlags,
                            const uint32_t& flags,
                            ze_graph_handle_t* graph) const;

    template <typename T = TableExtension, typename std::enable_if_t<!NotSupportLogHandle(T), bool> = true>
    std::string getLatestBuildError() const;

    template <typename T = TableExtension, typename std::enable_if_t<NotSupportLogHandle(T), bool> = true>
    std::string getLatestBuildError() const {
        return "";
    }

    void initialize_graph_through_command_list(ze_graph_handle_t graphHandle, const Config& config) const;

    ze_driver_handle_t _driverHandle = nullptr;
    ze_device_handle_t _deviceHandle = nullptr;
    ze_context_handle_t _context = nullptr;

    ze_graph_dditable_ext_curr_t& _graphDdiTableExt;
    ze_command_queue_npu_dditable_ext_curr_t& _commandQueueDdiTable;

    const uint32_t _group_ordinal;

    Logger _logger;
};

}  // namespace intel_npu
