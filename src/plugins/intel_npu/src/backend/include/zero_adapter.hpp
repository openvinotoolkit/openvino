// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <type_traits>
#include <utility>

#include "intel_npu/common/iadapter.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"
#include "zero_init.hpp"

namespace intel_npu {

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

// A bug inside the driver makes the "pfnGraphGetArgumentMetadata" call not safe for use prior to
// "ze_graph_dditable_ext_1_6_t".
// See: E#117498
#define NotSupportArgumentMetadata(T)                                                                              \
    (std::is_same<T, ze_graph_dditable_ext_1_2_t>::value || std::is_same<T, ze_graph_dditable_ext_1_3_t>::value || \
     std::is_same<T, ze_graph_dditable_ext_1_4_t>::value || std::is_same<T, ze_graph_dditable_ext_1_5_t>::value)

#define UseCopyForNativeBinary(T)                                                                                  \
    (std::is_same<T, ze_graph_dditable_ext_1_2_t>::value || std::is_same<T, ze_graph_dditable_ext_1_3_t>::value || \
     std::is_same<T, ze_graph_dditable_ext_1_4_t>::value || std::is_same<T, ze_graph_dditable_ext_1_5_t>::value || \
     std::is_same<T, ze_graph_dditable_ext_1_6_t>::value)

/**
 * Adapter to use CiD through ZeroAPI
 */
template <typename TableExtension>
class ZeroAdapter final : public IAdapter {
public:
    ZeroAdapter(const std::shared_ptr<ZeroInitStructsHolder>& initStructs);
    ZeroAdapter(const ZeroAdapter&) = delete;
    ZeroAdapter& operator=(const ZeroAdapter&) = delete;
    ~ZeroAdapter();

    std::unordered_set<std::string> queryResultFromSupportedLayers(
        std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
        const std::string& buildFlags) const override;
    ze_graph_handle_t getGraphHandle(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                     const std::string& buildFlags,
                                     const uint32_t& flags) const override;

    ze_graph_handle_t getGraphHandle(const std::vector<uint8_t>& network) const override;

    template <typename T = TableExtension, std::enable_if_t<!NotSupportQuery(T), bool> = true>
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

    ze_device_graph_properties_t getDeviceGraphProperties() const override;

private:
    template <typename T = TableExtension, typename std::enable_if_t<NotSupportArgumentMetadata(T), bool> = true>
    void getMetadata(ze_graph_handle_t graphHandle,
                     uint32_t index,
                     std::vector<IODescriptor>& inputs,
                     std::vector<IODescriptor>& outputs) const;

    template <typename T = TableExtension, typename std::enable_if_t<!NotSupportArgumentMetadata(T), bool> = true>
    void getMetadata(ze_graph_handle_t graphHandle,
                     uint32_t index,
                     std::vector<IODescriptor>& inputs,
                     std::vector<IODescriptor>& outputs) const;

    template <typename T = TableExtension, typename std::enable_if_t<UseCopyForNativeBinary(T), bool> = true>
    void getNativeBinary(ze_graph_handle_t graphHandle,
                         std::vector<uint8_t>& blob,
                         const uint8_t*& blobPtr,
                         size_t& blobSize) const;

    template <typename T = TableExtension, typename std::enable_if_t<!UseCopyForNativeBinary(T), bool> = true>
    void getNativeBinary(ze_graph_handle_t graphHandle,
                         std::vector<uint8_t>& /* unusedBlob */,
                         const uint8_t*& blobPtr,
                         size_t& blobSize) const;

    template <typename T = TableExtension, typename std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool> = true>
    ze_result_t seriazlideIRModelAndQueryNetworkCreateV2(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                                         const std::string& buildFlags,
                                                         ze_graph_query_network_handle_t& hGraphQueryNetwork) const;

    // ext version >= 1.5, support API (pfnCreate2, pfnQueryNetworkCreate2, pfnQueryContextMemory)
    template <typename T = TableExtension, typename std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool> = true>
    std::unordered_set<std::string> queryImpl(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                              const std::string& buildFlags) const;

    template <typename T = TableExtension, typename std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool> = true>
    ze_result_t seriazlideIRModelAndQueryNetworkCreateV1(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                                         const std::string& buildFlags,
                                                         ze_graph_query_network_handle_t& hGraphQueryNetwork) const;

    // ext version == 1.3 && 1.4, support API (pfnQueryNetworkCreate, pfnQueryNetworkDestroy,
    // pfnQueryNetworkGetSupportedLayers)
    template <typename T = TableExtension, typename std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool> = true>
    std::unordered_set<std::string> queryImpl(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                              const std::string& buildFlags) const;

    // For ext version < 1.3
    template <typename T = TableExtension, typename std::enable_if_t<NotSupportQuery(T), bool> = true>
    std::unordered_set<std::string> queryImpl(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                                              const std::string& buildFlags) const;

    template <typename T = TableExtension, typename std::enable_if_t<NotSupportGraph2(T), bool> = true>
    void createGraph(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                     const std::string& buildFlags,
                     const uint32_t& flags,
                     ze_graph_handle_t* graph) const;

    template <typename T = TableExtension, typename std::enable_if_t<!NotSupportGraph2(T), bool> = true>
    void createGraph(std::pair<size_t, std::shared_ptr<uint8_t>> serializedIR,
                     const std::string& buildFlags,
                     const uint32_t& flags,
                     ze_graph_handle_t* graph) const;

    void initialize_graph_through_command_list(ze_graph_handle_t graphHandle, const Config& config) const;

    std::shared_ptr<ZeroInitStructsHolder> _initStructs;

    uint32_t _groupOrdinal;

    Logger _logger;
};

}  // namespace intel_npu
