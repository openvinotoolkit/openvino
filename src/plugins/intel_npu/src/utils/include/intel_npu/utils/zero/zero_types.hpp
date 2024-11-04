// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_command_queue_npu_ext.h>
#include <ze_graph_ext.h>
#include <ze_graph_profiling_ext.h>

#include <string_view>

/**
 * @brief Last version of the Command Queue functions used within plugin
 */
using ze_command_queue_npu_dditable_ext_last_t = ze_command_queue_npu_dditable_ext_1_0_t;
/**
 * @brief Last version of the Graph Profiling functions used within plugin
 */
using ze_graph_profiling_dditable_ext_last_t = ze_graph_profiling_dditable_ext_t;

/**
 * @brief Table of Graph Extension functions pointers and function wrappers
 * @details Use original Graph Extension functions pointers from driver for function from within lower driver versions.
 * Use function wrappers for function from within higher driver versions in order to throw when loaded driver is older
 * than required
 */
struct ze_graph_dditable_ext_decorator final {
private:
    ze_graph_dditable_ext_t* const _impl;
    const uint32_t _driverExtVersion;

    ze_graph_dditable_ext_decorator(const ze_graph_dditable_ext_decorator&) = delete;
    ze_graph_dditable_ext_decorator(ze_graph_dditable_ext_decorator&&) = delete;

    void throwWhenUnsupported(std::string_view func, uint32_t since) {
        if (_driverExtVersion < since) {
            OPENVINO_THROW("Driver Graph extension function ",
                           func,
                           " is only available with version ",
                           ZE_MAJOR_VERSION(since),
                           ".",
                           ZE_MINOR_VERSION(since),
                           " or later");
        }
    }

public:
    ze_graph_dditable_ext_decorator(ze_graph_dditable_ext_t* impl, uint32_t driverExtVersion)
        : _impl(impl),
          _driverExtVersion(driverExtVersion),
          // version 1.0
          pfnCreate(_impl->pfnCreate),
          pfnDestroy(_impl->pfnDestroy),
          pfnGetProperties(_impl->pfnGetProperties),
          pfnGetArgumentProperties(_impl->pfnGetArgumentProperties),
          pfnSetArgumentValue(_impl->pfnSetArgumentValue),
          pfnAppendGraphInitialize(_impl->pfnAppendGraphInitialize),
          pfnAppendGraphExecute(_impl->pfnAppendGraphExecute),
          pfnGetNativeBinary(_impl->pfnGetNativeBinary),
          pfnDeviceGetGraphProperties(_impl->pfnDeviceGetGraphProperties),
          // version 1.1
          pfnGraphGetArgumentMetadata(_impl->pfnGraphGetArgumentMetadata),
          pfnGetArgumentProperties2(_impl->pfnGetArgumentProperties2),
          // version 1.2
          pfnGetArgumentProperties3(_impl->pfnGetArgumentProperties3) {
        // version 1.3
        // wrappers replace pointers
    }
    ~ze_graph_dditable_ext_decorator() = default;

    inline const uint32_t version() const {
        return _driverExtVersion;
    }

    // version 1.0
    ze_pfnGraphCreate_ext_t pfnCreate;
    ze_pfnGraphDestroy_ext_t pfnDestroy;
    ze_pfnGraphGetProperties_ext_t pfnGetProperties;
    ze_pfnGraphGetArgumentProperties_ext_t pfnGetArgumentProperties;
    ze_pfnGraphSetArgumentValue_ext_t pfnSetArgumentValue;
    ze_pfnAppendGraphInitialize_ext_t pfnAppendGraphInitialize;
    ze_pfnAppendGraphExecute_ext_t pfnAppendGraphExecute;
    ze_pfnGraphGetNativeBinary_ext_t pfnGetNativeBinary;
    ze_pfnDeviceGetGraphProperties_ext_t pfnDeviceGetGraphProperties;

    // version 1.1
    ze_pfnGraphGetArgumentMetadata_ext_t pfnGraphGetArgumentMetadata;
    ze_pfnGraphGetArgumentProperties_ext_2_t pfnGetArgumentProperties2;

    // version 1.2
    ze_pfnGraphGetArgumentProperties_ext_3_t pfnGetArgumentProperties3;

    // version 1.3
    ze_result_t ZE_APICALL pfnQueryNetworkCreate(ze_context_handle_t hContext,
                                                 ze_device_handle_t hDevice,
                                                 const ze_graph_desc_t* desc,
                                                 ze_graph_query_network_handle_t* phGraphQueryNetwork) {
        throwWhenUnsupported("pfnQueryNetworkCreate", ZE_GRAPH_EXT_VERSION_1_3);
        return _impl->pfnQueryNetworkCreate(hContext, hDevice, desc, phGraphQueryNetwork);
    }

    ze_result_t ZE_APICALL pfnQueryNetworkDestroy(ze_graph_query_network_handle_t hGraphQueryNetworkk) {
        throwWhenUnsupported("pfnQueryNetworkDestroy", ZE_GRAPH_EXT_VERSION_1_3);
        return _impl->pfnQueryNetworkDestroy(hGraphQueryNetworkk);
    }

    ze_result_t ZE_APICALL pfnQueryNetworkGetSupportedLayers(ze_graph_query_network_handle_t hGraphQueryNetwork,
                                                             size_t* pSize,
                                                             char* pSupportedLayers) {
        throwWhenUnsupported("pfnQueryNetworkGetSupportedLayers", ZE_GRAPH_EXT_VERSION_1_3);
        return _impl->pfnQueryNetworkGetSupportedLayers(hGraphQueryNetwork, pSize, pSupportedLayers);
    }

    // version 1.4
    ze_result_t ZE_APICALL pfnBuildLogGetString(ze_graph_handle_t hGraph, uint32_t* pSize, char* pBuildLog) {
        throwWhenUnsupported("pfnBuildLogGetString", ZE_GRAPH_EXT_VERSION_1_4);
        return _impl->pfnBuildLogGetString(hGraph, pSize, pBuildLog);
    }

    // version 1.5
    ze_result_t ZE_APICALL pfnCreate2(ze_context_handle_t hContext,
                                      ze_device_handle_t hDevice,
                                      const ze_graph_desc_2_t* desc,
                                      ze_graph_handle_t* phGraph) {
        throwWhenUnsupported("pfnCreate2", ZE_GRAPH_EXT_VERSION_1_5);
        return _impl->pfnCreate2(hContext, hDevice, desc, phGraph);
    }

    ze_result_t ZE_APICALL pfnQueryNetworkCreate2(ze_context_handle_t hContext,
                                                  ze_device_handle_t hDevice,
                                                  const ze_graph_desc_2_t* desc,
                                                  ze_graph_query_network_handle_t* phGraphQueryNetwork) {
        throwWhenUnsupported("pfnQueryNetworkCreate2", ZE_GRAPH_EXT_VERSION_1_5);
        return _impl->pfnQueryNetworkCreate2(hContext, hDevice, desc, phGraphQueryNetwork);
    }

    ze_result_t ZE_APICALL pfnQueryContextMemory(ze_context_handle_t hContext,
                                                 ze_graph_memory_query_type_t type,
                                                 ze_graph_memory_query_t* query) {
        throwWhenUnsupported("pfnQueryContextMemory", ZE_GRAPH_EXT_VERSION_1_5);
        return _impl->pfnQueryContextMemory(hContext, type, query);
    }

    // version 1.6
    ze_result_t ZE_APICALL pfnDeviceGetGraphProperties2(ze_device_handle_t hDevice,
                                                        ze_device_graph_properties_2_t* pDeviceGraphProperties) {
        throwWhenUnsupported("pfnDeviceGetGraphProperties2", ZE_GRAPH_EXT_VERSION_1_6);
        return _impl->pfnDeviceGetGraphProperties2(hDevice, pDeviceGraphProperties);
    }

    // version 1.7
    ze_result_t ZE_APICALL pfnGetNativeBinary2(ze_graph_handle_t hGraph,
                                               size_t* pSize,
                                               const uint8_t** pGraphNativeBinary) {
        throwWhenUnsupported("pfnGetNativeBinary2", ZE_GRAPH_EXT_VERSION_1_7);
        return _impl->pfnGetNativeBinary2(hGraph, pSize, pGraphNativeBinary);
    }

    // version 1.8
    ze_result_t ZE_APICALL pfnGetProperties2(ze_graph_handle_t hGraph, ze_graph_properties_2_t* pGraphProperties) {
        throwWhenUnsupported("ze_pfnGraphGetProperties_ext_2_t", ZE_GRAPH_EXT_VERSION_1_8);
        return _impl->pfnGetProperties2(hGraph, pGraphProperties);
    }

    ze_result_t ZE_APICALL pfnGraphInitialize(ze_graph_handle_t hGraph) {
        throwWhenUnsupported("ze_pfnGraphGetProperties_ext_2_t", ZE_GRAPH_EXT_VERSION_1_8);
        return _impl->pfnGraphInitialize(hGraph);
    }
};

/**
 * @brief Command Queue function wrappers
 * @details Use function wrappers for function from within higher driver versions in order to throw when loaded driver
 * is older than required
 */
struct ze_command_queue_npu_dditable_ext_decorator final {
private:
    ze_command_queue_npu_dditable_ext_last_t* const _impl;
    const uint32_t _commandQueueExtVersion;

    ze_command_queue_npu_dditable_ext_decorator(const ze_command_queue_npu_dditable_ext_decorator&) = delete;
    ze_command_queue_npu_dditable_ext_decorator(ze_command_queue_npu_dditable_ext_decorator&&) = delete;

    void throwWhenUnsupported(std::string_view func, uint32_t since) {
        if (_commandQueueExtVersion < since) {
            OPENVINO_THROW("Driver Command Queue extension function ",
                           func,
                           " is only available with version ",
                           ZE_MAJOR_VERSION(since),
                           ".",
                           ZE_MINOR_VERSION(since),
                           " or later");
        }
    }

public:
    ze_command_queue_npu_dditable_ext_decorator(ze_command_queue_npu_dditable_ext_last_t* impl,
                                                uint32_t commandQueueExtVersion)
        : _impl(impl),
          _commandQueueExtVersion(commandQueueExtVersion) {}
    ~ze_command_queue_npu_dditable_ext_decorator() = default;

    inline const uint32_t version() const {
        return _commandQueueExtVersion;
    }

    // version 1.0
    ze_result_t ZE_APICALL pfnSetWorkloadType(ze_command_queue_handle_t hCommandQueue,
                                              ze_command_queue_workload_type_t workloadType) {
        throwWhenUnsupported("pfnSetWorkloadType", ZE_COMMAND_QUEUE_NPU_EXT_VERSION_1_0);
        return _impl->pfnSetWorkloadType(hCommandQueue, workloadType);
    }
};

/**
 * @brief Graph profiling function pointers
 * @details Use original Graph profiling functions pointers from driver for function from within lower driver versions.
 */
struct ze_graph_profiling_ddi_table_ext_decorator final {
private:
    ze_graph_profiling_dditable_ext_last_t* const _impl;

    ze_graph_profiling_ddi_table_ext_decorator(const ze_graph_profiling_ddi_table_ext_decorator&) = delete;
    ze_graph_profiling_ddi_table_ext_decorator(ze_graph_profiling_ddi_table_ext_decorator&&) = delete;

public:
    ze_graph_profiling_ddi_table_ext_decorator(ze_graph_profiling_dditable_ext_last_t* impl)
        : _impl(impl),
          // version 1.0
          pfnProfilingPoolCreate(_impl->pfnProfilingPoolCreate),
          pfnProfilingPoolDestroy(_impl->pfnProfilingPoolDestroy),
          pfnProfilingQueryCreate(_impl->pfnProfilingQueryCreate),
          pfnProfilingQueryDestroy(_impl->pfnProfilingQueryDestroy),
          pfnProfilingQueryGetData(_impl->pfnProfilingQueryGetData),
          pfnDeviceGetProfilingDataProperties(_impl->pfnDeviceGetProfilingDataProperties),
          pfnProfilingLogGetString(_impl->pfnProfilingLogGetString) {}
    ~ze_graph_profiling_ddi_table_ext_decorator() = default;

    // version 1.0
    ze_pfnGraphProfilingPoolCreate_ext_t pfnProfilingPoolCreate;
    ze_pfnGraphProfilingPoolDestroy_ext_t pfnProfilingPoolDestroy;
    ze_pfnGraphProfilingQueryCreate_ext_t pfnProfilingQueryCreate;
    ze_pfnGraphProfilingQueryDestroy_ext_t pfnProfilingQueryDestroy;
    ze_pfnGraphProfilingQueryGetData_ext_t pfnProfilingQueryGetData;
    ze_pfnDeviceGetProfilingDataProperties_ext_t pfnDeviceGetProfilingDataProperties;
    ze_pfnGraphProfilingLogGetString_ext_t pfnProfilingLogGetString;
};

using ze_graph_dditable_ext_curr_t = ze_graph_dditable_ext_decorator;
using ze_command_queue_npu_dditable_ext_curr_t = ze_command_queue_npu_dditable_ext_decorator;
using ze_graph_profiling_dditable_ext_curr_t = ze_graph_profiling_ddi_table_ext_decorator;
