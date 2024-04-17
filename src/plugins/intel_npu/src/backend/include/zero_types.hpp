// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/al/config/runtime.hpp"

/**
 * @brief Last version of Table of Graph Extension functions used within plugin
 */
using ze_graph_dditable_ext_last_t = ze_graph_dditable_ext_1_5_t;

/**
 * @brief Table of Graph Extension functions pointers and function wrappers
 * @details Use original Graph Extension functions pointers from driver for function from within lower driver versions.
 * Use function wrappers for function from within higher driver versions in order to throw when loaded driver is older
 * than required
 */
struct ze_graph_dditable_ext_decorator final {
private:
    ze_graph_dditable_ext_last_t* const _impl;
    const uint32_t _driverExtVersion;

    ze_graph_dditable_ext_decorator(const ze_graph_dditable_ext_decorator&) = delete;
    ze_graph_dditable_ext_decorator(ze_graph_dditable_ext_decorator&&) = delete;

    void throwWhenUnsupported(const std::string func, uint32_t since) {
        if (_driverExtVersion < since) {
            OPENVINO_THROW("L0 extension function ",
                           func,
                           " is only available with driver version ",
                           ZE_MAJOR_VERSION(since),
                           ".",
                           ZE_MINOR_VERSION(since),
                           " or later");
        }
    }

public:
    ze_graph_dditable_ext_decorator(ze_graph_dditable_ext_last_t* impl, uint32_t driverExtVersion)
        : _impl(impl),
          _driverExtVersion(driverExtVersion) {
        // version 1.0
        pfnCreate = _impl->pfnCreate;
        pfnDestroy = _impl->pfnDestroy;
        pfnGetProperties = _impl->pfnGetProperties;
        pfnGetArgumentProperties = _impl->pfnGetArgumentProperties;
        pfnSetArgumentValue = _impl->pfnSetArgumentValue;
        pfnAppendGraphInitialize = _impl->pfnAppendGraphInitialize;
        pfnAppendGraphExecute = _impl->pfnAppendGraphExecute;
        pfnGetNativeBinary = _impl->pfnGetNativeBinary;
        pfnDeviceGetGraphProperties = _impl->pfnDeviceGetGraphProperties;

        // version 1.1
        pfnGraphGetArgumentMetadata = _impl->pfnGraphGetArgumentMetadata;
        pfnGetArgumentProperties2 = _impl->pfnGetArgumentProperties2;

        // version 1.2
        pfnGetArgumentProperties3 = _impl->pfnGetArgumentProperties3;

        // version 1.3
        pfnQueryNetworkCreate = _impl->pfnQueryNetworkCreate;
        pfnQueryNetworkDestroy = _impl->pfnQueryNetworkDestroy;
        pfnQueryNetworkGetSupportedLayers = _impl->pfnQueryNetworkGetSupportedLayers;

        // version 1.4
        pfnBuildLogGetString = _impl->pfnBuildLogGetString;

        // version 1.5
        // wrappers replace pointers
    }
    ~ze_graph_dditable_ext_decorator() = default;

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
    ze_pfnGraphQueryNetworkCreate_ext_t pfnQueryNetworkCreate;
    ze_pfnGraphQueryNetworkDestroy_ext_t pfnQueryNetworkDestroy;
    ze_pfnGraphQueryNetworkGetSupportedLayers_ext_t pfnQueryNetworkGetSupportedLayers;

    // version 1.4
    ze_pfnGraphBuildLogGetString_ext_t pfnBuildLogGetString;

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
};

using ze_graph_dditable_ext_curr_t = ze_graph_dditable_ext_decorator;
