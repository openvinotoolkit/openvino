// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <tuple>
#include <memory>

#include <ie_compound_blob.h>

#include <gpu/gpu_config.hpp>
#include <common_test_utils/test_common.hpp>
#include <common_test_utils/test_constants.hpp>

#ifdef _WIN32
#ifdef  ENABLE_DX11

#ifndef D3D11_NO_HELPERS
#define D3D11_NO_HELPERS
#define D3D11_NO_HELPERS_DEFINED_CTX_UT
#endif

#ifndef NOMINMAX
#define NOMINMAX
#define NOMINMAX_DEFINED_CTX_UT
#endif

#include <gpu/gpu_context_api_dx.hpp>
#include <atlbase.h>
#include <d3d11.h>
#include <d3d11_4.h>

#ifdef NOMINMAX_DEFINED_CTX_UT
#undef NOMINMAX
#undef NOMINMAX_DEFINED_CTX_UT
#endif

#ifdef D3D11_NO_HELPERS_DEFINED_CTX_UT
#undef D3D11_NO_HELPERS
#undef D3D11_NO_HELPERS_DEFINED_CTX_UT
#endif

using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::gpu;

class DX11RemoteCtx_Test : public CommonTestUtils::TestsCommon {
protected:
    CComPtr<IDXGIFactory> factory;
    std::vector<CComPtr<IDXGIAdapter>> intel_adapters;
    std::vector<CComPtr<IDXGIAdapter>> other_adapters;

    void SetUp() override {
        IDXGIFactory* out_factory = nullptr;
        HRESULT err = CreateDXGIFactory(__uuidof(IDXGIFactory),
                                        reinterpret_cast<void**>(&out_factory));
        if (FAILED(err)) {
            throw std::runtime_error("Cannot create CreateDXGIFactory, error: " + std::to_string(HRESULT_CODE(err)));
        }

        factory.Attach(out_factory);

        UINT adapter_index = 0;
        const unsigned int refIntelVendorID = 0x8086;
        IDXGIAdapter* out_adapter = nullptr;
        while (factory->EnumAdapters(adapter_index, &out_adapter) != DXGI_ERROR_NOT_FOUND) {
            CComPtr<IDXGIAdapter> adapter(out_adapter);

            DXGI_ADAPTER_DESC desc{};
            adapter->GetDesc(&desc);
            if (desc.VendorId == refIntelVendorID) {
                intel_adapters.push_back(adapter);
            } else {
                other_adapters.push_back(adapter);
            }
            ++adapter_index;
        }
    }

    std::tuple<CComPtr<ID3D11Device>, CComPtr<ID3D11DeviceContext>>
    create_device_with_ctx(CComPtr<IDXGIAdapter> adapter) {
        UINT flags = 0;
        D3D_FEATURE_LEVEL feature_levels[] = { D3D_FEATURE_LEVEL_11_1,
                                               D3D_FEATURE_LEVEL_11_0,
                                             };
        D3D_FEATURE_LEVEL featureLevel;
        ID3D11Device* ret_device_ptr = nullptr;
        ID3D11DeviceContext* ret_ctx_ptr = nullptr;
        HRESULT err = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN,
                                        nullptr, flags,
                                        feature_levels,
                                        ARRAYSIZE(feature_levels),
                                        D3D11_SDK_VERSION, &ret_device_ptr,
                                        &featureLevel, &ret_ctx_ptr);
        if (FAILED(err)) {
            throw std::runtime_error("Cannot create D3D11CreateDevice, error: " +
                                     std::to_string(HRESULT_CODE(err)));
        }

        return std::make_tuple(ret_device_ptr, ret_ctx_ptr);
    }
};

TEST_F(DX11RemoteCtx_Test, smoke_make_shared_context) {
    auto ie = InferenceEngine::Core();

    for (auto adapter : intel_adapters) {
        CComPtr<ID3D11Device> device_ptr;
        CComPtr<ID3D11DeviceContext> ctx_ptr;

        ASSERT_NO_THROW(std::tie(device_ptr, ctx_ptr) =
                        create_device_with_ctx(adapter));
        auto remote_context = make_shared_context(ie,
                                                  CommonTestUtils::DEVICE_GPU,
                                                  device_ptr);
        ASSERT_TRUE(remote_context);
    }

    for (auto adapter : other_adapters) {
        CComPtr<ID3D11Device> device_ptr;
        CComPtr<ID3D11DeviceContext> ctx_ptr;

        ASSERT_NO_THROW(std::tie(device_ptr, ctx_ptr) =
                        create_device_with_ctx(adapter));
        ASSERT_THROW(make_shared_context(ie, CommonTestUtils::DEVICE_GPU,
                                         device_ptr),
                     std::runtime_error);
    }
}

#endif // ENABLE_DX11
#endif // WIN32
