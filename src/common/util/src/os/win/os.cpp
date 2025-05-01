// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/os.hpp"

namespace ov::util {

size_t get_os_encoded_version(void) {
    static size_t os_encoded_version = 0lu;

    if (os_encoded_version == 0lu) {
        OSVERSIONINFOEXW os_version_info = {};
        ZeroMemory(&os_version_info, sizeof(OSVERSIONINFOEX));

        // Extended OS detection. We need it because of changed OS detection
        // mechanism on Windows 11
        // https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/nf-wdm-rtlgetversion
        HMODULE hNTOSKRNL = LoadLibrary("ntoskrnl.exe");
        typedef NTSTATUS (*fnRtlGetVersion)(PRTL_OSVERSIONINFOW lpVersionInformation);
        fnRtlGetVersion RtlGetVersion = nullptr;
        if (hNTOSKRNL) {
            RtlGetVersion = (fnRtlGetVersion)GetProcAddress(hNTOSKRNL, "RtlGetVersion");
        }

        // On Windows 11 GetVersionExW returns wrong information (like a Windows 8, build 9200)
        // Because of that we update inplace information if RtlGetVersion is available
        os_version_info.dwOSVersionInfoSize = sizeof(OSVERSIONINFOW);
        if (RtlGetVersion && SUCCEEDED(RtlGetVersion((PRTL_OSVERSIONINFOW)&os_version_info))) {
            os_encoded_version = (os_version_info.dwMajorVersion << 8) | os_version_info.dwMinorVersion;
            return os_encoded_version;
        }

        os_version_info.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEXW);
        #pragma warning(push)
#pragma warning(disable : 4996)
        if (FAILED(GetVersionExW((LPOSVERSIONINFOW)&os_version_info))) {
            return 0lu;
        }
#pragma warning(pop)

        os_encoded_version = (os_version_info.dwMajorVersion << 8) | os_version_info.dwMinorVersion;
    }

    return os_encoded_version;
}

#if (_WIN32_WINNT >= 0x0602)
bool ov::util::may_i_use_dynamic_code() {
    // Need to check OS version, due to the binary could have been built on different platform.
    if (get_os_encoded_version() >= 0x0602) {
        HANDLE handle = GetCurrentProcess();
        PROCESS_MITIGATION_DYNAMIC_CODE_POLICY dynamic_code_policy = {0};
        auto res = GetProcessMitigationPolicy(handle, ProcessDynamicCodePolicy, &dynamic_code_policy, sizeof(dynamic_code_policy));

        return dynamic_code_policy.ProhibitDynamicCode != TRUE;
    } else {
        return true;
    }
}
#endif

}  // namespace ov::util
