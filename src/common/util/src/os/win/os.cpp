// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/os.hpp"

namespace ov::util {

#if (_WIN32_WINNT >= 0x0602)
bool may_i_use_dynamic_code() {
    // The function GetProcessMitigationPolicy may not be available in the kernel32 library. It depends on the Windows version.
    // Need to check this at runtime if the project was built on a different platform.
    if (HMODULE kernel32 = LoadLibrary("kernel32")) {
        typedef BOOL (*fnGetProcessMitigationPolicy)(HANDLE, _PROCESS_MITIGATION_POLICY, PVOID, SIZE_T);
        fnGetProcessMitigationPolicy get_process_mitigation_policy =
            (fnGetProcessMitigationPolicy)GetProcAddress(kernel32, "GetProcessMitigationPolicy");

        if (get_process_mitigation_policy) {
            HANDLE handle = GetCurrentProcess();
            PROCESS_MITIGATION_DYNAMIC_CODE_POLICY dynamic_code_policy = {0};
            if (SUCCEEDED(get_process_mitigation_policy(handle,
                                                        ProcessDynamicCodePolicy,
                                                        &dynamic_code_policy,
                                                        sizeof(dynamic_code_policy)))) {
                return dynamic_code_policy.ProhibitDynamicCode != TRUE;
            }
        }
    }

    return true;
}
#endif

}  // namespace ov::util
