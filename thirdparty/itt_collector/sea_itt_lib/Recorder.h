/*********************************************************************************************************************************************************************************************************************************************************************************************
#   Intel® Single Event API
#
#   This file is provided under the BSD 3-Clause license.
#   Copyright (c) 2021, Intel Corporation
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#       Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#       Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#       Neither the name of the Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#   IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
**********************************************************************************************************************************************************************************************************************************************************************************************/

#pragma once

//#define TURBO_MODE

#ifdef _WIN32
#    include <windows.h>
#else
#    include <dlfcn.h>
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif
#include <string>

#include "IttNotifyStdSrc.h"
#include "TraceEventFormat.h"
#include "ittnotify.h"

inline size_t GetMemPageSize() {
#ifdef _WIN32
    SYSTEM_INFO si = {};
    GetSystemInfo(&si);
    return si.dwAllocationGranularity;
#else
    return sysconf(_SC_PAGE_SIZE);
#endif
}

class CMemMap {
    CMemMap(const CMemMap&) = delete;
    CMemMap& operator=(const CMemMap&) = delete;

public:
    CMemMap(const std::string& path, size_t size, size_t offset = 0);

    void* Remap(size_t size, size_t offset = 0);

    void* GetPtr() {
        return m_pView;
    }
    size_t GetSize() {
        return m_size;
    }

    void Unmap();

    bool Resize(size_t size);

    ~CMemMap();

protected:
#ifdef _WIN32
    HANDLE m_hFile = nullptr;
    HANDLE m_hMapping = nullptr;
#else
    int m_fdin = 0;
#endif
    size_t m_size = 0;
    void* m_pView = nullptr;
};

class CRecorder {
    CRecorder(const CRecorder&) = delete;
    CRecorder& operator=(const CRecorder&) = delete;

public:
    CRecorder();
    bool Init(const std::string& path, uint64_t time, void* pCut);
    size_t CheckCapacity(size_t size);
    void* Allocate(size_t size);
    uint64_t GetCount() {
        return m_counter;
    }
    uint64_t GetCreationTime() {
        return m_time;
    }
    void Close(bool bSave);
    inline bool SameCut(void* pCut) {
        return pCut == m_pCut;
    }
    ~CRecorder();

protected:
#ifdef IN_MEMORY_RING
    size_t m_nBufferSize = 1024 * 1024;
    void* m_pAlloc = nullptr;
    size_t m_nBackSize = 0;
    void* m_pBackBuffer = nullptr;
#endif
    std::string m_path;

    std::unique_ptr<CMemMap> m_memmap;
    size_t m_nWroteTotal = 0;
    void* m_pCurPos = nullptr;
    uint64_t m_time = 0;
    uint64_t m_counter = 0;
    void* m_pCut = nullptr;
};

enum class ERecordType : uint8_t {
    BeginTask,
    EndTask,
    BeginOverlappedTask,
    EndOverlappedTask,
    Metadata,
    Marker,
    Counter,
    BeginFrame,
    EndFrame,
    ObjectNew,
    ObjectSnapshot,
    ObjectDelete,
    Relation
};

struct SRecord {
    const CTraceEventFormat::SRegularFields& rf;
    const __itt_domain& domain;
    const __itt_id& taskid;
    const __itt_id& parentid;
    const __itt_string_handle* pName;
    double* pDelta;
    const char* pData;
    size_t length;
    void* function;
};
double* WriteRecord(ERecordType type, const SRecord& record);
void WriteMeta(const CTraceEventFormat::SRegularFields& main,
               __itt_string_handle* pKey,
               const char* name,
               double* pDelta = nullptr);

namespace sea {
struct IHandler;
bool WriteThreadName(const CTraceEventFormat::SRegularFields& rf, const char* name);
bool WriteGroupName(int64_t pid, const char* name);
bool ReportString(__itt_string_handle* pStr);
bool ReportModule(void* fn);
bool InitJit();
bool WriteJit(const void* buff, size_t size);
bool InitMemStat();
bool WriteMemStat(const void* buff, size_t size);
}  // namespace sea

sea::IHandler& GetSEARecorder();
