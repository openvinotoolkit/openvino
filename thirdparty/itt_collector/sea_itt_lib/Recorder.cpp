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

#include <fcntl.h>
#include <sys/types.h>

#include <cstring>
#include <vector>

#include "IttNotifyStdSrc.h"

#ifdef _WIN32
#    include <direct.h>
#    include <io.h>
#    include <windows.h>

#    define open  crossopen
#    define write _write
#    define close _close
int crossopen(_In_z_ const char* _Filename, _In_ int _Openflag, int perm) {
    int fd = 0;
    _sopen_s(&fd, _Filename, _Openflag | _O_BINARY, _SH_DENYWR, perm);
    return fd;
}
// FIXME: support wide char mode
#endif

CRecorder::CRecorder() : m_pCurPos(nullptr) {}

size_t ChunkSize = 1 * 1020 * 1024;

bool CRecorder::Init(const std::string& path, uint64_t time, void* pCut) {
    Close(true);
    m_path = path;
#ifdef IN_MEMORY_RING
    m_pCurPos = m_pAlloc = VirtualAlloc(nullptr, m_nBufferSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    m_memmap.reset(new CMemMap(path, ChunkSize));
    m_pCurPos = m_memmap->GetPtr();
#endif
    m_nWroteTotal = 0;
    m_time = time;
    ++m_counter;
    m_pCut = pCut;
    return !!m_pCurPos;
}

size_t CRecorder::CheckCapacity(size_t size) {
#ifdef IN_MEMORY_RING
    size_t nWroteBytes = (char*)m_pCurPos - (char*)m_pAlloc;
    if (nWroteBytes + size > m_nBufferSize) {
        if (m_pBackBuffer)
            VirtualFree(m_pBackBuffer, 0, MEM_RELEASE);
        m_nBufferSize *= 2;        // We grow the buffer each time to accommodate needs
        m_pBackBuffer = m_pAlloc;  // back buffer will always be half of m_nBufferSize
        m_nBackSize = nWroteBytes;
        m_pCurPos = m_pAlloc = VirtualAlloc(nullptr, m_nBufferSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        sea::GetThreadRecord()->nMemMoveCounter += 1;
        if (!m_pCurPos)
            return 0;
    }
#else
    if (!m_memmap)
        return 0;
    size_t nWroteBytes = (char*)m_pCurPos - (char*)m_memmap->GetPtr();
    if (nWroteBytes + size > m_memmap->GetSize()) {
        m_pCurPos = m_memmap->Remap((std::max)(ChunkSize, size), m_nWroteTotal);
#    ifdef TURBO_MODE
        sea::GetThreadRecord()->nMemMoveCounter += 1;
#    endif
        if (!m_pCurPos)
            return 0;
    }
#endif
    return (std::max<size_t>)(m_nWroteTotal, 1);
}

void* CRecorder::Allocate(size_t size) {
    // must be called only from one thread
    void* pCurPos = m_pCurPos;
    m_nWroteTotal += size;
    m_pCurPos = (char*)m_pCurPos + size;
    return pCurPos;
}

void CRecorder::Close(bool bSave) {
#ifdef TURBO_MODE
    sea::GetThreadRecord()->nMemMoveCounter += 1;
#endif
#ifdef IN_MEMORY_RING
    if (bSave) {
        int fd = open(m_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, sea::FilePermissions);
        int res = 0;
        if (m_pBackBuffer)
            res = write(fd, m_pBackBuffer, uint32_t(m_nBackSize));
        if (m_pAlloc)
            res = write(fd, m_pAlloc, uint32_t((char*)m_pCurPos - (char*)m_pAlloc));
        close(fd);
    }
    if (m_pBackBuffer)
        VirtualFree(m_pBackBuffer, 0, MEM_RELEASE);
    if (m_pAlloc)
        VirtualFree(m_pAlloc, 0, MEM_RELEASE);
    m_pBackBuffer = m_pAlloc = nullptr;
#else   // IN_MEMORY_RING
    if (m_memmap)
        m_memmap->Resize(m_nWroteTotal);
    m_memmap.reset();
#endif  // IN_MEMORY_RING
    m_pCurPos = nullptr;
}

CRecorder::~CRecorder() {
    Close(true);
}

static_assert(sizeof(__itt_id) == 3 * 8, "sizeof(__itt_id) must be 3*8");
static_assert(sizeof(CTraceEventFormat::SRegularFields().tid) == 8, "sizeof(tid) must be 8");

enum EFlags {
    efHasId = 0x1,
    efHasParent = 0x2,
    efHasName = 0x4,
    efHasTid = 0x8,
    efHasData = 0x10,
    efHasDelta = 0x20,
    efHasFunction = 0x40,
    efHasPid = 0x80,
};

#pragma pack(push, 1)
// File tree is pid/domain/tid (pid is one per dll instance)
struct STinyRecord {
    uint64_t timestamp;
    ERecordType ert;
    uint8_t flags;  // EFlags
};
#pragma pack(pop)

static_assert(sizeof(STinyRecord) == 10, "SRecord must fit in 10 bytes");

template <class T>
inline T* WriteToBuff(CRecorder& recorder, const T& value) {
    T* ptr = (T*)recorder.Allocate(sizeof(T));
    if (ptr)
        *ptr = value;
    return ptr;
}

namespace sea {

extern uint64_t g_nRingBuffer;

extern std::shared_ptr<std::string> g_spCutName;

inline CRecorder* GetFile(const SRecord& record) {
    DomainExtra* pDomainExtra = reinterpret_cast<DomainExtra*>(record.domain.extra2);
    if (!pDomainExtra || !pDomainExtra->bHasDomainPath)
        return nullptr;

    static thread_local SThreadRecord* pThreadRecord = nullptr;
    if (!pThreadRecord)
        pThreadRecord = GetThreadRecord();

    if (pThreadRecord->bRemoveFiles) {
        pThreadRecord->pLastRecorder = nullptr;
        pThreadRecord->pLastDomain = nullptr;
        pThreadRecord->bRemoveFiles = false;
        pThreadRecord->files.clear();
    }
    // with very high probability the same thread will write into the same domain
    if (pThreadRecord->pLastRecorder && (pThreadRecord->pLastDomain == record.domain.nameA) &&
        (100 > pThreadRecord->nSpeedupCounter++))
        return reinterpret_cast<CRecorder*>(pThreadRecord->pLastRecorder);
    pThreadRecord->nSpeedupCounter = 0;  // we can't avoid checking ring size
    pThreadRecord->pLastDomain = record.domain.nameA;

    auto it = pThreadRecord->files.find(record.domain.nameA);
    CRecorder* pRecorder = nullptr;
    if (it != pThreadRecord->files.end()) {
        pRecorder = &it->second;
        int64_t diff = record.rf.nanoseconds - pRecorder->GetCreationTime();  // timestamp can be in the past, it's ok
        // just checking pointer of g_spCutName.get() is thread safe without any locks: we don't access internals.
        // And if it's the same we work with the old path.
        // but if it's changed we will lock and access the value below
        bool bSameCut = pRecorder->SameCut(g_spCutName.get());
        if (bSameCut && (!g_nRingBuffer || (static_cast<uint64_t>(diff) < g_nRingBuffer))) {
            pThreadRecord->pLastRecorder = pRecorder;
            return pRecorder;  // normal flow
        }
        pRecorder->Close(!bSameCut);  // time to create new file
    }

    if (!pRecorder) {
        pRecorder = &pThreadRecord->files[record.domain.nameA];
    }
    CIttLocker lock;  // locking only on file creation
    // this is theoretically possible because we check pDomainExtra->bHasDomainPath without lock above
    if (pDomainExtra->strDomainPath.empty()) {
        pThreadRecord->pLastRecorder = nullptr;
        return nullptr;
    }
    std::shared_ptr<std::string> spCutName = g_spCutName;

    CTraceEventFormat::SRegularFields rf = CTraceEventFormat::GetRegularFields();
    char path[1024] = {};
    _sprintf(path,
             "%s%llu%s%s.sea",
             pDomainExtra->strDomainPath.c_str(),
             (unsigned long long)rf.tid,
             spCutName ? (std::string("!") + *spCutName).c_str() : "",
             (g_nRingBuffer ? ((pRecorder->GetCount() % 2) ? "-1" : "-0") : ""));
    try {
        VerbosePrint("Opening: %s\n", path);
        if (!pRecorder->Init(path, rf.nanoseconds, spCutName.get())) {
            VerbosePrint("Failed to init recorder\n");
            pThreadRecord->files.erase(record.domain.nameA);
            pRecorder = nullptr;
        }
    } catch (const std::exception& exc) {
        VerbosePrint("Exception: %s\n", exc.what());
        pThreadRecord->files.erase(record.domain.nameA);
        pRecorder = nullptr;
    }
    pThreadRecord->pLastRecorder = pRecorder;
    return pRecorder;
}
}  // namespace sea

double* WriteRecord(ERecordType type, const SRecord& record) {
    CRecorder* pFile = sea::GetFile(record);
    if (!pFile)
        return nullptr;

    CRecorder& stream = *pFile;

    const size_t MaxSize =
        sizeof(STinyRecord) + 2 * sizeof(__itt_id) + 3 * sizeof(uint64_t) + sizeof(double) + sizeof(void*);
    size_t size = stream.CheckCapacity(MaxSize + record.length);
    if (!size)
        return nullptr;

    STinyRecord* pRecord = WriteToBuff(stream, STinyRecord{record.rf.nanoseconds, type});
    if (!pRecord)
        return nullptr;

    struct ShortId {
        unsigned long long a, b;
    };
    if (record.taskid.d1) {
        WriteToBuff(stream, *(ShortId*)&record.taskid);
        pRecord->flags |= efHasId;
    }

    if (record.parentid.d1) {
        WriteToBuff(stream, *(ShortId*)&record.parentid);
        pRecord->flags |= efHasParent;
    }

    if (record.pName) {
        WriteToBuff(stream, (uint64_t)record.pName);
        pRecord->flags |= efHasName;
    }

    if ((long long)record.rf.tid < 0) {
        WriteToBuff(stream, record.rf.tid);
        pRecord->flags |= efHasTid;
    }

    if (record.pData) {
        WriteToBuff(stream, (uint64_t)record.length);

        void* ptr = stream.Allocate(record.length);
        memcpy(ptr, record.pData, (unsigned int)record.length);

        pRecord->flags |= efHasData;
    }

    double* pDelta = nullptr;
    if (record.pDelta) {
        pDelta = WriteToBuff(stream, *record.pDelta);
        pRecord->flags |= efHasDelta;
    }

    if (record.function) {
        WriteToBuff(stream, (uint64_t)record.function);
        pRecord->flags |= efHasFunction;
    }

    if ((long long)record.rf.pid < 0) {
        WriteToBuff(stream, record.rf.pid);
        pRecord->flags |= efHasPid;
    }

    if (sea::g_nAutoCut && (size >= sea::g_nAutoCut)) {
        static size_t autocut = 0;
        sea::SetCutName(std::string("autocut#") + std::to_string(autocut++));
    }

    return pDelta;
}

CMemMap::CMemMap(const std::string& path, size_t size, size_t offset) {
#ifdef _WIN32
    m_hFile = CreateFile(path.c_str(),
                         GENERIC_READ | GENERIC_WRITE,
                         FILE_SHARE_READ,
                         NULL,
                         CREATE_ALWAYS,
                         FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_SEQUENTIAL_SCAN,
                         NULL);
    if (INVALID_HANDLE_VALUE == m_hFile) {
        m_hFile = NULL;
        throw std::runtime_error("Failed to open file: " + path + " err=" + std::to_string(GetLastError()));
    }
#else
    m_fdin = open(path.c_str(), O_CREAT | O_TRUNC | O_RDWR, sea::FilePermissions);
    if (-1 == m_fdin) {
        m_fdin = 0;
        throw std::runtime_error("Failed to open file: " + path + " err=" + std::to_string(errno));
    }
#endif
    Remap(size, offset);
}

void* CMemMap::Remap(size_t size, size_t offset) {
    Resize(size + offset);
    static const size_t PageSize = GetMemPageSize();
    size_t nRoundOffset = offset / PageSize * PageSize;  // align by memory page size
    m_size = size + offset % PageSize;
#ifdef _WIN32
    m_hMapping = CreateFileMapping(m_hFile, NULL, PAGE_READWRITE, 0, 0, NULL);
    ULARGE_INTEGER uliOffset = {};
    uliOffset.QuadPart = nRoundOffset;
    m_pView = ::MapViewOfFile(m_hMapping, FILE_MAP_WRITE, uliOffset.HighPart, uliOffset.LowPart, m_size);
#else
    m_pView = mmap(0, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fdin, nRoundOffset);
    if (m_pView == MAP_FAILED)
        throw std::runtime_error("Failed to map file: err=" + std::to_string(errno));

#endif
    return (char*)m_pView + offset % PageSize;
}

void CMemMap::Unmap() {
#ifdef _WIN32
    if (m_pView) {
        UnmapViewOfFile(m_pView);
        m_pView = nullptr;
    }
    if (m_hMapping) {
        CloseHandle(m_hMapping);
        m_hMapping = nullptr;
    }
#else
    if (m_pView) {
        munmap(m_pView, m_size);
        m_pView = nullptr;
    }
#endif
}

bool CMemMap::Resize(size_t size) {
    Unmap();
#ifdef _WIN32
    // resize
    LARGE_INTEGER liSize = {};
    liSize.QuadPart = size;
    return SetFilePointerEx(m_hFile, liSize, nullptr, FILE_BEGIN) && ::SetEndOfFile(m_hFile);
#else
    return 0 == ftruncate(m_fdin, size);
#endif
}

CMemMap::~CMemMap() {
    Unmap();
#ifdef _WIN32
    if (m_hMapping) {
        CloseHandle(m_hMapping);
    }
    if (m_hFile) {
        CloseHandle(m_hFile);
    }
#else
    if (m_fdin) {
        close(m_fdin);
    }
#endif
}

using namespace sea;
const bool g_bWithStacks = !!(GetFeatureSet() & sfStack);

void WriteMeta(const CTraceEventFormat::SRegularFields& main,
               __itt_string_handle* pKey,
               const char* name,
               double* pDelta) {
    WriteRecord(ERecordType::Metadata,
                SRecord{main, *g_pIntelSEAPIDomain, __itt_null, __itt_null, pKey, pDelta, name, strlen(name)});
}

class CSEARecorder : public IHandler{void Init(const CTraceEventFormat::SRegularFields& main) override{
                         // write process name into trace
                         __itt_string_handle* pKey = UNICODE_AGNOSTIC(string_handle_create)("__process__");
const char* name = GetProcessName(true);

double delta = -1;  // sort order - highest for processes written thru SEA
WriteMeta(main, pKey, name, &delta);

if (!g_savepath.empty()) {
    std::ofstream ss(GetDir(g_savepath) + "process.dct");
    ss << "{";
    ss << "'time_freq':" << GetTimeFreq();
#if INTPTR_MAX == INT64_MAX
    ss << ", 'bits':64";
#else
    ss << ", 'bits':32";
#endif
    ss << "}";
}
}

void TaskBegin(STaskDescriptor& oTask, bool bOverlapped) override {
    const char* pData = nullptr;
    size_t length = 0;
    if (g_bWithStacks) {
        static thread_local TStack* pStack = nullptr;
        if (!pStack)
            pStack = (TStack*)malloc(sizeof(TStack));
        length = (GetStack(*pStack) - 2) * sizeof(void*);
        pData = reinterpret_cast<const char*>(&(*pStack)[2]);
    }
#ifdef TURBO_MODE
    double duration = 0;
    oTask.pDur = WriteRecord(
        bOverlapped ? ERecordType::BeginOverlappedTask : ERecordType::BeginTask,
        SRecord{oTask.rf, *oTask.pDomain, oTask.id, oTask.parent, oTask.pName, &duration, pData, length, oTask.fn});
    oTask.nMemCounter = GetThreadRecord()->nMemMoveCounter;
#else
    WriteRecord(
        bOverlapped ? ERecordType::BeginOverlappedTask : ERecordType::BeginTask,
        SRecord{oTask.rf, *oTask.pDomain, oTask.id, oTask.parent, oTask.pName, nullptr, pData, length, oTask.fn});
#endif
}

void AddArg(STaskDescriptor& oTask, const __itt_string_handle* pKey, const char* data, size_t length) override {
    WriteRecord(ERecordType::Metadata,
                SRecord{oTask.rf, *oTask.pDomain, oTask.id, __itt_null, pKey, nullptr, data, length});
#ifdef TURBO_MODE
    oTask.pDur = nullptr;  // for now we don't support turbo tasks with arguments. But if count of arguments was saved
                           // it could work.
#endif
}

void AddArg(STaskDescriptor& oTask, const __itt_string_handle* pKey, double value) override {
    WriteRecord(ERecordType::Metadata, SRecord{oTask.rf, *oTask.pDomain, oTask.id, __itt_null, pKey, &value});
#ifdef TURBO_MODE
    oTask.pDur = nullptr;  // for now we don't support turbo tasks with arguments. But if count of arguments was saved
                           // it could work.
#endif
}

void AddRelation(const CTraceEventFormat::SRegularFields& rf,
                 const __itt_domain* pDomain,
                 __itt_id head,
                 __itt_string_handle* relation,
                 __itt_id tail) override {
    WriteRecord(ERecordType::Relation, SRecord{rf, *pDomain, head, tail, relation});
}

void TaskEnd(STaskDescriptor& oTask, const CTraceEventFormat::SRegularFields& rf, bool bOverlapped) override {
#ifdef TURBO_MODE
    if (oTask.pDur && (oTask.nMemCounter == GetThreadRecord()->nMemMoveCounter))
        *oTask.pDur = double(rf.nanoseconds - oTask.rf.nanoseconds);
    else
        WriteRecord(bOverlapped ? ERecordType::EndOverlappedTask : ERecordType::EndTask,
                    SRecord{rf, *oTask.pDomain, oTask.id, oTask.parent, oTask.pName, nullptr, nullptr, 0, oTask.fn});
#else
    WriteRecord(bOverlapped ? ERecordType::EndOverlappedTask : ERecordType::EndTask,
                SRecord{rf, *oTask.pDomain, oTask.id, __itt_null});
#endif
}

void Marker(const CTraceEventFormat::SRegularFields& rf,
            const __itt_domain* pDomain,
            __itt_id id,
            __itt_string_handle* pName,
            __itt_scope theScope) override {
    const char* scope = GetScope(theScope);
    WriteRecord(ERecordType::Marker, SRecord{rf, *pDomain, id, __itt_null, pName, nullptr, scope, strlen(scope)});
}

void Counter(const CTraceEventFormat::SRegularFields& rf,
             const __itt_domain* pDomain,
             const __itt_string_handle* pName,
             double value) override {
    const char* pData = nullptr;
    size_t length = 0;
    if (g_bWithStacks) {
        static thread_local TStack* pStack = nullptr;
        if (!pStack)
            pStack = (TStack*)malloc(sizeof(TStack));
        length = (GetStack(*pStack) - 3) * sizeof(void*);
        pData = reinterpret_cast<const char*>(&(*pStack)[3]);
    }
    WriteRecord(ERecordType::Counter, SRecord{rf, *pDomain, __itt_null, __itt_null, pName, &value, pData, length});
}

void SetThreadName(const CTraceEventFormat::SRegularFields& rf, const char* name) override {
    WriteThreadName(rf, name);
}
}
*g_pSEARecorder = IHandler::Register<CSEARecorder>(true);

IHandler& GetSEARecorder() {
    return *g_pSEARecorder;
}

namespace sea {

bool WriteThreadName(const CTraceEventFormat::SRegularFields& rf, const char* name) {
    CIttLocker lock;
    if (g_savepath.empty())
        return true;
    std::string path = g_savepath + "/";
    path += std::to_string(rf.pid) + "," + std::to_string(rf.tid) + ".tid";
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, FilePermissions);
    if (-1 == fd)
        return true;  // file already exists, other thread was faster
    int res = write(fd, name, (unsigned int)strlen(name));
    close(fd);
    return res != -1;
}

bool WriteGroupName(int64_t pid, const char* name) {
    if (g_savepath.empty())
        return true;
    std::string path = g_savepath + "/";
    path += std::to_string(pid) + ".pid";
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, FilePermissions);
    if (-1 == fd)
        return true;  // file already exists, other thread was faster
    int res = write(fd, name, (unsigned int)strlen(name));
    close(fd);
    return res != -1;
}

bool ReportString(__itt_string_handle* pStr) {
    if (g_savepath.empty())
        return true;
    std::string path = g_savepath + "/";
    path += std::to_string((uint64_t)pStr) + ".str";
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, FilePermissions);
    if (-1 == fd)
        return true;  // file already exists, other thread was faster
    int res = write(fd, pStr->strA, (unsigned int)strlen(pStr->strA));
    close(fd);
    return res != -1;
}

bool ReportModule(void* fn) {
    if (g_savepath.empty())
        return true;

    SModuleInfo module_info = Fn2Mdl(fn);

    std::string path = GetDir(g_savepath) + std::to_string((uint64_t)module_info.base) + ".mdl";
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, FilePermissions);
    if (-1 == fd)
        return true;  // file already exists
    std::string text = module_info.path + " " + std::to_string(module_info.size);
    int res = write(fd, text.c_str(), (unsigned int)text.size());
    close(fd);
    return res != -1;
}

int g_jit_fd = 0;

bool InitJit() {
    std::string path = GetDir(g_savepath) + "/data.jit";
    g_jit_fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, FilePermissions);
    return -1 != g_jit_fd;
}

bool WriteJit(const void* buff, size_t size) {
    return -1 != write(g_jit_fd, buff, (unsigned int)size);
}

int g_mem_fd = 0;

bool InitMemStat() {
    std::string path = GetDir(g_savepath) + "stat.mem";
    g_mem_fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, FilePermissions);
    return -1 != g_mem_fd;
}

bool WriteMemStat(const void* buff, size_t size) {
    if (g_mem_fd > -1)
        return -1 != write(g_mem_fd, buff, (unsigned int)size);
    else
        return false;
}

}  // namespace sea
