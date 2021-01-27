#include "IttNotifyStdSrc.h"
#include "relogger.h"
#include "atlcomcli.h"
#include "shlwapi.h"
#include "Tdh.h"
#pragma comment(lib, "Tdh.lib")

class CRelogger : public ITraceEventCallback
{
protected:
    long volatile m_lvRefCount = 0;

    struct TPayload
    {
        uint64_t pid, tid, nanoseconds;
    };
    typedef std::map<uint64_t/*tid*/, TPayload> TPerThread;
    TPerThread m_map;
    LONGLONG m_llPerfFreq = 0;

    STDMETHODIMP QueryInterface(const IID& iid, void **obj) {
        if (iid == IID_IUnknown) {
            *obj = dynamic_cast<IUnknown*>(this);
        }
        else if (iid == __uuidof(ITraceEventCallback)) {
            *obj = dynamic_cast<ITraceEventCallback*>(this);
        }
        else {
            *obj = NULL;
            return E_NOINTERFACE;
        }

        AddRef();
        return S_OK;
    }

    STDMETHODIMP_(ULONG) AddRef(void) {
        return InterlockedIncrement(&m_lvRefCount);
    }

    STDMETHODIMP_(ULONG) Release() {
        ULONG ucount = InterlockedDecrement(&m_lvRefCount);
        if (ucount == 0) {
            delete this;
        }
        return ucount;
    }

    template<class T>
    TDHSTATUS DecodeEventField(PEVENT_RECORD pRecord, LPCWSTR name, T& value)
    {
        const DWORD BUF_SIZE = 10*1024;
        DWORD dwBufferSize = sizeof(TRACE_EVENT_INFO) + BUF_SIZE;
        BYTE pBuffer[sizeof(TRACE_EVENT_INFO) + BUF_SIZE] = {};
        PTRACE_EVENT_INFO pTraceEventInfo = NULL;
        pTraceEventInfo = reinterpret_cast<PTRACE_EVENT_INFO>(pBuffer);
        TDHSTATUS hr = TdhGetEventInformation(pRecord, 0, nullptr, pTraceEventInfo, &dwBufferSize);
        if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }

        LPWSTR pszPropertyName = NULL;
        PEVENT_PROPERTY_INFO pEventPropertyInfo = NULL;
        BYTE pPropertyBuffer[BUF_SIZE] = {};
        PROPERTY_DATA_DESCRIPTOR propertyDataDesc = {};

        for (DWORD index = 0; index < pTraceEventInfo->TopLevelPropertyCount; index++)
        {
            pEventPropertyInfo = &pTraceEventInfo->EventPropertyInfoArray[index];
            pszPropertyName = (LPWSTR)((PBYTE)pTraceEventInfo + pEventPropertyInfo->NameOffset);

            if (0 != StrCmpW(pszPropertyName, name))
                continue;

            propertyDataDesc.PropertyName = reinterpret_cast<ULONGLONG>(pszPropertyName);
            propertyDataDesc.ArrayIndex = ULONG_MAX;

            hr = ::TdhGetProperty(
                pRecord,
                0,
                NULL,
                1,
                &propertyDataDesc,
                BUF_SIZE,
                pPropertyBuffer);

            if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }
            value = *reinterpret_cast<T*>(pPropertyBuffer);
            return S_OK;
        }
        return ERROR_NOT_FOUND;
    }

    HRESULT STDMETHODCALLTYPE OnBeginProcessTrace(ITraceEvent* pHeaderEvent, ITraceRelogger* pTraceRelogger)
    {
        PEVENT_RECORD pRecord = nullptr;
        HRESULT hr = pHeaderEvent->GetEventRecord(&pRecord);
        if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }

        uint32_t uTimerResolution = 0;
        hr = DecodeEventField(pRecord, L"TimerResolution", uTimerResolution);
        if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }

        uint32_t PerfFreq = 0;
        hr = DecodeEventField(pRecord, L"PerfFreq", PerfFreq);
        if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }

        LARGE_INTEGER liQPF = {};
        BOOL res = QueryPerformanceFrequency(&liQPF);
        hr = HRESULT_FROM_WIN32(GetLastError());
        if (!res) { VerbosePrint("hr = 0x%X\n", hr); return hr; }
        if (liQPF.QuadPart != PerfFreq)
        {
            hr = HRESULT_FROM_WIN32(ERROR_BAD_FORMAT);
            VerbosePrint("Unsupported frequency hr = 0x%X", hr); return hr;
        }
        m_llPerfFreq = liQPF.QuadPart;
        return S_OK;//header event gets recorded anyways
    }

    HRESULT STDMETHODCALLTYPE OnEvent(ITraceEvent* pEvent, ITraceRelogger* pTraceRelogger)
    {
        PEVENT_RECORD pRecord = nullptr;
        HRESULT hr = pEvent->GetEventRecord(&pRecord);
        if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }
        if (0 == memcmp(&pRecord->EventHeader.ProviderId, &IntelSEAPI, sizeof(GUID)))
        {
            TPayload payload = {};
            hr = DecodeEventField(pRecord, L"corrector", payload);
            if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }

            LONG pid = (LONG)payload.pid;
            if (pid < 0) pid = 0;
            hr = pEvent->SetProcessId(pid);
            if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }

            LONG tid = (LONG)payload.tid;
            if (tid < 0) tid = 0;
            hr = pEvent->SetThreadId(tid);
            if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }

            LARGE_INTEGER li = {};
            //payload.nanoseconds == QPC / m_llPerfFreq * static_cast<SHiResClock::rep>(SHiResClock::period::den);
            li.QuadPart = LONGLONG(double(payload.nanoseconds) / static_cast<SHiResClock::rep>(SHiResClock::period::den) * m_llPerfFreq);
            hr = pEvent->SetTimeStamp(&li);
            if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }
        }
        return pTraceRelogger->Inject(pEvent);
    }


    HRESULT STDMETHODCALLTYPE OnFinalizeProcessTrace(ITraceRelogger*)
    {
        return S_OK;
    }

    CComPtr<ITraceRelogger> m_spTraceRelogger;
    TRACEHANDLE m_hTrace = 0;

public:
    HRESULT Process(LPCSTR szInput, LPCSTR szOutput)
    {
        static HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr;}
        hr = m_spTraceRelogger.CoCreateInstance(CLSID_TraceRelogger, nullptr, CLSCTX_INPROC_SERVER);
        if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }
        if (szInput)
        {
            hr = m_spTraceRelogger->AddLogfileTraceStream(CComBSTR(szInput), this, &m_hTrace);
            if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }
        }
        hr = m_spTraceRelogger->SetOutputFilename(CComBSTR(szOutput));
        if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }
        hr = m_spTraceRelogger->RegisterCallback(this);
        if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }
        if (szInput)
        {
            hr = m_spTraceRelogger->ProcessTrace();
            if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }
        }
/*XXX
        CComPtr<ITraceEvent> spTraceEvent;
        hr = m_spTraceRelogger->CreateEventInstance(m_hTrace, 0, &spTraceEvent);
        if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }
        hr = m_spTraceRelogger->Inject(spTraceEvent);
        if (S_OK != hr) { VerbosePrint("hr = 0x%X\n", hr); return hr; }
*/
        m_spTraceRelogger.Release();
        return S_OK;
    }
};

extern "C"
{
    SEA_EXPORT long relog_etl(const char* szInput, const char* szOutput)
    {
        return CRelogger().Process(szInput, szOutput);
    }
}


