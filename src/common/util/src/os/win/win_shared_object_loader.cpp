// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>

#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

//
// LoadLibraryA, LoadLibraryW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - OK
//  WINAPI_FAMILY_SYSTEM - OK
//
// GetModuleHandleExA, GetModuleHandleExW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - OK
//  WINAPI_FAMILY_SYSTEM - OK
//
// GetModuleHandleA, GetModuleHandleW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - OK
//  WINAPI_FAMILY_SYSTEM - OK
//
// SetDllDirectoryA, SetDllDirectoryW:
//  WINAPI_FAMILY_DESKTOP_APP - OK (default)
//  WINAPI_FAMILY_PC_APP - FAIL ?? (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL ??
//  WINAPI_FAMILY_GAMES - OK
//  WINAPI_FAMILY_SERVER - FAIL
//  WINAPI_FAMILY_SYSTEM - FAIL
//
// GetDllDirectoryA, GetDllDirectoryW:
//  WINAPI_FAMILY_DESKTOP_APP - FAIL
//  WINAPI_FAMILY_PC_APP - FAIL (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL
//  WINAPI_FAMILY_GAMES - FAIL
//  WINAPI_FAMILY_SERVER - FAIL
//  WINAPI_FAMILY_SYSTEM - FAIL
//
// SetupDiGetClassDevsA, SetupDiEnumDeviceInfo, SetupDiGetDeviceInstanceIdA, SetupDiDestroyDeviceInfoList:
//  WINAPI_FAMILY_DESKTOP_APP - FAIL (default)
//  WINAPI_FAMILY_PC_APP - FAIL (defined by cmake)
//  WINAPI_FAMILY_PHONE_APP - FAIL
//  WINAPI_FAMILY_GAMES - FAIL
//  WINAPI_FAMILY_SERVER - FAIL
//  WINAPI_FAMILY_SYSTEM - FAIL
//

#if defined(WINAPI_FAMILY) && !WINAPI_PARTITION_DESKTOP
#    error "Only WINAPI_PARTITION_DESKTOP is supported, because of LoadLibrary[A|W]"
#endif

#include <direct.h>

#include <mutex>

#ifndef NOMINMAX
#    define NOMINMAX
#endif

#include <windows.h>
#include <wintrust.h>
#include <Softpub.h>

#include <wincrypt.h>


#define ENCODING (X509_ASN_ENCODING | PKCS_7_ASN_ENCODING)



#include <tchar.h>
#include <string>
#include <iostream>
// #include <wintrust.h>

// #pragma comment (lib, "wintrust")

namespace {

DWORD getSignerInfo(
	std::wstring aFileName,
	std::shared_ptr<CMSG_SIGNER_INFO> &aSignerInfo,
	HCERTSTORE &aCertStore)
{
	BOOL lRetVal = TRUE;
	DWORD lEncoding = 0;
	DWORD lContentType = 0;
	DWORD lFormatType = 0;
	HCERTSTORE lStoreHandle = NULL;
	HCRYPTMSG lCryptMsgHandle = NULL;

	CERT_INFO CertInfo = { 0 };

	DWORD lSignerInfoSize = 0;

	lRetVal = CryptQueryObject(CERT_QUERY_OBJECT_FILE,
		aFileName.data(),
		CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED,
		CERT_QUERY_FORMAT_FLAG_BINARY,
		0,
		&lEncoding,
		&lContentType,
		&lFormatType,
		&lStoreHandle,
		&lCryptMsgHandle,
		NULL);

	if (!lRetVal)
	{
		return GetLastError();
	}

	lRetVal = CryptMsgGetParam(lCryptMsgHandle,
		CMSG_SIGNER_INFO_PARAM,
		0,
		NULL,
		&lSignerInfoSize);

	if (!lRetVal)
	{
		return GetLastError();
	}

	PCMSG_SIGNER_INFO lSignerInfoPtr = (PCMSG_SIGNER_INFO) new BYTE[lSignerInfoSize];

	// Get Signer Information.
	lRetVal = CryptMsgGetParam(lCryptMsgHandle,
		CMSG_SIGNER_INFO_PARAM,
		0,
		(PVOID)lSignerInfoPtr,
		&lSignerInfoSize);

	if (!lRetVal)
	{
		delete lSignerInfoPtr;
		return GetLastError();
	}

	aSignerInfo = std::shared_ptr<CMSG_SIGNER_INFO>(lSignerInfoPtr);
	aCertStore = lStoreHandle;

	return ERROR_SUCCESS;
}

DWORD getCertificateSerialNumber(
	PCCERT_CONTEXT aCertContext,
	std::wstring &aSerialNumberWstr)
{
	if (!aCertContext)
	{
		return ERROR_INVALID_PARAMETER;
	}

	const int lBufferSize = 3;

	wchar_t lTempBuffer[lBufferSize] = { 0 };

	aSerialNumberWstr = L"";

	auto lDataBytesCount = aCertContext->pCertInfo->SerialNumber.cbData;
	for (DWORD n = 0; n < lDataBytesCount; n++)
	{

		auto lSerialByte = aCertContext->pCertInfo->SerialNumber.pbData[lDataBytesCount - (n + 1)];

		swprintf(lTempBuffer, lBufferSize*2, L"%02x", lSerialByte);

		aSerialNumberWstr += std::wstring(lTempBuffer, 2);

	}

	return ERROR_SUCCESS;
}

DWORD getCertificateContext(
	std::shared_ptr<CMSG_SIGNER_INFO> aSignerInfo,
	HCERTSTORE aCertStore,
	PCCERT_CONTEXT &aCertContextPtr)
{

	PCCERT_CONTEXT pCertContext = NULL;
	CERT_INFO CertInfo = { 0 };

	CertInfo.Issuer = aSignerInfo->Issuer;
	CertInfo.SerialNumber = aSignerInfo->SerialNumber;
	
	pCertContext = CertFindCertificateInStore(
		aCertStore,
		ENCODING,
		0,
		CERT_FIND_SUBJECT_CERT,
		(PVOID)&CertInfo,
		NULL);

	if (!pCertContext)
	{
		return GetLastError();
	}

	aCertContextPtr = pCertContext;

	return ERROR_SUCCESS;
}

DWORD queryCertificateInfo(
	PCCERT_CONTEXT aCertContext,
	DWORD aType,
	std::wstring &aOutputName)
{

	DWORD lNameLength;

	lNameLength = CertGetNameString(aCertContext,
		CERT_NAME_SIMPLE_DISPLAY_TYPE,
		aType,
		NULL,
		NULL,
		0);

	if (!lNameLength)
	{
		return GetLastError();
	}

	std::vector<wchar_t> lNameVector;
	lNameVector.reserve(lNameLength);

	// Get Issuer name.
	lNameLength = CertGetNameStringW(aCertContext,
		CERT_NAME_SIMPLE_DISPLAY_TYPE,
		aType,
		NULL,
		lNameVector.data(),
		lNameLength);

	if (!lNameLength)
	{
		return GetLastError();
	}

	aOutputName.assign(lNameVector.data(), lNameLength);

	return ERROR_SUCCESS;
}


class SignerInfo
{

public:
	SignerInfo() {};
	virtual ~SignerInfo() {};

	virtual void PrintCertificateInfo()
	{
		std::wcout << "Serial number: " << serialNumber.c_str() << std::endl;
		std::wcout << "Issuer name: " << issuerName.c_str() << std::endl;
		std::wcout << "Subject name: " << subjectName.c_str() << std::endl;
		std::wcout << "Signing algorithm: " << signAlgorithm.c_str() << std::endl;
	};

public:
	std::wstring serialNumber;
	std::wstring subjectName;
	std::wstring issuerName;
	std::wstring signAlgorithm;

};

bool verify_embedded_signature2(LPCWSTR aFileName) {
    aFileName = L"C:\\Users\\vurusovs\\Downloads\\w_openvino_toolkit_windows_2023.0.0.dev20230205_x86_64\\w_openvino_toolkit_windows_2023.0.0.dev20230205_x86_64\\runtime\\bin\\intel64\\Release\\openvino_intel_cpu_plugin.dll";
    HCERTSTORE lCertStore;
	std::shared_ptr<CMSG_SIGNER_INFO> lSignerInfo;
	DWORD lRetVal = ERROR_SUCCESS;
	PCCERT_CONTEXT lCertContexPtr = NULL;

	lRetVal = getSignerInfo(aFileName, lSignerInfo, lCertStore);
	if (lRetVal != ERROR_SUCCESS)
	{
		return lRetVal;
	}

	lRetVal = getCertificateContext(lSignerInfo, lCertStore, lCertContexPtr);
	if (lRetVal != ERROR_SUCCESS)
	{
		return lRetVal;
	}

	auto aCertInfo = std::make_shared<SignerInfo>();

	std::wstring lSerialNumber;
	lRetVal = getCertificateSerialNumber(lCertContexPtr, lSerialNumber);
	if (lRetVal == ERROR_SUCCESS)
	{
		aCertInfo->serialNumber = lSerialNumber;
	}

	std::wstring lIssuerName;
	lRetVal = queryCertificateInfo(lCertContexPtr, CERT_NAME_ISSUER_FLAG, lIssuerName);
	if (lRetVal == ERROR_SUCCESS)
	{
		aCertInfo->issuerName = lIssuerName;
	}

	std::wstring lSubjectName;
	lRetVal = queryCertificateInfo(lCertContexPtr, 0, lSubjectName);
	if (lRetVal == ERROR_SUCCESS)
	{
		aCertInfo->subjectName = lSubjectName;
	}

	// std::wstring lSignAlgorithm;
	// lRetVal = getSignatureAlgoWstring(&lCertContexPtr->pCertInfo->SignatureAlgorithm, lSignAlgorithm);
	// if (lRetVal == ERROR_SUCCESS)
	// {
	// 	aCertInfo->signAlgorithm = lSignAlgorithm;
	// }

    // aFileName = L"C:\\work\\openvino\\bin\\intel64\\Debug\\openvino_intel_gna_plugind.dll";
    aFileName = L"C:\\Users\\vurusovs\\Downloads\\openvino_2022.3.0\\w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64\\runtime\\bin\\intel64\\Release\\openvino_intel_gna_plugin.dll";

    lRetVal = getSignerInfo(aFileName, lSignerInfo, lCertStore);
	if (lRetVal != ERROR_SUCCESS)
	{
		return lRetVal;
	}

	lRetVal = getCertificateContext(lSignerInfo, lCertStore, lCertContexPtr);
	if (lRetVal != ERROR_SUCCESS)
	{
		return lRetVal;
	}

    auto aCertInfo2 = std::make_shared<SignerInfo>();

	lRetVal = getCertificateSerialNumber(lCertContexPtr, lSerialNumber);
	if (lRetVal == ERROR_SUCCESS)
	{
		aCertInfo2->serialNumber = lSerialNumber;
	}

	lRetVal = queryCertificateInfo(lCertContexPtr, CERT_NAME_ISSUER_FLAG, lIssuerName);
	if (lRetVal == ERROR_SUCCESS)
	{
		aCertInfo2->issuerName = lIssuerName;
	}

	lRetVal = queryCertificateInfo(lCertContexPtr, 0, lSubjectName);
	if (lRetVal == ERROR_SUCCESS)
	{
		aCertInfo2->subjectName = lSubjectName;
	}

	if (lCertContexPtr)
	{
		CertFreeCertificateContext(lCertContexPtr);
	}
	
	return ERROR_SUCCESS;
}

bool verify_embedded_signature2(const char* aFileName) {
    return verify_embedded_signature2(ov::util::string_to_wstring(aFileName).c_str());
}

}  // namespace

namespace ov {
namespace util {
std::shared_ptr<void> load_shared_object(const char* path, const bool& verify_signature) {
    if (verify_signature) {
        if (!is_absolute_file_path(path)) {
            // TODO: check how it works with file names
            std::stringstream ss;
            ss << "Cannot verify signature of library '" << path << "': path isn't absolute.";
            throw std::runtime_error(ss.str());
        }
        if (!verify_embedded_signature2(path)) {
            std::stringstream ss;
            ss << "Signature verification of library '" << path << "' failed";
            throw std::runtime_error(ss.str());
        }
    }
    return load_shared_object(path);
}

std::shared_ptr<void> load_shared_object(const char* path) {
    void* shared_object = nullptr;
    using GetDllDirectoryA_Fnc = DWORD (*)(DWORD, LPSTR);
    GetDllDirectoryA_Fnc IEGetDllDirectoryA = nullptr;
    if (HMODULE hm = GetModuleHandleW(L"kernel32.dll")) {
        IEGetDllDirectoryA = reinterpret_cast<GetDllDirectoryA_Fnc>(GetProcAddress(hm, "GetDllDirectoryA"));
    }
#if !WINAPI_PARTITION_SYSTEM
    // ExcludeCurrentDirectory
    if (IEGetDllDirectoryA && IEGetDllDirectoryA(0, NULL) <= 1) {
        SetDllDirectoryA("");
    }
    // LoadPluginFromDirectory
    if (IEGetDllDirectoryA) {
        DWORD nBufferLength = IEGetDllDirectoryA(0, NULL);
        std::vector<CHAR> lpBuffer(nBufferLength);
        IEGetDllDirectoryA(nBufferLength, &lpBuffer.front());

        // GetDirname
        auto dirname = [path] {
            auto pos = strchr(path, '\\');
            if (pos == nullptr) {
                return std::string{path};
            }
            std::string original(path);
            original[pos - path] = 0;
            return original;
        }();

        SetDllDirectoryA(dirname.c_str());
        shared_object = LoadLibraryA(path);

        SetDllDirectoryA(&lpBuffer.front());
    }
#endif
    if (!shared_object) {
        shared_object = LoadLibraryA(path);
    }

    if (!shared_object) {
        char cwd[1024];
        std::stringstream ss;
        ss << "Cannot load library '" << path << "': " << GetLastError() << " from cwd: " << _getcwd(cwd, sizeof(cwd));
        throw std::runtime_error(ss.str());
    }
    return {shared_object, [](void* shared_object) {
                FreeLibrary(reinterpret_cast<HMODULE>(shared_object));
            }};
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
std::shared_ptr<void> load_shared_object(const wchar_t* path, const bool& verify_signature) {
    if (verify_signature) {
        if (!is_absolute_file_path(ov::util::wstring_to_string(path))) {
            // TODO: check how it works with file names
            std::stringstream ss;
            ss << "Cannot verify signature of library '" << ov::util::wstring_to_string(std::wstring(path)) << "': path isn't absolute.";
            throw std::runtime_error(ss.str());
        }
        if (!verify_embedded_signature2(path)) {
            std::stringstream ss;
            ss << "Signature verification of library '" << ov::util::wstring_to_string(std::wstring(path)) << "' failed";
            throw std::runtime_error(ss.str());
        }
    }
    return load_shared_object(path);
}

std::shared_ptr<void> load_shared_object(const wchar_t* path) {
    void* shared_object = nullptr;
    using GetDllDirectoryW_Fnc = DWORD (*)(DWORD, LPWSTR);
    static GetDllDirectoryW_Fnc IEGetDllDirectoryW = nullptr;
    if (HMODULE hm = GetModuleHandleW(L"kernel32.dll")) {
        IEGetDllDirectoryW = reinterpret_cast<GetDllDirectoryW_Fnc>(GetProcAddress(hm, "GetDllDirectoryW"));
    }
    // ExcludeCurrentDirectory
#    if !WINAPI_PARTITION_SYSTEM
    if (IEGetDllDirectoryW && IEGetDllDirectoryW(0, NULL) <= 1) {
        SetDllDirectoryW(L"");
    }
    if (IEGetDllDirectoryW) {
        DWORD nBufferLength = IEGetDllDirectoryW(0, NULL);
        std::vector<WCHAR> lpBuffer(nBufferLength);
        IEGetDllDirectoryW(nBufferLength, &lpBuffer.front());
        auto dirname = [path] {
            auto pos = wcsrchr(path, '\\');
            if (pos == nullptr) {
                return std::wstring{path};
            }
            std::wstring original(path);
            original[pos - path] = 0;
            return original;
        }();
        SetDllDirectoryW(dirname.c_str());
        shared_object = LoadLibraryW(path);

        SetDllDirectoryW(&lpBuffer.front());
    }
#    endif
    if (!shared_object) {
        shared_object = LoadLibraryW(path);
    }
    if (!shared_object) {
        char cwd[1024];
        std::stringstream ss;
        ss << "Cannot load library '" << ov::util::wstring_to_string(std::wstring(path)) << "': " << GetLastError()
           << " from cwd: " << _getcwd(cwd, sizeof(cwd));
        throw std::runtime_error(ss.str());
    }
    return {shared_object, [](void* shared_object) {
                FreeLibrary(reinterpret_cast<HMODULE>(shared_object));
            }};
}
#endif

void* get_symbol(const std::shared_ptr<void>& shared_object, const char* symbol_name) {
    if (!shared_object) {
        std::stringstream ss;
        ss << "Cannot get '" << symbol_name << "' content from unknown library!";
        throw std::runtime_error(ss.str());
    }
    auto procAddr = reinterpret_cast<void*>(
        GetProcAddress(reinterpret_cast<HMODULE>(const_cast<void*>(shared_object.get())), symbol_name));
    if (procAddr == nullptr) {
        std::stringstream ss;
        ss << "GetProcAddress cannot locate method '" << symbol_name << "': " << GetLastError();
        throw std::runtime_error(ss.str());
    }
    return procAddr;
}
}  // namespace util
}  // namespace ov
