// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/postgres_helpers.hpp"

namespace ov {
namespace test {
namespace utils {

const char* PGQL_ENV_CONN_NAME = "OV_POSTGRES_CONN";    // Environment variable with connection settings
const char* PGQL_ENV_SESS_NAME = "OV_TEST_SESSION_ID";  // Environment variable identifies current session
const char* PGQL_ENV_RUN_NAME = "OV_TEST_RUN_ID";       // Environment variable with external run id
const char* PGQL_ENV_RLVL_NAME = "OV_TEST_REPORT_LVL";  // Environment variable identifies reporting
                                                        // level: default ("", empty), "fast", "suite"
fnPQconnectdb PQconnectdb;
fnPQescapeStringConn PQescapeStringConn;
fnPQstatus PQstatus;
fnPQfinish PQfinish;
fnPQerrorMessage PQerrorMessage;

fnPQexec PQexec;
fnPQresultStatus PQresultStatus;
fnPQgetvalue PQgetvalue;
fnPQgetisnull PQgetisnull;
fnPQclear PQclear;
fnPQresultErrorMessage PQresultErrorMessage;

const char* PGPrefix(const char* text, ::testing::internal::GTestColor color) {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif
    ::testing::internal::ColoredPrintf(color, text);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    return "";
}

PGresultHolder PostgreSQLConnection::common_query(const char* query) {
#ifdef PGQL_DEBUG
    std::cerr << query << std::endl;
#endif
    if (!m_is_connected)
        return PGresultHolder();
    PGresultHolder result(PQexec(this->m_active_connection, query));
    // Connection could be closed by a timeout, we may try to reconnect once.
    // We don't reconnect on each call because it may make testing significantly slow in
    // case of connection issues. Better to finish testing with incomplete results and
    // free a machine. Otherwise we will lose all results.
    if (result.get() == nullptr) {
        try_reconnect();
        // If reconnection attempt was successfull - let's try to set new query
        if (m_is_connected) {
            result.reset(PQexec(this->m_active_connection, query));
        }
    }
    if (result.get() == nullptr) {
        std::cerr << PG_ERR << "Error while querying PostgreSQL\n";
    }
    return result;
}

PGresultHolder PostgreSQLConnection::query(const char* query,
                                           const ExecStatusType expectedStatus,
                                           const bool smartRetry) {
    PGresultHolder result = common_query(query);
    uint8_t queryCounter = 1;
    size_t selectPos = smartRetry ? std::string::npos : std::string(query).find("SELECT");

    while (result.get() != nullptr && queryCounter < serializationTriesCount) {
        ExecStatusType execStatus = PQresultStatus(result.get());
        if (execStatus == expectedStatus) {
            break;
        }
        std::string errStr = PQresultErrorMessage(result.get());
        std::cerr << PG_WRN << "Received unexpected result (" << static_cast<unsigned int>(execStatus)
                  << ") from PostgreSQL, expected: " << static_cast<unsigned int>(expectedStatus) << std::endl;

        // After transactional queries were introduced - we need to check error message and try
        // do a query again if it is expected serialization error.
        // More about serialization: https://www.postgresql.org/docs/9.5/transaction-iso.html
        if (errStr.find("could not serialize access") != std::string::npos ||
            errStr.find("current transaction is aborted") != std::string::npos) {
            std::cerr << PG_WRN << "Serialization error: " << errStr
                      << "\nTrying again, try attempt: " << static_cast<uint32_t>(queryCounter++) << std::endl;
            uint32_t waitTime = 50 + static_cast<uint32_t>(std::rand()) % 150;
#ifdef _WIN32
            Sleep(waitTime);  // Wait some time for the next attempt
#else
            struct timespec waitTimeTS = {0, waitTime * 1000};
            if (nanosleep(&waitTimeTS, &waitTimeTS) != 0) {
                std::cerr << PG_WRN << "nanosleep returned value != 0\n";
            }
#endif
            // We may have some connection issues, each tenth step try to reconnect
            if (smartRetry && (queryCounter % 10) == 0) {
                try_reconnect();
            }
            // Each fifth step it tries to call non-transactional part of query
            if (smartRetry && selectPos != std::string::npos && (queryCounter % 5) == 0) {
                std::cerr << PG_WRN << "Sending a request with no transactional part\n";
                result = common_query(query + selectPos);
                continue;
            }
            result = common_query(query);
        } else {
            std::cerr << PG_ERR << "Error message: " << errStr << std::endl;
            result.reset(nullptr);
        }
    }
    if (queryCounter >= serializationTriesCount) {
        std::cerr << PG_ERR << "Cannot execute query due to serialization error, failing" << std::endl;
        result.reset(nullptr);
    }
    return result;
}

/// \brief Tries to reconnect in case of connection issues (usual usage - connection timeout).
void PostgreSQLConnection::try_reconnect(void) {
    if (!m_is_connected) {
        return;
    }
    if (m_active_connection != nullptr) {
        try {
            PQfinish(m_active_connection);
        } catch (...) {
            std::cerr << PG_ERR << "An exception while finishing PostgreSQL connection\n";
        }
        this->m_active_connection = nullptr;
        this->m_is_connected = false;
    }
    std::cerr << PG_INF << "Reconnecting to the PostgreSQL server...\n";
    initialize();
}

std::shared_ptr<PostgreSQLConnection> connection(nullptr);
std::shared_ptr<PostgreSQLConnection> PostgreSQLConnection::get_instance(void) {
    if (connection.get() == nullptr) {
        connection.reset(new PostgreSQLConnection());
    }
    return connection;
}

PostgreSQLConnection::~PostgreSQLConnection(void) {
    if (m_active_connection) {
        PQfinish(this->m_active_connection);
        this->m_active_connection = nullptr;
        this->m_is_connected = false;
    }
}

/// \brief Initialization of exact object. Uses environment variable PGQL_ENV_CONN_NAME for making a connection.
/// \returns Returns false in case of failure or absence of ENV-variable.
///          Returns true in case of connection has been succesfully established.
bool PostgreSQLConnection::initialize(void) {
    if (this->m_active_connection != nullptr) {
        std::cerr << PG_WRN << "PostgreSQL connection is already established.\n";
        return true;
    }

#ifdef PGQL_DYNAMIC_LOAD
    if (!load_libpq()) {
        return false;
    }
#endif

    const char* envConnString = nullptr;
    envConnString = std::getenv(PGQL_ENV_CONN_NAME);

    if (envConnString == nullptr) {
        std::cerr << PG_WRN << "PostgreSQL connection string isn't found in Environment (" << PGQL_ENV_CONN_NAME
                  << ")\n";
        return false;
    } else {
        std::cerr << PG_INF << "PostgreSQL connection string:\n";
        std::cerr << PG_INF << envConnString << std::endl;
    }

    this->m_active_connection = PQconnectdb(envConnString);

    ConnStatusType connStatus = PQstatus(this->m_active_connection);

    if (connStatus != CONNECTION_OK) {
        std::cerr << PG_ERR << "Cannot connect to PostgreSQL: " << static_cast<uint32_t>(connStatus) << std::endl;
        return false;
    } else {
        std::cerr << PG_INF << "Connected to PostgreSQL successfully\n";
    }

    this->m_is_connected = true;

    return true;
}

#ifdef PGQL_DYNAMIC_LOAD
/// \brief Loads libpq module in runtime
bool PostgreSQLConnection::load_libpq(void) {
#    ifdef _WIN32
    modLibPQ = std::shared_ptr<HMODULE>(new HMODULE(LoadLibrary("libpq.dll")), [](HMODULE* ptr) {
        if (*ptr != (HMODULE)0) {
            std::cerr << PG_INF << "Freeing libPQ.dll handle\n";
            try {
                FreeLibrary(*ptr);
            } catch (...) {
            }
        }
    });
#    else
    modLibPQ = std::shared_ptr<HMODULE>(new HMODULE(dlopen("libpq.so", RTLD_LAZY)), [](HMODULE* ptr) {
        if (*ptr != (HMODULE)0) {
            std::cerr << PG_INF << "Freeing libPQ.so handle\n";
            try {
                dlclose(*ptr);
            } catch (...) {
            }
        }
    });
    if (*modLibPQ == (HMODULE)0) {
        modLibPQ = std::shared_ptr<HMODULE>(new HMODULE(dlopen("libpq.so.5", RTLD_LAZY)), [](HMODULE* ptr) {
            if (*ptr != (HMODULE)0) {
                std::cerr << PG_INF << "Freeing libPQ.so.5 handle\n";
                try {
                    dlclose(*ptr);
                } catch (...) {
                }
            }
        });
    }
    if (*modLibPQ == (HMODULE)0) {
        modLibPQ = std::shared_ptr<HMODULE>(new HMODULE(dlopen("libpq.dylib", RTLD_LAZY)), [](HMODULE* ptr) {
            if (*ptr != (HMODULE)0) {
                std::cerr << PG_INF << "Freeing libPQ.dylib handle\n";
                try {
                    dlclose(*ptr);
                } catch (...) {
                }
            }
        });
    }
    if (*modLibPQ == (HMODULE)0) {
        modLibPQ = std::shared_ptr<HMODULE>(new HMODULE(dlopen("libpq.5.dylib", RTLD_LAZY)), [](HMODULE* ptr) {
            if (*ptr != (HMODULE)0) {
                std::cerr << PG_INF << "Freeing libPQ.5.dylib handle\n";
                try {
                    dlclose(*ptr);
                } catch (...) {
                }
            }
        });
    }
    if (*modLibPQ == (HMODULE)0) {
        modLibPQ = std::shared_ptr<HMODULE>(
            new HMODULE(dlopen("/opt/homebrew/opt/libpq/lib/libpq.dylib", RTLD_LAZY)),
            [](HMODULE* ptr) {
                if (*ptr != (HMODULE)0) {
                    std::cerr << PG_INF << "Freeing /opt/homebrew/opt/libpq/lib/libPQ.dylib handle\n";
                    try {
                        dlclose(*ptr);
                    } catch (...) {
                    }
                }
            });
    }
#    endif
    if (*modLibPQ == (HMODULE)0) {
        std::cerr << PG_WRN << "Cannot load PostgreSQL client module libPQ, reporting is unavailable\n";
        return false;
    } else {
        std::cerr << PG_INF << "PostgreSQL client module libPQ has been loaded\n";
    }

#    ifdef _WIN32
#        define GETPROC(name)                                                                  \
            name = (fn##name)GetProcAddress(*modLibPQ, #name);                                 \
            if (name == nullptr) {                                                             \
                std::cerr << PG_ERR << "Couldn't load procedure " << #name << " from libPQ\n"; \
                return false;                                                                  \
            }
#    else
#        define GETPROC(name)                                                               \
            name = (fn##name)dlsym(*modLibPQ, #name);                                       \
            if (name == nullptr) {                                                          \
                std::cerr << PG_ERR << "Couldn't load symbol " << #name << " from libPQ\n"; \
                return false;                                                               \
            }
#    endif

    GETPROC(PQconnectdb);
    GETPROC(PQstatus);
    GETPROC(PQescapeStringConn);
    GETPROC(PQfinish);
    GETPROC(PQerrorMessage);
    GETPROC(PQexec);
    GETPROC(PQresultStatus);
    GETPROC(PQgetvalue);
    GETPROC(PQgetisnull);
    GETPROC(PQclear);
    GETPROC(PQresultErrorMessage);

    return true;
}
#endif

namespace PostgreSQLHelpers {

std::vector<std::string> parse_value_param(std::string text) {
    std::vector<std::string> results;
    size_t beginning = 0;
    size_t chrPos = 0;
    char pairingChar = 0;
    for (auto it = text.begin(); it != text.end(); ++it, ++chrPos) {
        if (pairingChar == 0) {  // Looking for opening char
            switch (*it) {
            case '"':
            case '\'':
                pairingChar = *it;
                break;
            case '{':
                pairingChar = '}';
                break;
            }
            beginning = chrPos + 1;
        } else if (*it != pairingChar) {  // Skip while don't face with paring char
            continue;
        } else {
            if (chrPos < 3 || (text[chrPos - 1] != '\\' && text[chrPos - 2] != '\\')) {
                size_t substrLength = chrPos - beginning;
                if (substrLength > 0 && (beginning + substrLength) < text.length()) {
                    results.push_back(text.substr(beginning, chrPos - beginning));
                }
                pairingChar = 0;
            }
        }
    }
    return results;
}

std::string get_os_version(void) {
#ifndef _WIN32
    struct utsname uts;
    uname(&uts);
    return uts.sysname;
#else
    OSVERSIONINFOEXW osVersionInfo = {};

    // Extended OS detection. We need it because of changed OS detection
    // mechanism on Windows 11
    // https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/nf-wdm-rtlgetversion
    HMODULE hNTOSKRNL = LoadLibrary("ntoskrnl.exe");
    typedef NTSTATUS (*fnRtlGetVersion)(PRTL_OSVERSIONINFOW lpVersionInformation);
    fnRtlGetVersion RtlGetVersion = nullptr;
    if (hNTOSKRNL) {
        RtlGetVersion = (fnRtlGetVersion)GetProcAddress(hNTOSKRNL, "RtlGetVersion");
    }

    ZeroMemory(&osVersionInfo, sizeof(OSVERSIONINFOEX));
    osVersionInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEXW);

    std::stringstream winVersion;
    winVersion << "Windows ";

#    pragma warning(push)
#    pragma warning(disable : 4996)
    if (FAILED(GetVersionExW((LPOSVERSIONINFOW)&osVersionInfo))) {
        return "Unknown Windows OS";
    }
#    pragma warning(pop)

    // On Windows 11 GetVersionExW returns wrong information (like a Windows 8, build 9200)
    // Because of that we update inplace information if RtlGetVersion is available
    osVersionInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFOW);
    if (RtlGetVersion && SUCCEEDED(RtlGetVersion((PRTL_OSVERSIONINFOW)&osVersionInfo))) {
        if (osVersionInfo.dwBuildNumber >= 22000)
            winVersion << "11";
    }

    DWORD encodedVersion = (osVersionInfo.dwMajorVersion << 8) | osVersionInfo.dwMinorVersion;
    switch (encodedVersion) {
    case 0x0A00:
        if (osVersionInfo.dwBuildNumber < 22000)
            winVersion << ((osVersionInfo.wProductType == VER_NT_WORKSTATION) ? "10" : "2016");
        break;
    case 0x0603:
        winVersion << ((osVersionInfo.wProductType == VER_NT_WORKSTATION) ? "8.1" : "2012 R2");
        break;
    case 0x0602:
        winVersion << ((osVersionInfo.wProductType == VER_NT_WORKSTATION) ? "8" : "2012");
        break;
    case 0x0601:
        winVersion << ((osVersionInfo.wProductType == VER_NT_WORKSTATION) ? "7" : "2008 R2");
        break;
    case 0x0600:
        winVersion << ((osVersionInfo.wProductType == VER_NT_WORKSTATION) ? "Vista" : "2008");
        break;
    default:
        winVersion << osVersionInfo.dwMajorVersion << "." << osVersionInfo.dwMinorVersion;
        break;
    }
    if (osVersionInfo.wSuiteMask & VER_SUITE_BACKOFFICE)
        winVersion << " BackOffice";
    if (osVersionInfo.wSuiteMask & VER_SUITE_BLADE)
        winVersion << " Web Edition";
    if (osVersionInfo.wSuiteMask & VER_SUITE_COMPUTE_SERVER)
        winVersion << " Compute Cluster Edition";
    if (osVersionInfo.wSuiteMask & VER_SUITE_DATACENTER)
        winVersion << " Datacenter Edition";
    if (osVersionInfo.wSuiteMask & VER_SUITE_ENTERPRISE)
        winVersion << " Enterprise Edition";
    if (osVersionInfo.wSuiteMask & VER_SUITE_EMBEDDEDNT)
        winVersion << " Embedded";
    if (osVersionInfo.wSuiteMask & VER_SUITE_PERSONAL)
        winVersion << " Home";
    if (osVersionInfo.wSuiteMask & VER_SUITE_SMALLBUSINESS)
        winVersion << " Small Business";

    winVersion << " Build: " << osVersionInfo.dwBuildNumber;

    return winVersion.str();
#endif
}

std::string get_executable_name(void) {
#ifdef _WIN32
    char cFilePath[MAX_PATH] = {};
    GetModuleFileName(nullptr, cFilePath, MAX_PATH);
    std::string filePath(cFilePath);
#elif !defined(__APPLE__)
    std::string filePath;
    std::ifstream("/proc/self/comm") >> filePath;
    return filePath;
#else
    uint32_t cFilePathSize = MAXPATHLEN;
    std::vector<char> cFilePath(static_cast<size_t>(cFilePathSize));
    if (_NSGetExecutablePath(cFilePath.data(), &cFilePathSize) == -1) {
        // Trying to reallocate, once
        cFilePath.reserve(cFilePathSize + 1);
        if (_NSGetExecutablePath(cFilePath.data(), &cFilePathSize) == -1) {
            return "macos_failed";
        }
    }
    std::string filePath(cFilePath.data());
#endif
    return filePath.substr(filePath.find_last_of("/\\") + 1);
}

std::string get_hostname(void) {
#ifdef _WIN32
    DWORD szHostName = MAX_COMPUTERNAME_LENGTH;
    char cHostName[MAX_COMPUTERNAME_LENGTH + 1] = {};
    if (FAILED(GetComputerName(cHostName, &szHostName))) {
        std::cerr << PG_ERR << "Cannot get a host name\n";
#else
    char cHostName[HOST_NAME_MAX];
    if (gethostname(cHostName, HOST_NAME_MAX)) {
        std::cerr << PG_ERR << "Cannot get a host name\n";
        return "NOT_FOUND";
#endif
    }
    return cHostName;
}  // namespace PostgreSQLHelpers

void add_pair(std::map<std::string, std::string>& keyValues, const std::string& key, const std::string& value) {
    size_t dPos;
    // Parse IR_name for opName and hash
    if (key == "IR_name" && (dPos = value.find('_')) != std::string::npos) {
        keyValues["opName"] = value.substr(0, dPos);
        keyValues["opSet"] = "unknown";  // Need to set
        keyValues["hashXml"] = value.substr(dPos + 1);
        keyValues["pathXml"] = value;
        return;
    }
    if (key == "Op") {
        if ((dPos = value.find('.')) == std::string::npos) {
            keyValues["opName"] = value.substr(0, dPos);
            keyValues["opSet"] = "unknown";  // Need to set later
        } else {
            keyValues["opName"] = value.substr(0, dPos);
            keyValues["opSet"] = value.substr(dPos + 1);
        }
    }
    // Defining a subgraph extractors as an operations
    if (key == "Extractor") {
        keyValues["opName"] = value;
        keyValues["opSet"] = "subgraph";  // Need to set later
    }
    // Parse IR for opName and hash
    if (key == "IR") {
        if ((dPos = value.find_last_of('/')) != std::string::npos ||
            (dPos = value.find_last_of('\\')) != std::string::npos) {
            dPos += 1;                                                             // Ignores slash
            keyValues["hashXml"] = value.substr(dPos, value.length() - dPos - 4);  // exclude extension
            keyValues["pathXml"] = value.substr(dPos);
        } else {
            keyValues["hashXml"] = value;
            keyValues["pathXml"] = value + ".xml";
        }
        return;
    }
    // Parse Function for opName and opSet
    if (key == "Function" && (dPos = value.find('_')) != std::string::npos) {
        keyValues["opName"] = value.substr(0, dPos);
        keyValues["opSet"] = value.substr(dPos + 6);  // Skipping "opset"
        return;
    }
    // Normalize target devices
    if (key == "target_device" || key == "TargetDevice" || key == "Device" || key == "targetDevice") {
        if (value == "CPU") {
// see https://sourceforge.net/p/predef/wiki/Architectures/
#if defined(__arm__) || defined(_M_ARM) || defined(__ARMEL__)
            keyValues["targetDeviceArch"] = "arm";
#elif defined(__aarch64__) || defined(_M_ARM64)
            keyValues["targetDeviceArch"] = "arm64";
#elif defined(i386) || defined(__i386) || defined(__i386__) || defined(__IA32__) || defined(_M_I86) || \
    defined(_M_IX86) || defined(__X86__) || defined(_X86_) || defined(__I86__) || defined(__386) ||    \
    defined(__ILP32__) || defined(_ILP32) || defined(__wasm32__) || defined(__wasm32)
            keyValues["targetDeviceArch"] = "x86";
#elif defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) || \
    defined(_M_AMD64)
            keyValues["targetDeviceArch"] = "x64";
#elif defined(__riscv)
            keyValues["targetDeviceArch"] = "riskv64";
#endif
        }
        keyValues["targetDevice"] = value;
        return;
    }
    std::string lKey = key;
    std::transform(lKey.begin(), lKey.end(), lKey.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    if (lKey == "config") {
        keyValues[lKey] = value;
        return;
    }
    keyValues[key] = value;
}

bool parse_test_name(const char* line, std::map<std::string, std::string>& keyValues) {
    const std::vector<std::string> knownExceptions = {"target_device", "IR_name"};

    std::string paramName;

    // Looking for '/' as a delimeter between group name and parameters
    const char *ptr = line, *grpNameEnd = nullptr;
    while (*ptr != 0 && *ptr != '/')
        ++ptr;
    if (*ptr == 0) {
        return false;  // group name isn't identified, wrong line on input
    }
    keyValues["__groupName__"] = std::string(line, ptr - line);
    grpNameEnd = ptr + 1;

    // Try to parse param1=value1_param2=(paramX=valueX_)... as a key=value
    const char *paramNameStart = ptr + 1, *paramValueStart = nullptr;
    // Brakets counter to be able catch inherited values like ((paramX=valueX)(paramY=valueY))
    unsigned int bkt = 0;

    while (*ptr != 0) {  // Do until line ends
        while (*ptr != 0 && *ptr != '=')
            ++ptr;  // find key=value delimeter or EOL
        if (paramNameStart == ptr)
            break;                                                      // break if nothing found
        paramName = std::string(paramNameStart, ptr - paramNameStart);  // store parameter name
        if (*ptr == '=') {                                              // if we found a key=value delimeter (not EOL)
            paramValueStart = ++ptr;                                    // start looking for value after '=' char
            while (*ptr != 0) {                                         // try to find a value  end
                if (*ptr == '(')  // braket found - ignores key=value delimeter until string end or all closing braket
                                  // will be found
                    ++bkt;
                else if (*ptr == ')')  // closing braket found - decrease breaket counter
                    --bkt;
                else if (*ptr == '=' && bkt == 0)  // stop on key=value delimeter only outside of brakets
                    break;
                ++ptr;
            }
            if (*ptr == 0) {  // if we stopped at the end of line
                // Removing trailing underscore and non-printed symbols
                while (ptr > paramValueStart && (*(ptr - 1) == '_' || *(ptr - 1) < 0x20))
                    --ptr;
                add_pair(keyValues, std::string(paramName), std::string(paramValueStart, ptr - paramValueStart));
            } else if (*ptr == '=') {  // if we stopped by item's delimeter (paramN=valueN_paramN+1=valueN+1)
                // Because we have params which contains underscore, this algorithm may interpret '_' wrong. Best way -
                // prohibit usage of '_' in param names, or change item's delimeter '_' to another one. In such case
                // this part of code should be removed
                auto excCheckLambda = [ptr, paramValueStart](const std::string& exc) {
                    size_t len = exc.length();
                    if ((ptr - len) < paramValueStart)
                        return false;
                    return exc == std::string(ptr - len, len);
                };
                auto found = std::find_if(knownExceptions.begin(), knownExceptions.end(), excCheckLambda);
                if (found != knownExceptions.end()) {
                    ptr -= found->length();
                    paramNameStart = ptr;
                    --ptr;
                } else {  // in case no underscores are found (param1=value1_param2=value2)
                    while (ptr > paramValueStart && *ptr != '_')
                        --ptr;  // we  /\ will stop here, but we need to rewind until '_' to get a valueN
                    paramNameStart = ptr + 1;
                }
                add_pair(keyValues, std::string(paramName), std::string(paramValueStart, ptr - paramValueStart));
            }
        }
    }
    // If we found no key_value parameters - just store whole line after group name as a pseudo-__name__ element
    if (keyValues.size() == 1) {
        ptr = grpNameEnd;
        while (*ptr >= 0x20)
            ++ptr;
        keyValues["__name__"] = std::string(grpNameEnd, ptr - grpNameEnd);
    }
    return true;
}

bool compile_string(const std::string& srcStr,
                    const std::map<std::string, std::string>& keyValue,
                    std::string& result) {
    size_t varPos = std::string::npos;
    size_t readPos = 0;
    std::string varName;
    result.clear();
    varName.reserve(srcStr.length());
    result.reserve(srcStr.length());
    while (readPos < srcStr.length()) {
        varPos = srcStr.find('$', readPos);
        if (varPos == std::string::npos) {
            result += srcStr.substr(readPos, srcStr.length() - readPos);
            return true;
        }
        if (varPos > readPos)
            result += srcStr.substr(readPos, varPos - readPos);
        const char *ptr = srcStr.c_str() + varPos + 1, *varNamePtr = ptr;
        while (*ptr > 0x20 && ((*ptr >= 'a' && *ptr <= 'z') || (*ptr >= 'A' && *ptr <= 'Z') ||
                               (*ptr >= '0' && *ptr <= '9') || (*ptr == '-') || (*ptr == '_')))
            ++ptr;
        varName = std::string(varNamePtr, ptr - varNamePtr);
        auto val = keyValue.find(varName);
        if (val != keyValue.end())
            result += val->second;
        readPos = varPos + (ptr - varNamePtr) + 1;
    }
    // Trim right
    while (result.length() > 1 && result[result.length() - 1] == ' ')
        result.resize(result.length() - 1);
    return readPos = srcStr.length();
}
}  // namespace PostgreSQLHelpers
}  // namespace utils
}  // namespace test
}  // namespace ov
