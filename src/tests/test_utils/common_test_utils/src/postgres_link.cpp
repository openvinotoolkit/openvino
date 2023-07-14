// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/postgres_link.hpp"

#include <gtest/gtest.h>
#include <stdlib.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

/// \brief Enables dynamic load of libpq module
#define PGQL_DYNAMIC_LOAD
/// \brief Enables extended debug messages to the stderr
#define PGQL_DEBUG
#undef PGQL_DEBUG
static const char* PGQL_ENV_CONN_NAME = "OV_POSTGRES_CONN";    // Environment variable with connection settings
static const char* PGQL_ENV_SESS_NAME = "OV_TEST_SESSION_ID";  // Environment variable identifies current session
static const char* PGQL_ENV_RUN_NAME = "OV_TEST_RUN_ID";       // Environment variable with external run id
static const char* PGQL_ENV_RLVL_NAME = "OV_TEST_REPORT_LVL";  // Environment variable identifies reporting
                                                               // level: default ("", empty), "fast", "suite"
std::map<std::string, std::string>
    ExtTestQueries;  // Map of extended test queries. It is used for do a custom query after inserting mandatory row
std::map<std::string, std::string>
    ExtTestNames;  // Map of extended test name convertors. It is used to change a test name automatically.

typedef enum {
    /// \brief Most careful reporting, but slowest
    REPORT_LVL_DEFAULT = 0,
    /// \brief Reports less information about each test case and accumulates it in joined query which will
    ///        be executed on TestSuiteEnd
    REPORT_LVL_FAST,
    /// \brief Reports only suite states, no detailed info about test cases will be available
    REPORT_LVL_SUITES_ONLY
} PostgreSQLReportingLevel;

#if !defined(_WIN32) && !defined(__APPLE__)
#    ifndef __USE_POSIX
#        define __USE_POSIX
#    endif
#    include <limits.h>
#    include <sys/utsname.h>
#    include <unistd.h>
#elif defined(__APPLE__)
#    include <sys/param.h>
#    include <sys/utsname.h>
#    include <unistd.h>
#    ifndef HOST_NAME_MAX
#        define HOST_NAME_MAX MAXHOSTNAMELEN
#    endif
#endif

#ifndef PGQL_DYNAMIC_LOAD
#    include "libpq-fe.h"
#else
#    ifdef _WIN32
#        include <Windows.h>
#    else
#        include <dlfcn.h>
typedef void* HMODULE;
#    endif
typedef enum {
    CONNECTION_OK,
    CONNECTION_BAD,
    CONNECTION_STARTED,
    CONNECTION_MADE,
    CONNECTION_AWAITING_RESPONSE,
    CONNECTION_AUTH_OK,
    CONNECTION_SETENV,
    CONNECTION_SSL_STARTUP,
    CONNECTION_NEEDED,
    CONNECTION_CHECK_WRITABLE,
    CONNECTION_CONSUME,
    CONNECTION_GSS_STARTUP,
    CONNECTION_CHECK_TARGET,
    CONNECTION_CHECK_STANDBY
} ConnStatusType;

typedef enum {
    PGRES_EMPTY_QUERY = 0,
    PGRES_COMMAND_OK,
    PGRES_TUPLES_OK,
    PGRES_COPY_OUT,
    PGRES_COPY_IN,
    PGRES_BAD_RESPONSE,
    PGRES_NONFATAL_ERROR,
    PGRES_FATAL_ERROR,
    PGRES_COPY_BOTH,
    PGRES_SINGLE_TUPLE,
    PGRES_PIPELINE_SYNC,
    PGRES_PIPELINE_ABORTED
} ExecStatusType;

struct PGconn;
struct PGresult;

typedef PGconn* (*fnPQconnectdb)(const char* conninfo);
typedef ConnStatusType (*fnPQstatus)(const PGconn* conn);
typedef size_t (*fnPQescapeStringConn)(PGconn* conn, char* to, const char* from, size_t length, int* error);
typedef void (*fnPQfinish)(PGconn* conn);
typedef char* (*fnPQerrorMessage)(const PGconn* conn);

typedef PGresult* (*fnPQexec)(PGconn* conn, const char* query);
typedef ExecStatusType (*fnPQresultStatus)(const PGresult* res);
typedef char* (*fnPQgetvalue)(const PGresult* res, int tup_num, int field_num);
typedef int (*fnPQgetisnull)(const PGresult* res, int row_number, int column_number);
typedef void (*fnPQclear)(PGresult* res);
typedef char* (*fnPQresultErrorMessage)(const PGresult* res);

static fnPQconnectdb PQconnectdb;
static fnPQescapeStringConn PQescapeStringConn;
static fnPQstatus PQstatus;
static fnPQfinish PQfinish;
static fnPQerrorMessage PQerrorMessage;

static fnPQexec PQexec;
static fnPQresultStatus PQresultStatus;
static fnPQgetvalue PQgetvalue;
static fnPQgetisnull PQgetisnull;
static fnPQclear PQclear;
static fnPQresultErrorMessage PQresultErrorMessage;
#endif

char* PGPrefix(const char* text, ::testing::internal::GTestColor color) {
    ::testing::internal::ColoredPrintf(color, text);
    return "";
}

#define PG_ERR PGPrefix("[ PG ERROR ] ", ::testing::internal::COLOR_RED)
#define PG_WRN PGPrefix("[ PG WARN  ] ", ::testing::internal::COLOR_YELLOW)
#define PG_INF PGPrefix("[ PG INFO  ] ", ::testing::internal::COLOR_GREEN)

/// \brief Count of tries when serialization error is detected after query
const uint8_t serializationTryiesCount = 30;  // Pause between each attempt is not less than 50ms

namespace CommonTestUtils {

/*
    PostgreSQL Handler class members
*/
/// \brief This manager is using for a making correct removal of PGresult object.
///        shared/unique_ptr cannot be used due to incomplete type of PGresult.
///        It is minimal implementatio which is compatible with shared/uinque_ptr
///        interface usage (reset, get)
class PGresultHolder {
    PGresult* _ptr;
    volatile uint32_t* refCounter;

    inline void decRefCounter(void) {
        if (_ptr != nullptr && refCounter != nullptr) {
            if (*refCounter > 0) {
                --*refCounter;
            }
            if (*refCounter == 0) {
                delete refCounter;
                PQclear(_ptr);
                _ptr = nullptr;
                refCounter = nullptr;
            }
        }
    }

public:
    PGresultHolder(void) : _ptr(nullptr), refCounter(nullptr) {}
    PGresultHolder(PGresult* ptr) : _ptr(ptr), refCounter(new uint32_t()) {
        *refCounter = 1;
    }
    PGresultHolder(const PGresultHolder& object) {
        _ptr = object._ptr;
        refCounter = object.refCounter;
        ++*refCounter;
    }
    PGresultHolder& operator=(const PGresultHolder& object) {
        if (_ptr != object._ptr) {
            decRefCounter();
            _ptr = object._ptr;
            refCounter = object.refCounter;
            ++*refCounter;
        }
        return *this;
    }
    void reset(PGresult* ptr) {
        if (_ptr != ptr) {
            decRefCounter();
            refCounter = new uint32_t();
            *refCounter = 1;
            _ptr = ptr;
        }
    }
    PGresult* get(void) {
        return _ptr;
    }
    ~PGresultHolder(void) {
        decRefCounter();
        _ptr = nullptr;
        refCounter = nullptr;
    }
};

/// \briaf This class implements singleton which operates with a connection to PostgreSQL server.
class PostgreSQLConnection {
#ifdef PGQL_DYNAMIC_LOAD
    std::shared_ptr<HMODULE> modLibPQ;
    bool LoadLibPQ(void);
#endif
    PGconn* activeConnection;

    PostgreSQLConnection(void) : activeConnection(nullptr), isConnected(false) {}

    /// \brief Prohobit creation outsize of class, need to make a Singleton
    PostgreSQLConnection(const PostgreSQLConnection&) = delete;
    PostgreSQLConnection& operator=(const PostgreSQLConnection&) = delete;

public:
    bool isConnected;

    static std::shared_ptr<PostgreSQLConnection> GetInstance(void);
    bool Initialize(void);
    /// \brief Make a common query to a server. Result will be returned as self-desctructable pointer. But application
    /// should check result pointer isn't a nullptr. And result status by itself. \param[in] query SQL query to a server
    /// \returns Object which keep pointer on received PGresult. It contains nullptr in case of any error.
    PGresultHolder CommonQuery(const char* query) {
#ifdef PGQL_DEBUG
        std::cerr << query << std::endl;
#endif
        if (!isConnected)
            return PGresultHolder();
        PGresultHolder result(PQexec(this->activeConnection, query));
        // Connection could be closed by a timeout, we may try to reconnect once.
        // We don't reconnect on each call because it may make testing significantly slow in
        // case of connection issues. Better to finish testing with incomplete results and
        // free a machine. Otherwise we will lose all results.
        if (result.get() == nullptr) {
            TryReconnect();
            // If reconnection attempt was successfull - let's try to set new query
            if (isConnected) {
                result.reset(PQexec(this->activeConnection, query));
            }
        }
        if (result.get() == nullptr) {
            std::cerr << PG_ERR << "Error while querying PostgreSQL\n";
        }
        return result;
    }

    /// \brief Queries a server. Result will be returned as self-desctructable pointer. But application should check
    /// result pointer isn't a nullptr.
    /// \param[in] query SQL query to a server
    /// \param[in] expectedStatus Query result will be checked for passed status, if it isn't equal - result pointer
    /// will be nullptr. \returns Object which keep pointer on received PGresult. It contains nullptr in case of any
    /// error.
    PGresultHolder Query(const char* query, const ExecStatusType expectedStatus = PGRES_TUPLES_OK) {
        PGresultHolder result = CommonQuery(query);
        uint8_t queryCounter = 1;

        while (result.get() != nullptr && queryCounter < serializationTryiesCount) {
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
            if (errStr.find("could not serialize access") != std::string::npos) {
                std::cerr << PG_WRN << "Serialization error, trying again, try attempt: " << queryCounter++
                          << std::endl;
#ifdef _WIN32
                Sleep(50);  // Wait some time for the next attempt
#else
                usleep(50000);
#endif
            } else {
                std::cerr << PG_ERR << "Error message: " << errStr << std::endl;
                result.reset(nullptr);
            }
        }
        if (queryCounter >= serializationTryiesCount) {
            std::cerr << PG_ERR << "Cannot execute query due to serialization error, failing" << std::endl;
            result.reset(nullptr);
        }
        return result;
    }

    /// \brief Tries to reconnect in case of connection issues (usual usage - connection timeout).
    void TryReconnect(void) {
        if (!isConnected) {
            return;
        }
        if (activeConnection != nullptr) {
            try {
                PQfinish(activeConnection);
            } catch (...) {
                std::cerr << PG_ERR << "An exception while finishing PostgreSQL connection\n";
            }
            this->activeConnection = nullptr;
            this->isConnected = false;
        }
        std::cerr << PG_INF << "Reconnecting to the PostgreSQL server...\n";
        Initialize();
    }

    PGconn* GetConnection(void) const {
        return this->activeConnection;
    }
    ~PostgreSQLConnection(void);
};

static std::shared_ptr<PostgreSQLConnection> connection(nullptr);
std::shared_ptr<PostgreSQLConnection> PostgreSQLConnection::GetInstance(void) {
    if (connection.get() == nullptr) {
        connection.reset(new PostgreSQLConnection());
    }
    return connection;
}

PostgreSQLConnection::~PostgreSQLConnection(void) {
    if (activeConnection) {
        PQfinish(this->activeConnection);
        this->activeConnection = nullptr;
        this->isConnected = false;
    }
}

/// \brief Initialization of exact object. Uses environment variable PGQL_ENV_CONN_NAME for making a connection.
/// \returns Returns false in case of failure or absence of ENV-variable.
///          Returns true in case of connection has been succesfully established.
bool PostgreSQLConnection::Initialize(void) {
    if (this->activeConnection != nullptr) {
        std::cerr << PG_WRN << "PostgreSQL connection is already established.\n";
        return true;
    }

#ifdef PGQL_DYNAMIC_LOAD
    if (!LoadLibPQ()) {
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

    this->activeConnection = PQconnectdb(envConnString);

    ConnStatusType connStatus = PQstatus(this->activeConnection);

    if (connStatus != CONNECTION_OK) {
        std::cerr << PG_ERR << "Cannot connect to PostgreSQL: " << static_cast<uint32_t>(connStatus) << std::endl;
        return false;
    } else {
        std::cerr << PG_INF << "Connected to PostgreSQL successfully\n";
    }

    this->isConnected = true;

    return true;
}

#ifdef PGQL_DYNAMIC_LOAD
/// \brief Loads libpq module in runtime
bool PostgreSQLConnection::LoadLibPQ(void) {
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

/// \brief This method is used for parsing serialized value_param string.
///        Known limitations:
///        It doesn't read values in inner tuples/arrays/etc.
static std::vector<std::string> ParseValueParam(std::string text) {
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

/// \brief Function returns OS version in runtime.
/// \returns String which contains OS version
static std::string GetOSVersion(void) {
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

/// \brief Function returns executable name of current application.
/// \returs File name as a std::string
static std::string GetExecutableName(void) {
#ifdef _WIN32
    char cFilePath[MAX_PATH] = {};
    GetModuleFileName(nullptr, cFilePath, MAX_PATH);
    std::string filePath(cFilePath);
#else
    std::string filePath;
    std::ifstream("/proc/self/comm") >> filePath;
    return filePath;
#endif
    return filePath.substr(filePath.find_last_of("/\\") + 1);
}

/// \brief Cross-platform implementation of getting host name
/// \returns String with host name or "NOT_FOUND" in case of error
static std::string GetHostname(void) {
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
}

// Procedure uses for possible customization of addint key=value pairs
static void addPair(std::map<std::string, std::string>& keyValues, const std::string& key, const std::string& value) {
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
    // Parse IR for opName and hash
    if (key == "IR" && (dPos = value.find('_')) != std::string::npos) {
        keyValues["hashXml"] = value.substr(dPos + 1);
        keyValues["pathXml"] = value;
        return;
    }
    // Parse Function for opName and opSet
    if (key == "Function" && (dPos = value.find('_')) != std::string::npos) {
        keyValues["opName"] = value.substr(0, dPos);
        keyValues["opSet"] = value.substr(dPos + 1);
        return;
    }
    // Normalize target devices
    if (key == "target_device" || key == "TargetDevice" || key == "Device") {
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

// Function parses test name for key=value pairs
static bool parseTestName(const char* line, std::map<std::string, std::string>& keyValues) {
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
                addPair(keyValues, std::string(paramName), std::string(paramValueStart, ptr - paramValueStart));
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
                addPair(keyValues, std::string(paramName), std::string(paramValueStart, ptr - paramValueStart));
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

/// \brief Compiles string and replaces variables defined as $[0-9A-Za-z-_] by values from
/// provided key=value map. Replaces by blank in case of key isn't found in the map.
/// \param[in] srcStr String contains variables
/// \param[in] keyValue Key=value map with variable values
/// \param[out] result String for result
/// \returns Returns true if all input string was compiled, false in case of any compilation error
static bool compileString(const std::string& srcStr,
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

/// \brief Helper for checking PostgreSQL results
#define CHECK_PGRESULT(var_name, error_message, action)    \
    if (var_name.get() == nullptr) {                       \
        std::cerr << PG_ERR << error_message << std::endl; \
        action;                                            \
    }
/// \brief Helper for registering ID-retrieval functions
#define GET_PG_IDENTIFIER(funcDefinition, sqlQuery, varName, fieldName) \
    bool funcDefinition {                                               \
        std::stringstream sstr;                                         \
        sstr << sqlQuery;                                               \
        return _internalRequestId(sstr.str(), #fieldName, varName);     \
    }

/// \brief Class which handles gtest keypoints and send data to PostgreSQL database.
///        May be separated for several source files in case it'll become to huge.
class PostgreSQLEventListener : public ::testing::EmptyTestEventListener {
    std::shared_ptr<PostgreSQLConnection> connectionKeeper;

    const char* session_id = nullptr;  // String value of session id, it will be converted to sessinoId
    const char* run_id = nullptr;      // String value of run (might be char[33]), it will be converted to testRunId
    bool isPostgresEnabled = false;
    PostgreSQLReportingLevel reportingLevel = REPORT_LVL_DEFAULT;

    /* Dynamic information about current session*/
    uint64_t sessionId = 0;
    uint64_t appId = 0;
    uint64_t hostId = 0;
    uint64_t testIterationId = 0;
    uint64_t testSuiteNameId = 0;
    uint64_t testNameId = 0;
    uint64_t testSuiteId = 0;
    uint64_t testId = 0;
    uint64_t testRunId = 0;
    uint64_t testExtId = 0;

    size_t jqLastCommandOffset;  // Stores offset of beginning of a last command, to be able to rewind
    std::stringstream joinedQuery;
    bool isRefusedResult = false;  // Signals test result is a waste and shouldn't be stored in DB

    /* Test name parsing */
    std::map<std::string, std::string>
        testDictionary;  // Contains key=value pairs which should be used while constructing queries and names
    std::string testName;
    bool isTestNameParsed = false;  // Signals test name was successfully parsed to the testDictionary pairs
    bool isFieldsUpdated = false;   // Signals testDictionary was updated between OnTestStart and OnTestEnd

    // Unused event handlers, kept here for possible use in the future
    /*
    void OnTestProgramStart(const ::testing::UnitTest& unit_test) override {}
    void OnTestIterationStart(const ::testing::UnitTest& unit_test, int iteration) override {}
    void OnEnvironmentsSetUpStart(const ::testing::UnitTest& unit_test) override {}
    void OnEnvironmentsSetUpEnd(const ::testing::UnitTest& unit_test) override {}
    void OnEnvironmentsTearDownStart(const ::testing::UnitTest& unit_test) override {}
    void OnEnvironmentsTearDownEnd(const ::testing::UnitTest& unit_test) override {}
    void OnTestIterationEnd(const ::testing::UnitTest& unit_test, int iteration) override {}
    void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override {}
    */

    /*
        Transaction-optimized ID retrieval.
        If string doesn't begin with "BEGIN" and contain "SELECT" regular flow is used.
        Otherwise - tries to move pointer to start of "SELECT" part of query and queries ID first.
        In case query has returned NULL - means no corresponding data found in database.
        If no information found - calls transactional insert and retrieve.
        sqlQuery = "BEGIN TRANSACTION ... COMMIT; SELECT QUERY";
                    ^ full query will be executed ^ only if this part has returned NULL
        Using described approach we do transactional queries only in case we don't
        retrieve correct data from database. Otherwise it will use fast path without
        transaction.
    */
    bool _internalRequestId(const std::string& sqlQuery, const char* fieldName, uint64_t& result) {
        auto selectPos = sqlQuery.find("SELECT");
        const char *query = sqlQuery.c_str(), *query_start = query;
        bool isTransactionalQuery = (selectPos != std::string::npos) && (sqlQuery.find("BEGIN") == 0);
        if (isTransactionalQuery) {
            query += selectPos;
        }
        auto pgresult = connectionKeeper->Query(query);
        CHECK_PGRESULT(pgresult, "Cannot retrieve a correct " << fieldName, return false);

        bool isNull = PQgetisnull(pgresult.get(), 0, 0) != 0;
        if (isNull && isTransactionalQuery) {
            pgresult = connectionKeeper->Query(query_start);
            CHECK_PGRESULT(pgresult, "Cannot retrieve a correct transactional " << fieldName, return false);
            isNull = PQgetisnull(pgresult.get(), 0, 0) != 0;
        }

        if (!isNull && (result = std::atoi(PQgetvalue(pgresult.get(), 0, 0))) == 0) {
            std::cerr << PG_ERR << "Cannot interpret a returned " << fieldName
                      << ", value : " << (!isNull ? PQgetvalue(pgresult.get(), 0, 0) : "NULL") << std::endl;
            return false;
        }
        return true;
    }

    /// \brief Escapes control symbols in a string by using PostgreSQL escapeStringConn function
    /// \returns Return escaped string or throws an error in case of any error
    std::string EscapeString(const std::string& sourceString) const {
        if (sourceString.length() == 0) {
            return std::string("");
        }

        std::vector<char> escapedString;
        escapedString.resize(sourceString.length() * 2);  // Doc requires to allocate two times more than initial length
        escapedString[0] = 0;
        int errCode = 0;
        size_t writtenSize = 0;
        writtenSize = PQescapeStringConn(connectionKeeper->GetConnection(),
                                         escapedString.data(),
                                         sourceString.c_str(),
                                         sourceString.length(),
                                         &errCode);
        if (errCode == 0 && writtenSize >= sourceString.length()) {
            return std::string(escapedString.data());
        } else {
            throw std::runtime_error("Error while escaping string");
        }
    }

    GET_PG_IDENTIFIER(RequestApplicationId(void),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_APPLICATION('" << GetExecutableName() << "');"
                          << "COMMIT; "
                          << "SELECT GET_APPLICATION('" << GetExecutableName() << "');",
                      appId,
                      app_id)
    GET_PG_IDENTIFIER(RequestRunId(void),
                      "SELECT GET_RUN('" << this->run_id << "', " << this->appId << ", " << this->sessionId << ", "
                                         << this->hostId << ", " << static_cast<uint16_t>(this->reportingLevel)
                                         << "::smallint);",
                      testRunId,
                      run_id)
    GET_PG_IDENTIFIER(RequestHostId(void),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_HOST('" << GetHostname() << "', '" << GetOSVersion() << "');"
                          << "COMMIT; "
                          << "SELECT GET_HOST('" << GetHostname() << "', '" << GetOSVersion() << "');",
                      hostId,
                      host_info_id)
    GET_PG_IDENTIFIER(RequestSessionId(void),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_SESSION('" << this->session_id << "');"
                          << "COMMIT;"
                          << "SELECT GET_SESSION('" << this->session_id << "');",
                      sessionId,
                      session_id)
    GET_PG_IDENTIFIER(RequestSuiteNameId(const char* test_suite_name),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_TEST_SUITE('" << test_suite_name << "', " << this->appId << ");"
                          << "COMMIT; "
                          << "SELECT GET_TEST_SUITE('" << test_suite_name << "', " << this->appId << ");",
                      testSuiteNameId,
                      sn_id)
    GET_PG_IDENTIFIER(RequestTestNameId(std::string query),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_" << query << "; COMMIT;"
                          << "SELECT GET_" << query,
                      testNameId,
                      tn_id)
    GET_PG_IDENTIFIER(RequestSuiteId(void),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_SUITE_ID(" << this->testSuiteNameId << ", " << this->sessionId << ", "
                          << this->testRunId << ");"
                          << "COMMIT; "
                          << "SELECT GET_SUITE_ID(" << this->testSuiteNameId << ", " << this->sessionId << ", "
                          << this->testRunId << ");",
                      testSuiteId,
                      sr_id)
    GET_PG_IDENTIFIER(RequestSuiteId(std::string query), query, testSuiteId, sr_id)
    GET_PG_IDENTIFIER(RequestTestId(std::string query), query, testId, tr_id)
    GET_PG_IDENTIFIER(RequestTestExtId(std::string query),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_" << query << "; COMMIT;"
                          << "SELECT ON_START_" << query,
                      testExtId,
                      t_id)
    GET_PG_IDENTIFIER(UpdateTestExtId(std::string query),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_END_MISS_" << query << "; COMMIT;"
                          << "SELECT ON_END_" << query,
                      testExtId,
                      t_id)

    /// \brief Send an update query. Depends on mode in calls specific UPDATE query in case of default reporting
    /// and regular ON_START query in case of fast reporting.
    /// In such case UPDATE query is used only for updating changed values in runtime, when
    /// ON_START query in this place already has all updated data, that's why UPDATE query is skipping
    void UpdateTestResults(const ::testing::TestInfo* test_info = nullptr) {
        auto grpName = testDictionary.end();

        if (isTestNameParsed == true && (reportingLevel == REPORT_LVL_FAST || isFieldsUpdated == true) &&
            (grpName = testDictionary.find("__groupName__")) != testDictionary.end()) {
            auto extQuery = ExtTestQueries.end();
            if (reportingLevel == REPORT_LVL_DEFAULT) {
                // Looks query with GroupName + "_ON_END", "ReadIR_ON_END" as example
                extQuery = ExtTestQueries.find(grpName->second + "_ON_END");
            } else if (reportingLevel == REPORT_LVL_FAST) {
                // Looks query with GroupName + "_ON_START", "ReadIR_ON_START" as example
                extQuery = ExtTestQueries.find(grpName->second + "_ON_START");
            }
            if (extQuery != ExtTestQueries.end()) {
                std::string query;
                if (compileString(extQuery->second, testDictionary, query)) {
                    if (reportingLevel == REPORT_LVL_DEFAULT) {
                        if (!UpdateTestExtId(query)) {
                            std::cerr << PG_WRN << "Failed extended update query: " << query << std::endl;
                        } else {
                            isFieldsUpdated = false;
                        }
                    } else if (reportingLevel == REPORT_LVL_FAST) {
                        size_t jqLen = joinedQuery.tellp();
                        std::vector<char> addQuery(jqLen - jqLastCommandOffset);
                        joinedQuery.seekg(jqLastCommandOffset);
                        joinedQuery.read(addQuery.data(),
                                         jqLen - jqLastCommandOffset - 3);  // Ignores ");\n" at the end
                        joinedQuery.seekp(jqLastCommandOffset);
                        joinedQuery << "WITH rows AS (" << addQuery.data() << ") AS test_id) SELECT ON_START_" << query
                                    << " FROM rows;\n";
                    }
                } else {
                    std::cerr << PG_WRN << "Preparing extended update query is failed: "
                              << (test_info != nullptr ? test_info->name() : "[no test info]") << std::endl;
                }
            }
        }
    }

    void OnTestSuiteStart(const ::testing::TestSuite& test_suite) override {
        if (!this->isPostgresEnabled || !this->testRunId || !this->sessionId)
            return;
        try {
            if (!RequestSuiteNameId(EscapeString(test_suite.name()).c_str()))
                return;
        } catch (const std::exception& e) {
            std::cerr << PG_ERR << "Requesting suite name is failed with exception: " << e.what() << std::endl;
            return;
        }
        /*
        // This part of code left because it is preferred way, but incompatible with external parallel running
        std::stringstream sstr;
        sstr << "INSERT INTO suite_results (sr_id, session_id, run_id, suite_id) VALUES (DEFAULT, " << this->sessionId
             << ", " << this->testRunId << ", " << this->testSuiteNameId << ") RETURNING sr_id";
        if (!RequestSuiteId(sstr.str()))
            return;
        */
        if (!RequestSuiteId())
            return;

        // Cleanup accumulator for quieries
        if (reportingLevel == REPORT_LVL_FAST) {
            jqLastCommandOffset = 0;
            joinedQuery.str("");
            joinedQuery.clear();
        }
    }

//  Legacy API is deprecated but still available
#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
    void OnTestCaseStart(const ::testing::TestCase& test_case) override {
        if (this->testSuiteNameId == 0)
            OnTestSuiteStart(test_case);
    }
#endif  //  GTEST_REMOVE_LEGACY_TEST_CASEAPI_

    void OnTestStart(const ::testing::TestInfo& test_info) override {
        if (!this->isPostgresEnabled || !this->testRunId || !this->sessionId || !this->testSuiteNameId ||
            !this->testSuiteId)
            return;

        if (reportingLevel == REPORT_LVL_SUITES_ONLY) {
            // Do not report per-case results
            return;
        }

        isRefusedResult = false;
        testDictionary.clear();

        auto grpName = testDictionary.end();
        testName = test_info.name();

        if ((isTestNameParsed = parseTestName(test_info.name(), testDictionary)) == true &&
            (grpName = testDictionary.find("__groupName__")) != testDictionary.end()) {
            auto nameConvertStr = ExtTestNames.find(grpName->second);
            if (nameConvertStr != ExtTestNames.end()) {
                if (!compileString(nameConvertStr->second, testDictionary, testName)) {
                    std::cerr << PG_WRN << "Error compiling test name: " << test_info.name() << std::endl;
                    testName = grpName->second;
                }
            } else {
                testName = grpName->second;
            }
        } else {
            std::cerr << PG_WRN << "Error parsing test name: " << test_info.name() << std::endl;
        }

        std::stringstream sstr;
        try {
            if (reportingLevel == REPORT_LVL_DEFAULT) {
                sstr << "TEST_NAME(" << this->testSuiteNameId << ", '" << EscapeString(testName) << "'";
            } else if (reportingLevel == REPORT_LVL_FAST) {
                sstr << "SELECT ADD_TEST_RESULT(" << this->appId << ", " << this->sessionId << ", " << this->testRunId
                     << ", " << this->testSuiteId << ", " << this->testSuiteNameId << ", '" << EscapeString(testName)
                     << "'";
            }
        } catch (std::exception& e) {
            std::cerr << PG_ERR << "Query building is failed with exception: " << e.what() << std::endl;
            return;
        }

        {
            /* If we need a query customization - it could be done here. */
            sstr << ", NULL";
        }

        testDictionary["__suite_id"] = std::to_string(this->testSuiteId);
        testDictionary["__suite_name_id"] = std::to_string(this->testSuiteNameId);
        testDictionary["__session_id"] = std::to_string(this->testId);
        testDictionary["__run_id"] = std::to_string(this->testId);
        testDictionary["__is_temp"] = (reportingLevel == REPORT_LVL_DEFAULT ? "1::boolean" : "0::boolean");
        isFieldsUpdated = false;

        if (reportingLevel == REPORT_LVL_DEFAULT) {
            sstr << ")";
        } else if (reportingLevel == REPORT_LVL_FAST) {
            jqLastCommandOffset = joinedQuery.tellp();
            joinedQuery << sstr.str();
            // This will allow to pass checks later, but this values isn't expected to see in real
            this->testNameId = ~0;
            this->testId = ~0;
            // In case of fast reporting __test_id will be unset because of lazy execution.
            // It should be replaced in WITH ... AS (... AS test_id) SELECT ... construction to test_id
            testDictionary["__test_id"] = "test_id";
            testDictionary["__test_ext_id"] = "test_id";
            return;
        }

        if (!RequestTestNameId(sstr.str()))
            return;

        sstr.str("");
        sstr.clear();
        // Creates temporary record
        sstr << "INSERT INTO test_results_temp (tr_id, session_id, suite_id, run_id, test_id) VALUES (DEFAULT, "
             << this->sessionId << ", " << this->testSuiteId << ", " << this->testRunId << ", " << this->testNameId
             << ") RETURNING tr_id";

        if (!RequestTestId(sstr.str()))
            return;

        if (grpName != testDictionary.end()) {
            // Looks query with GroupName + "_ON_START", "ReadIR_ON_START" as example
            auto extQuery = ExtTestQueries.find(grpName->second + "_ON_START");
            if (extQuery != ExtTestQueries.end()) {
                std::string query;
                testDictionary["__test_id"] = std::to_string(this->testId);
                if (compileString(extQuery->second, testDictionary, query)) {
                    if (!RequestTestExtId(query)) {
                        std::cerr << PG_WRN << "Failed extended query: " << query << std::endl;
                    } else {
                        testDictionary["__test_ext_id"] = std::to_string(this->testExtId);
                    }
                } else {
                    std::cerr << PG_WRN << "Preparing extended query is failed: " << test_info.name() << std::endl;
                }
            }
        }
    }

    void OnTestPartResult(const ::testing::TestPartResult& test_part_result) override {
        if (!this->isPostgresEnabled || !this->testRunId || !this->sessionId || !this->testSuiteNameId ||
            !this->testSuiteId || !this->testNameId || !this->testId)
            return;

        if (reportingLevel == REPORT_LVL_SUITES_ONLY) {
            return;
        }

        if (isRefusedResult) {
            return;
        }
    }

    void OnTestEnd(const ::testing::TestInfo& test_info) override {
        if (!this->isPostgresEnabled || !this->testRunId || !this->sessionId || !this->testSuiteNameId ||
            !this->testSuiteId || !this->testNameId || !this->testId)
            return;

        if (reportingLevel == REPORT_LVL_SUITES_ONLY) {
            return;
        }

        if (isRefusedResult) {
            if (reportingLevel == REPORT_LVL_DEFAULT) {
                auto grpName = testDictionary.end();

                if (isTestNameParsed == true && isFieldsUpdated == true &&
                    (grpName = testDictionary.find("__groupName__")) != testDictionary.end()) {
                    // Looks query with GroupName + "_ON_REFUSE", "ReadIR_ON_REFUSE" as example
                    auto extQuery = ExtTestQueries.find(grpName->second + "_ON_REFUSE");
                    if (extQuery != ExtTestQueries.end()) {
                        std::string query;
                        if (compileString(extQuery->second, testDictionary, query)) {
                            auto pgresult =
                                connectionKeeper->Query((std::string("CALL ON_REFUSE_") + query).c_str(), PGRES_COMMAND_OK);
                            CHECK_PGRESULT(pgresult, "Cannot remove extended waste results", /* no return */);
                        } else {
                            std::cerr << PG_WRN
                                      << "Preparing extended waste cleanup query is failed: " << test_info.name()
                                      << std::endl;
                        }
                    }
                }

                // Remove temporary record
                std::stringstream sstr;
                sstr << "DELETE FROM test_results_temp WHERE tr_id=" << this->testId;
                auto pgresult = connectionKeeper->Query(sstr.str().c_str(), PGRES_COMMAND_OK);
                CHECK_PGRESULT(pgresult, "Cannot remove waste results", return);

                this->testId = 0;
                testDictionary.clear();
            } else if (reportingLevel == REPORT_LVL_FAST) {
                // Rewind to a last command
                std::string tmp = joinedQuery.str();
                tmp.resize(jqLastCommandOffset);
                joinedQuery.str(tmp);
            }
            return;
        }

        uint32_t testResult = 0;
        if (test_info.result()->Passed())
            testResult = 1;
        else if (test_info.result()->Skipped())
            testResult = 2;

        if (reportingLevel == REPORT_LVL_DEFAULT) {
            std::stringstream sstr;
            uint64_t testTempId = this->testId;
            sstr << "INSERT INTO test_results (session_id, suite_id, run_id, test_id, started_at, finished_at, "
                    "duration, test_result) "
                 << "(SELECT session_id, suite_id, run_id, test_id, started_at, NOW(), "
                 << test_info.result()->elapsed_time() << ", " << testResult << "::smallint FROM test_results_temp "
                 << "WHERE tr_id=" << this->testId << ") RETURNING tr_id";
            if (!RequestTestId(sstr.str())) {
                return;
            } else {
                sstr.str("");
                sstr.clear();
                sstr << "DELETE FROM test_results_temp WHERE tr_id=" << testTempId;
                auto pgresult = connectionKeeper->Query(sstr.str().c_str(), PGRES_COMMAND_OK);
                CHECK_PGRESULT(pgresult, "Cannot remove temporary test results", /* no return */);

                // Set correct information about
                SetCustomField("__test_ext_id", std::to_string(this->testId), true);
                SetCustomField("__test_id", std::to_string(testTempId), true);
                // Force updating fields because in some conditions IDs might be equal
                isFieldsUpdated = true;
            }
        } else if (reportingLevel == REPORT_LVL_FAST) {
            joinedQuery << ", " << testResult << "::smallint, " << test_info.result()->elapsed_time() << ");\n";
        }

        UpdateTestResults(&test_info);

        this->testId = 0;
        testDictionary.clear();
    }

    void OnTestSuiteEnd(const ::testing::TestSuite& test_suite) override {
        if (!this->isPostgresEnabled || !this->testRunId || !this->sessionId || !this->testSuiteNameId ||
            !this->testSuiteId)
            return;

        std::stringstream sstr;
        sstr << "UPDATE suite_results SET finished_at=NOW(), duration=" << test_suite.elapsed_time()
             << ", suite_result=" << (test_suite.Passed() ? 1 : 0)
             << ", successful_count=" << test_suite.successful_test_count()
             << ", skipped_count=" << test_suite.skipped_test_count()
             << ", failed_count=" << test_suite.failed_test_count()
             << ", disabled_count=" << test_suite.disabled_test_count()
             << ", run_count=" << test_suite.test_to_run_count() << ", total_count=" << test_suite.total_test_count()
             << " WHERE sr_id=" << this->testSuiteId;
        auto pgresult = connectionKeeper->Query(sstr.str().c_str(), PGRES_COMMAND_OK);
        CHECK_PGRESULT(pgresult, "Cannot update test suite results", return);
        this->testSuiteId = 0;

        if (reportingLevel == REPORT_LVL_FAST) {
            pgresult = connectionKeeper->Query(joinedQuery.str().c_str());
            CHECK_PGRESULT(pgresult, "Cannot update test cases results", return);
        }
    }
#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
    void OnTestCaseEnd(const ::testing::TestCase& test_case) override {
        if (this->testSuiteId != 0)
            OnTestSuiteEnd(test_case);
    }
#endif  //  GTEST_REMOVE_LEGACY_TEST_CASEAPI_

    /* Do nothing here. If you need to do anything on creation - it should be fully undersandable. */
    PostgreSQLEventListener() {
        this->session_id = std::getenv(PGQL_ENV_SESS_NAME);
        if (this->session_id != nullptr) {
            isPostgresEnabled = false;

            char* env_report_lvl = std::getenv(PGQL_ENV_RLVL_NAME);
            if (env_report_lvl == nullptr) {
                reportingLevel = REPORT_LVL_DEFAULT;
                std::cerr << PG_INF << "Default reporting level is using\n";
            } else if (strcmp(env_report_lvl, "fast") == 0) {
                reportingLevel = REPORT_LVL_FAST;
                std::cerr << PG_INF << "Fast reporting level is using\n";
            } else if (strcmp(env_report_lvl, "suite") == 0) {
                reportingLevel = REPORT_LVL_SUITES_ONLY;
                std::cerr << PG_INF << "Suites-only reporting level is using\n";
            } else {
                reportingLevel = REPORT_LVL_DEFAULT;
                std::cerr << PG_WRN << "Wrong reporting level is passed, default reporting level is using\n";
            }

            this->run_id = std::getenv(PGQL_ENV_RUN_NAME);
            if (this->run_id != nullptr) {
                std::cerr << PG_INF << "External Run ID is provided: " << this->run_id << std::endl;
            } else {
                // Run id will be generated on database side, each run unique
                this->run_id = "";
            }

            std::cerr << PG_INF << "Test session ID has been found\n";
            connectionKeeper = PostgreSQLConnection::GetInstance();
            bool connInitResult = connectionKeeper->Initialize();

            if (!connInitResult)
                return;

            isPostgresEnabled = connInitResult;

            if (isPostgresEnabled)
                isPostgresEnabled &= RequestApplicationId();
            if (isPostgresEnabled)
                isPostgresEnabled &= RequestHostId();
            if (isPostgresEnabled)
                isPostgresEnabled &= RequestSessionId();
            if (isPostgresEnabled)
                isPostgresEnabled &= RequestRunId();

            if (isPostgresEnabled) {
                connectionKeeper = connection;
            }
        } else {
            std::cerr << PG_ERR << "Test session ID hasn't been found, continues without database reporting\n";
        }
    }

    ~PostgreSQLEventListener(void) {
        if (!this->isPostgresEnabled)
            return;

        std::stringstream sstr;
        sstr << "UPDATE runs SET end_time=NOW() WHERE run_id=" << this->testRunId << " AND end_time<NOW()";
        auto pgresult = connectionKeeper->Query(sstr.str().c_str(), PGRES_COMMAND_OK);
        CHECK_PGRESULT(pgresult, "Cannot update run finish info", return);

        sstr.str("");
        sstr.clear();
        sstr << "UPDATE sessions SET end_time=NOW() WHERE session_id=" << this->sessionId << " AND end_time<NOW()";
        pgresult = connectionKeeper->Query(sstr.str().c_str(), PGRES_COMMAND_OK);
        CHECK_PGRESULT(pgresult, "Cannot update session finish info", return);
    }

    /* Prohobit creation outsize of class, need to make a Singleton */
    PostgreSQLEventListener(const PostgreSQLEventListener&) = delete;
    PostgreSQLEventListener& operator=(const PostgreSQLEventListener&) = delete;

    friend class PostgreSQLEnvironment;

public:
    void SetRefuseResult(bool value = true) {
        isRefusedResult = value;
    }

    bool SetCustomField(const std::string fieldName, const std::string fieldValue, const bool rewrite) {
        auto field = this->testDictionary.find(fieldName);
        if (rewrite || field == this->testDictionary.end()) {
            isFieldsUpdated |= (field == this->testDictionary.end()) ||
                               (field->second != fieldValue);  // Signals only in case value not equal with existing
            this->testDictionary[fieldName] = fieldValue;
            return true;
        }
        return false;
    }

    std::string GetCustomField(const std::string fieldName, const std::string defaultValue) const {
        auto field = this->testDictionary.find(fieldName);
        if (field != this->testDictionary.end()) {
            return field->second;
        }
        return defaultValue;
    }

    bool RemoveCustomField(const std::string fieldName) {
        auto field = this->testDictionary.find(fieldName);
        if (field != this->testDictionary.end()) {
            this->testDictionary.erase(field);
            isFieldsUpdated = true;
            return true;
        }
        return false;
    }
};

/// \brief Global variable (scoped only to this file) which contains pointer to PostgreSQLEventListener
///        Might be replaced by a bool, but left as is for possible future use.
static PostgreSQLEventListener* pgEventListener = nullptr;

/// \brief Class is used for registering environment handler in gtest. It prepares in-time set up
///        for registering PostgreSQLEventListener
class PostgreSQLEnvironment : public ::testing::Environment {
public:
    PostgreSQLEnvironment(void) {}
    ~PostgreSQLEnvironment(void) {}
    void SetUp(void) override {
        if (std::getenv(PGQL_ENV_SESS_NAME) != nullptr && std::getenv(PGQL_ENV_CONN_NAME) != nullptr) {
            if (pgEventListener == nullptr) {
                pgEventListener = new PostgreSQLEventListener();
                ::testing::UnitTest::GetInstance()->listeners().Append(pgEventListener);
            }
        } else {
            std::cerr << PG_INF << "PostgreSQL Reporting is disabled due to missing environment settings\n";
        }
    }
    void TearDown(void) override {
        // Don't see any reason to do additional tear down
    }
};

/// \brief Global variable which stores a pointer to active instance (only one is expected) of
/// PostgreSQLEnvironment.
::testing::Environment* PostgreSQLEnvironment_Reg = ::testing::AddGlobalTestEnvironment(new PostgreSQLEnvironment());

/// \brief This class is for internal usage, don't need to move it to the header. It holds an internal state of
///        PostgreSQLLink instance. Introduced to simplify header.
class PostgreSQLCustomData {
    // Reserved place for storing temporary data
};

PostgreSQLLink::PostgreSQLLink() : parentObject(nullptr), customData(new PostgreSQLCustomData()) {
#ifdef PGQL_DEBUG
    std::cout << PG_INF << "PostgreSQLLink Started\n";
#endif
}

PostgreSQLLink::PostgreSQLLink(void* ptrParentObject)
    : parentObject(ptrParentObject),
      customData(new PostgreSQLCustomData()) {
#ifdef PGQL_DEBUG
    std::cout << PG_INF << "PostgreSQLLink with parentObject Started\n";
#endif
}

PostgreSQLLink::~PostgreSQLLink(void) {
    if (this->customData) {
        delete this->customData;
        this->customData = nullptr;
    }

    this->parentObject = nullptr;
#ifdef PGQL_DEBUG
    std::cout << PG_INF << "PostgreSQLLink Finished\n";
#endif
}

std::map<std::string, std::string>* PostgreSQLLink::getExtTestQueries(void) {
    return &ExtTestQueries;
}

std::map<std::string, std::string>* PostgreSQLLink::getExtTestNames(void) {
    return &ExtTestNames;
}

void PostgreSQLLink::SetRefuseResult(bool value) const {
    if (pgEventListener) {
        pgEventListener->SetRefuseResult(value);
    }
}

bool PostgreSQLLink::SetCustomField(const std::string fieldName,
                                    const std::string fieldValue,
                                    const bool rewrite) const {
    if (pgEventListener) {
        return pgEventListener->SetCustomField(fieldName, fieldValue, rewrite);
    }
    return false;
}

std::string PostgreSQLLink::GetCustomField(const std::string fieldName, const std::string defaultValue) const {
    if (pgEventListener) {
        return pgEventListener->GetCustomField(fieldName, defaultValue);
    }
    return defaultValue;
}

bool PostgreSQLLink::RemoveCustomField(const std::string fieldName) const {
    if (pgEventListener) {
        pgEventListener->RemoveCustomField(fieldName);
    }
    return false;
}

}  // namespace CommonTestUtils
namespace PostgreSQLLink {
std::map<std::string, std::string>* getExtTestQueries(void) {
    return ::CommonTestUtils::PostgreSQLLink::getExtTestQueries();
}

std::map<std::string, std::string>* getExtTestNames(void) {
    return ::CommonTestUtils::PostgreSQLLink::getExtTestNames();
}
}  // namespace PostgreSQLLink
