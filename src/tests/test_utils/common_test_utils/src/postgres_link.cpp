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
#include <pugixml.hpp>
#include <sstream>

/// \brief Enables dynamic load of libpq module
#define PGQL_DYNAMIC_LOAD
/// \brief Enables extended debug messages to the stderr
#define PGQL_DEBUG
#undef PGQL_DEBUG
static const char* PGQL_ENV_CONN_NAME = "OV_POSTGRES_CONN";    // Environment variable with connection settings
static const char* PGQL_ENV_SESS_NAME = "OV_TEST_SESSION_ID";  // Environment variable identifies current session

#if !defined(_WIN32) && !defined(__APPLE__)
#    ifndef __USE_POSIX
#        define __USE_POSIX
#    endif
#    include <limits.h>
#    include <sys/utsname.h>
#elif defined(__APPLE__)
#    include <sys/param.h>
#    include <sys/utsname.h>
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

typedef PGresult* (*fnPQexec)(PGconn* conn, const char* query);
typedef ExecStatusType (*fnPQresultStatus)(const PGresult* res);
typedef char* (*fnPQgetvalue)(const PGresult* res, int tup_num, int field_num);
typedef void (*fnPQclear)(PGresult* res);

static fnPQconnectdb PQconnectdb;
static fnPQescapeStringConn PQescapeStringConn;
static fnPQstatus PQstatus;
static fnPQfinish PQfinish;

static fnPQexec PQexec;
static fnPQresultStatus PQresultStatus;
static fnPQgetvalue PQgetvalue;
static fnPQclear PQclear;
#endif

char* PGPrefix(const char* text, ::testing::internal::GTestColor color) {
    ::testing::internal::ColoredPrintf(color, text);
    return "";
}

#define PG_ERR PGPrefix("[ PG ERROR ] ", ::testing::internal::COLOR_RED)
#define PG_WRN PGPrefix("[ PG WARN  ] ", ::testing::internal::COLOR_YELLOW)
#define PG_INF PGPrefix("[ PG INFO  ] ", ::testing::internal::COLOR_GREEN)

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

    inline void decRefCounter() {
        if (_ptr != nullptr && refCounter != nullptr) {
            if (*refCounter > 0) {
                --*refCounter;
            }
            if (*refCounter == 0) {
                delete refCounter;
                PQclear(_ptr);
                _ptr = nullptr;
            }
        }
    }

public:
    PGresultHolder() : _ptr(nullptr), refCounter(nullptr) {}
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
            if (ptr != nullptr && refCounter == nullptr) {
                refCounter = new uint32_t();
                *refCounter = 1;
            }
            _ptr = ptr;
        }
    }
    PGresult* get() {
        return _ptr;
    }
    ~PGresultHolder() {
        decRefCounter();
        _ptr = nullptr;
        refCounter = nullptr;
    }
};

/// \briaf This class implements singleton which operates with a connection to PostgreSQL server.
class PostgreSQLConnection {
#ifdef PGQL_DYNAMIC_LOAD
    std::shared_ptr<HMODULE> modLibPQ;
#endif
    PGconn* activeConnection;

    PostgreSQLConnection() : activeConnection(nullptr), isConnected(false) {}

    /// \brief Prohobit creation outsize of class, need to make a Singleton
    PostgreSQLConnection(const PostgreSQLConnection&) = delete;
    PostgreSQLConnection& operator=(const PostgreSQLConnection&) = delete;

public:
    bool isConnected;

    static std::shared_ptr<PostgreSQLConnection> GetInstance(void);
    bool Initialize();
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
        if (result.get() != nullptr) {
            ExecStatusType execStatus = PQresultStatus(result.get());
            if (execStatus != expectedStatus) {
                std::cerr << PG_ERR << "Received unexpected result (" << static_cast<unsigned int>(execStatus)
                          << ") from PostgreSQL, expected: " << static_cast<unsigned int>(expectedStatus) << std::endl;
                result.reset(nullptr);
            }
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

    PGconn* GetConnection(void) {
        return this->activeConnection;
    }
    ~PostgreSQLConnection();

    friend std::unique_ptr<PostgreSQLConnection>;
};

static std::shared_ptr<PostgreSQLConnection> connection(nullptr);
std::shared_ptr<PostgreSQLConnection> PostgreSQLConnection::GetInstance(void) {
    if (connection.get() == nullptr) {
        connection = std::shared_ptr<PostgreSQLConnection>(new PostgreSQLConnection());
    }
    return connection;
}

PostgreSQLConnection::~PostgreSQLConnection() {
    if (activeConnection) {
        PQfinish(this->activeConnection);
        this->activeConnection = nullptr;
        this->isConnected = false;
    }
}

/// \brief Initialization of exact object. Uses environment variable PGQL_ENV_CONN_NAME for making a connection.
/// \returns Returns false in case of failure or absence of ENV-variable.
///          Returns true in case of connection has been succesfully established.
bool PostgreSQLConnection::Initialize() {
    if (this->activeConnection != nullptr) {
        std::cerr << PG_WRN << "PostgreSQL connection already established.\n";
        return true;
    }

#ifdef PGQL_DYNAMIC_LOAD

#    ifdef _WIN32
    modLibPQ = std::shared_ptr<HMODULE>(new HMODULE(LoadLibrary("libpq.dll")), [](HMODULE* ptr) {
        if (*ptr != (HMODULE)0) {
            std::cerr << PG_INF << "Freeing libPQ.dll handle\n";
            FreeLibrary(*ptr);
        }
    });
#    else
    modLibPQ = std::shared_ptr<HMODULE>(new HMODULE(dlopen("libpq.so", RTLD_LAZY)), [](HMODULE* ptr) {
        if (*ptr != (HMODULE)0) {
            std::cerr << PG_INF << "Freeing libPQ.so handle\n";
            dlclose(*ptr);
        }
    });
#    endif
    if (*modLibPQ == (HMODULE)0) {
        std::cerr << PG_WRN << "Cannot load PostgreSQL client module libPQ, reporting is unavailable\n";
        return false;
    } else {
        std::cerr << PG_INF << "PostgreSQL cliend module libPQ has been loaded\n";
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
    GETPROC(PQexec);
    GETPROC(PQresultStatus);
    GETPROC(PQgetvalue);
    GETPROC(PQclear);
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

/// \brief Helper for checking PostgreSQL results
#define CHECK_PGRESULT(var_name, error_message, action)    \
    if (var_name.get() == nullptr) {                       \
        std::cerr << PG_ERR << error_message << std::endl; \
        action;                                            \
    }
/// \brief Helper for registering ID-retrieval functions
#define GET_PG_IDENTIFIER(funcDefinition, sqlQuery, varName, fieldName)                  \
    bool funcDefinition {                                                                \
        std::stringstream sstr;                                                          \
                                                                                         \
        sstr << sqlQuery;                                                                \
        auto pgresult = connectionKeeper->Query(sstr.str().c_str());                     \
        CHECK_PGRESULT(pgresult, "Cannot retrieve a correct " #fieldName, return false); \
                                                                                         \
        this->varName = std::atoi(PQgetvalue(pgresult.get(), 0, 0));                     \
        if (this->varName == 0) {                                                        \
            std::cerr << PG_ERR << "Cannot interpret a returned " << #fieldName          \
                      << ", value : " << PQgetvalue(pgresult.get(), 0, 0) << std::endl;  \
            return false;                                                                \
        }                                                                                \
        return true;                                                                     \
    }

/*
    Known issues:
    - String escape isn't applied for all fields (PoC limitation)
*/
/// \brief Class which handles gtest keypoints and send data to PostgreSQL database.
///        May be separated for several source files in case it'll become to huge.
class PostgreSQLEventListener : public ::testing::EmptyTestEventListener {
    std::shared_ptr<PostgreSQLConnection> connectionKeeper;

    const char* session_id = nullptr;
    bool isPostgresEnabled = false;

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
    std::map<std::string, std::string> testCustomFields;

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

    GET_PG_IDENTIFIER(RequestApplicationId(void),
                      "SELECT GET_APPLICATION('" << GetExecutableName() << "')",
                      appId,
                      app_id)
    GET_PG_IDENTIFIER(RequestRunId(void),
                      "SELECT GET_RUN(" << this->testRunId << ", " << this->appId << ", " << this->sessionId << ")",
                      testRunId,
                      run_id)
    GET_PG_IDENTIFIER(RequestHostId(void),
                      "SELECT GET_HOST('" << GetHostname() << "', '" << GetOSVersion() << "')",
                      hostId,
                      host_id)
    GET_PG_IDENTIFIER(RequestSessionId(void), "SELECT GET_SESSION('" << this->session_id << "')", sessionId, session_id)
    GET_PG_IDENTIFIER(RequestSuiteNameId(const char* test_suite_name),
                      "SELECT GET_TEST_SUITE('" << test_suite_name << "', " << this->appId << ")",
                      testSuiteNameId,
                      sn_id)
    GET_PG_IDENTIFIER(RequestSuiteId(std::string query), query, testSuiteId, sr_id)
    GET_PG_IDENTIFIER(RequestTestNameId(std::string query), query, testNameId, tn_id)
    GET_PG_IDENTIFIER(RequestTestId(std::string query), query, testId, tr_id)

    void OnTestSuiteStart(const ::testing::TestSuite& test_suite) override {
        if (!this->isPostgresEnabled || !this->testRunId || !this->sessionId)
            return;

        if (!RequestSuiteNameId(test_suite.name()))
            return;

        std::stringstream sstr;
        sstr << "INSERT INTO suite_results (sr_id, session_id, suite_id) VALUES (DEFAULT, " << this->sessionId << ", "
             << this->testSuiteNameId << ") RETURNING sr_id";
        if (!RequestSuiteId(sstr.str()))
            return;
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

        std::stringstream sstr;
        sstr << "SELECT GET_TEST_NAME(" << this->testSuiteNameId << ", '" << test_info.name() << "'";
        /*
            This part might be specific for different tests. In case amount of cases will be greater than, for example,
           2 the code should be refactored to use a map on test-dependent functions/methods.
        */
        if (test_info.value_param() != NULL &&
            strcmp(::testing::UnitTest::GetInstance()->current_test_suite()->name(), "conformance/ReadIRTest") == 0) {
            /*
            This part of code responsible for cleaning source model XML from
            meaningless information which might be changed run2run by SubgraphDumper
            or similar tool
            */
            std::string testDescription;
            {
                std::ostringstream normalizedXml;
                std::vector<std::string> params =
                    ParseValueParam(test_info.value_param());  // Extracting value_params from serialized string

                if (params.size() > 0) {
                    struct modelNormalizer : pugi::xml_tree_walker {
                        bool for_each(pugi::xml_node& node) override {
                            if (node.type() == pugi::node_element) {
                                node.remove_attribute("name");
                            }
                            return true;
                        }
                    } modelNorm;

                    pugi::xml_document normalizedModel;
                    normalizedModel.load_file(params[0].c_str());
                    normalizedModel.traverse(modelNorm);
                    normalizedModel.save(normalizedXml, "");  // No indent expected, let's reduce size of xml
                    testDescription = normalizedXml.str();
                }
            }
            if (!testDescription.empty()) {
                /*
                    A generated XML may contains characters should be escaped in a query.
                */
                std::vector<char> escapedDescription;
                escapedDescription.resize(testDescription.length() *
                                          2);  // Doc requires to allocate two times more than initial length
                escapedDescription[0] = 0;     //
                int errCode = 0;
                size_t writtenSize = 0;
                writtenSize = PQescapeStringConn(connectionKeeper->GetConnection(),
                                                 escapedDescription.data(),
                                                 testDescription.c_str(),
                                                 testDescription.length(),
                                                 &errCode);
                if (writtenSize >= testDescription.length()) {
                    sstr << ", '" << std::string(escapedDescription.data()) << "'";
                } else {
                    std::cerr << PG_ERR << "Cannot escape string (error code is " << errCode << "):\n"
                              << testDescription;
                }
            }
        }
        sstr << ")";

        if (!RequestTestNameId(sstr.str()))
            return;

        sstr.str("");
        sstr.clear();
        sstr << "INSERT INTO test_results (tr_id, session_id, suite_id, test_id) VALUES (DEFAULT, " << this->sessionId
             << ", " << this->testSuiteId << ", " << this->testNameId << ") RETURNING tr_id";

        if (!RequestTestId(sstr.str()))
            return;
    }

    void OnTestPartResult(const ::testing::TestPartResult& test_part_result) override {
        if (!this->isPostgresEnabled || !this->testRunId || !this->sessionId || !this->testSuiteNameId ||
            !this->testSuiteId || !this->testNameId || !this->testId)
            return;
        //        std::stringstream sstr;
        //        sstr << "INSERT INTO test_starts(part) (name) VALUES (\"partresult\")";
    }

    void OnTestEnd(const ::testing::TestInfo& test_info) override {
        if (!this->isPostgresEnabled || !this->testRunId || !this->sessionId || !this->testSuiteNameId ||
            !this->testSuiteId || !this->testNameId || !this->testId)
            return;

        std::stringstream sstr;
        uint32_t testResult = 0;
        if (test_info.result()->Passed())
            testResult = 1;
        else if (test_info.result()->Skipped())
            testResult = 2;
        sstr << "UPDATE test_results SET finished_at=NOW(), duration=" << test_info.result()->elapsed_time()
             << ", test_result=" << testResult << " WHERE tr_id=" << this->testId;
        auto pgresult = connectionKeeper->Query(sstr.str().c_str(), PGRES_COMMAND_OK);
        CHECK_PGRESULT(pgresult, "Cannot update test results", return );
        this->testId = 0;
    }

    void OnTestSuiteEnd(const ::testing::TestSuite& test_suite) override {
        if (!this->isPostgresEnabled || !this->testRunId || !this->sessionId || !this->testSuiteNameId ||
            !this->testSuiteId)
            return;

        std::stringstream sstr;
        sstr << "UPDATE suite_results SET finished_at=NOW(), duration=" << test_suite.elapsed_time()
             << ", suite_result=" << (test_suite.Passed() ? 1 : 0) << " WHERE sr_id=" << this->testSuiteId;
        auto pgresult = connectionKeeper->Query(sstr.str().c_str(), PGRES_COMMAND_OK);
        CHECK_PGRESULT(pgresult, "Cannot update test suite results", return );
        this->testSuiteId = 0;
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

    ~PostgreSQLEventListener() {
        if (!this->isPostgresEnabled)
            return;

        std::stringstream sstr;
        sstr << "UPDATE runs SET end_time=NOW() WHERE run_id=" << this->testRunId << " AND end_time<NOW()";
        auto pgresult = connectionKeeper->Query(sstr.str().c_str(), PGRES_COMMAND_OK);
        CHECK_PGRESULT(pgresult, "Cannot update run finish info", return );

        sstr.str("");
        sstr.clear();
        sstr << "UPDATE sessions SET end_time=NOW() WHERE session_id=" << this->sessionId << " AND end_time<NOW()";
        pgresult = connectionKeeper->Query(sstr.str().c_str(), PGRES_COMMAND_OK);
        CHECK_PGRESULT(pgresult, "Cannot update session finish info", return );
    }

    /* Prohobit creation outsize of class, need to make a Singleton */
    PostgreSQLEventListener(const PostgreSQLEventListener&) = delete;
    PostgreSQLEventListener& operator=(const PostgreSQLEventListener&) = delete;

    friend class PostgreSQLEnvironment;

public:
    bool SetCustomField(const std::string fieldName, const std::string fieldValue, const bool rewrite) {
        auto field = this->testCustomFields.find(fieldName);
        if (rewrite || field != this->testCustomFields.end()) {
            this->testCustomFields[fieldName] = fieldValue;
            return true;
        }
        return false;
    }

    std::string GetCustomField(const std::string fieldName, const std::string defaultValue) const {
        auto field = this->testCustomFields.find(fieldName);
        if (field != this->testCustomFields.end()) {
            return field->second;
        }
        return defaultValue;
    }

    bool RemoveCustomField(const std::string fieldName) {
        auto field = this->testCustomFields.find(fieldName);
        if (field != this->testCustomFields.end()) {
            this->testCustomFields.erase(field);
            return true;
        }
        return false;
    }

    void ClearCustomFields() {
        this->testCustomFields.clear();
    }
};

/// \brief Global variable (scoped only to this file) which contains pointer to PostgreSQLEventListener
///        Might be replaced by a bool, but left as is for possible future use.
static PostgreSQLEventListener* pgEventListener = nullptr;

/// \brief Class is used for registering environment handler in gtest. It prepares in-time set up
///        for registering PostgreSQLEventListener
class PostgreSQLEnvironment : public ::testing::Environment {
public:
    PostgreSQLEnvironment() {}
    ~PostgreSQLEnvironment() {}
    void SetUp() override {
        if (std::getenv(PGQL_ENV_SESS_NAME) != nullptr && std::getenv(PGQL_ENV_CONN_NAME) != nullptr) {
            if (pgEventListener == nullptr) {
                pgEventListener = new PostgreSQLEventListener();
                ::testing::UnitTest::GetInstance()->listeners().Append(pgEventListener);
            }
        } else {
            std::cerr << PG_INF << "PostgreSQL Reporting is disabled due to missing environment settings\n";
        }
    }
    void TearDown() override {
        // Don't see any reason to do additional tear down
    }
};

/// \brief Global variable which stores a pointer to active instance (only one is expected) of
/// PostgreSQLEnvironment.
::testing::Environment* PostgreSQLEnvironment_Reg = ::testing::AddGlobalTestEnvironment(new PostgreSQLEnvironment());

/// \brief This class is for internal usage, don't need to move it to the header. It holds an internal state of
///        PostgreSQLLink instance. Introduced to simplify header.
class PostgreSQLCustomData {
public:
    std::map<std::string, std::string> customFields;
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

PostgreSQLLink::~PostgreSQLLink() {
    if (this->customData) {
        delete this->customData;
        this->customData = nullptr;
    }

    this->parentObject = nullptr;
#ifdef PGQL_DEBUG
    std::cout << PG_INF << "PostgreSQLLink Finished\n";
#endif
}

bool PostgreSQLLink::SetCustomField(const std::string fieldName, const std::string fieldValue, const bool rewrite) {
    if (pgEventListener) {
        if (!pgEventListener->SetCustomField(fieldName, fieldValue, rewrite))
            return false;
    }
    auto field = this->customData->customFields.find(fieldName);
    if (rewrite || field != this->customData->customFields.end()) {
        this->customData->customFields[fieldName] = fieldValue;
        return true;
    }
    return false;
}

std::string PostgreSQLLink::GetCustomField(const std::string fieldName, const std::string defaultValue) const {
    if (pgEventListener) {
        return pgEventListener->GetCustomField(fieldName, defaultValue);
    }
    auto field = this->customData->customFields.find(fieldName);
    if (field != this->customData->customFields.end()) {
        return field->second;
    }
    return defaultValue;
}

bool PostgreSQLLink::RemoveCustomField(const std::string fieldName) {
    if (pgEventListener) {
        pgEventListener->RemoveCustomField(fieldName);
    }
    auto field = this->customData->customFields.find(fieldName);
    if (field != this->customData->customFields.end()) {
        this->customData->customFields.erase(field);
        return true;
    }
    return false;
}

}  // namespace CommonTestUtils
