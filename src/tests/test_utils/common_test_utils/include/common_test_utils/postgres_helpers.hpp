// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gtest/gtest.h>
#include <stdlib.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

namespace ov {
namespace test {
namespace utils {

/// \brief Enables dynamic load of libpq module
#define PGQL_DYNAMIC_LOAD
/// \brief Enables extended debug messages to the stderr
#define PGQL_DEBUG
#undef PGQL_DEBUG

extern const char* PGQL_ENV_CONN_NAME;  // Environment variable with connection settings
extern const char* PGQL_ENV_SESS_NAME;  // Environment variable identifies current session
extern const char* PGQL_ENV_RUN_NAME;   // Environment variable with external run id
extern const char* PGQL_ENV_RLVL_NAME;  // Environment variable identifies reporting

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
#    include <time.h>
#    include <unistd.h>
#elif defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <sys/param.h>
#    include <sys/utsname.h>
#    include <time.h>
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

extern fnPQconnectdb PQconnectdb;
extern fnPQescapeStringConn PQescapeStringConn;
extern fnPQstatus PQstatus;
extern fnPQfinish PQfinish;
extern fnPQerrorMessage PQerrorMessage;

extern fnPQexec PQexec;
extern fnPQresultStatus PQresultStatus;
extern fnPQgetvalue PQgetvalue;
extern fnPQgetisnull PQgetisnull;
extern fnPQclear PQclear;
extern fnPQresultErrorMessage PQresultErrorMessage;
#endif

extern const char* PGPrefix(const char* text, ::testing::internal::GTestColor color);

#define PG_ERR PGPrefix("[ PG ERROR ] ", ::testing::internal::COLOR_RED)
#define PG_WRN PGPrefix("[ PG WARN  ] ", ::testing::internal::COLOR_YELLOW)
#define PG_INF PGPrefix("[ PG INFO  ] ", ::testing::internal::COLOR_GREEN)

/// \brief Count of tries when serialization error is detected after query
const uint8_t serializationTriesCount = 30;  // Pause between each attempt is not less than 50ms

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
    bool load_libpq(void);
#endif
    PGconn* m_active_connection;

    PostgreSQLConnection(void) : m_active_connection(nullptr), m_is_connected(false) {}

    /// \brief Prohobit creation outsize of class, need to make a Singleton
    PostgreSQLConnection(const PostgreSQLConnection&) = delete;
    PostgreSQLConnection& operator=(const PostgreSQLConnection&) = delete;

public:
    bool m_is_connected;

    static std::shared_ptr<PostgreSQLConnection> get_instance(void);
    bool initialize(void);
    /// \brief Make a common query to a server. Result will be returned as self-desctructable pointer. But application
    /// should check result pointer isn't a nullptr. And result status by itself. \param[in] query SQL query to a server
    /// \returns Object which keep pointer on received PGresult. It contains nullptr in case of any error.
    PGresultHolder common_query(const char* query);

    /// \brief Queries a server. Result will be returned as self-desctructable pointer. But application should check
    /// result pointer isn't a nullptr.
    /// \param[in] query SQL query to a server
    /// \param[in] expectedStatus Query result will be checked for passed status, if it isn't equal - result pointer
    /// \param[in] smartRetry Useful for transactional queries, allows to call non-transactional part in case
    /// of transactional errors
    /// will be nullptr. \returns Object which keep pointer on received PGresult. It contains nullptr in case of any
    /// error.
    PGresultHolder query(const char* query,
                         const ExecStatusType expectedStatus = PGRES_TUPLES_OK,
                         const bool smartRetry = false);

    /// \brief Tries to reconnect in case of connection issues (usual usage - connection timeout).
    void try_reconnect(void);

    PGconn* get_connection(void) const {
        return this->m_active_connection;
    }
    ~PostgreSQLConnection(void);
};
extern std::shared_ptr<PostgreSQLConnection> connection;

namespace PostgreSQLHelpers {
/// \brief This method is used for parsing serialized value_param string.
///        Known limitations:
///        It doesn't read values in inner tuples/arrays/etc.
std::vector<std::string> parse_value_param(std::string text);

/// \brief Function returns OS version in runtime.
/// \returns String which contains OS version
std::string get_os_version(void);

/// \brief Function returns executable name of current application.
/// \returs File name as a std::string
std::string get_executable_name(void);

/// \brief Cross-platform implementation of getting host name
/// \returns String with host name or "NOT_FOUND" in case of error
std::string get_hostname(void);

// Procedure uses for possible customization of addint key=value pairs
void add_pair(std::map<std::string, std::string>& keyValues, const std::string& key, const std::string& value);

// Function parses test name for key=value pairs
bool parse_test_name(const char* line, std::map<std::string, std::string>& keyValues);

/// \brief Compiles string and replaces variables defined as $[0-9A-Za-z-_] by values from
/// provided key=value map. Replaces by blank in case of key isn't found in the map.
/// \param[in] srcStr String contains variables
/// \param[in] keyValue Key=value map with variable values
/// \param[out] result String for result
/// \returns Returns true if all input string was compiled, false in case of any compilation error
bool compile_string(const std::string& srcStr, const std::map<std::string, std::string>& keyValue, std::string& result);
}  // namespace PostgreSQLHelpers
}  // namespace utils
}  // namespace test
}  // namespace ov
