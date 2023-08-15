// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/postgres_link.hpp"

#include "common_test_utils/postgres_helpers.hpp"
#include "openvino/core/version.hpp"

static std::map<std::string, std::string>
    ExtTestQueries;  // Map of extended test queries. It is used for do a custom query after inserting mandatory row
static std::map<std::string, std::string>
    ExtTestNames;  // Map of extended test name convertors. It is used to change a test name automatically.

namespace ov {
namespace test {
namespace utils {

using namespace PostgreSQLHelpers;

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
        return _internal_request_id(sstr.str(), #fieldName, varName);   \
    }

/// \brief Class which handles gtest keypoints and send data to PostgreSQL database.
///        May be separated for several source files in case it'll become to huge.
class PostgreSQLEventListener : public ::testing::EmptyTestEventListener {
    std::shared_ptr<PostgreSQLConnection> connectionKeeper;

    std::string bin_version;
    std::string lib_version;

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
    bool isManualStart = false;    // Signals test case start will be called manually

    /* Test name parsing */
    std::map<std::string, std::string>
        testDictionary;  // Contains key=value pairs which should be used while constructing queries and names
    std::map<std::string, std::string>::iterator grpName;
    std::string testName, fullTestName;
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
    bool _internal_request_id(const std::string& sqlQuery, const char* fieldName, uint64_t& result) {
        auto selectPos = sqlQuery.find("SELECT");
        const char *query = sqlQuery.c_str(), *query_start = query;
        bool isTransactionalQuery = (selectPos != std::string::npos) && (sqlQuery.find("BEGIN") == 0);
        if (isTransactionalQuery) {
            query += selectPos;
        }
#ifdef PGQL_DEBUG
        std::cout << "Query: " << query << std::endl;
#endif
        auto pgresult = connectionKeeper->query(query);
        CHECK_PGRESULT(pgresult, "Cannot retrieve a correct " << fieldName, return false);

        bool isNull = PQgetisnull(pgresult.get(), 0, 0) != 0;
        if (isNull && isTransactionalQuery) {
#ifdef PGQL_DEBUG
            std::cout << "Transactional query: " << query << std::endl;
#endif
            pgresult = connectionKeeper->query(query_start, PGRES_TUPLES_OK, true);
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
    std::string escape_string(const std::string& sourceString) const {
        if (sourceString.length() == 0) {
            return std::string("");
        }

        std::vector<char> escapedString;
        escapedString.resize(sourceString.length() * 2);  // Doc requires to allocate two times more than initial length
        escapedString[0] = 0;
        int errCode = 0;
        size_t writtenSize = 0;
        writtenSize = PQescapeStringConn(connectionKeeper->get_connection(),
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

    GET_PG_IDENTIFIER(request_application_id(void),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_APPLICATION('" << get_executable_name() << "');"
                          << "COMMIT; "
                          << "SELECT GET_APPLICATION('" << get_executable_name() << "');",
                      appId,
                      app_id)
    GET_PG_IDENTIFIER(request_run_id(void),
                      "SELECT GET_RUN('" << this->run_id << "', " << this->appId << ", " << this->sessionId << ", "
                                         << this->hostId << ", " << static_cast<uint16_t>(this->reportingLevel)
                                         << "::smallint);",
                      testRunId,
                      run_id)
    GET_PG_IDENTIFIER(request_host_id(void),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_HOST('" << get_hostname() << "', '" << get_os_version() << "');"
                          << "COMMIT; "
                          << "SELECT GET_HOST('" << get_hostname() << "', '" << get_os_version() << "');",
                      hostId,
                      host_id)
    GET_PG_IDENTIFIER(request_session_id(void),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_SESSION('" << this->session_id << "', '" << this->bin_version << "', '"
                          << this->lib_version << "');"
                          << "COMMIT;"
                          << "SELECT GET_SESSION('" << this->session_id << "', '" << this->bin_version << "', '"
                          << this->lib_version << "');",
                      sessionId,
                      session_id)
    GET_PG_IDENTIFIER(request_suite_name_id(const char* test_suite_name),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_TEST_SUITE('" << test_suite_name << "', " << this->appId << ");"
                          << "COMMIT; "
                          << "SELECT GET_TEST_SUITE('" << test_suite_name << "', " << this->appId << ");",
                      testSuiteNameId,
                      sn_id)
    GET_PG_IDENTIFIER(request_test_name_id(std::string query),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_" << query << "; COMMIT;"
                          << "SELECT GET_" << query,
                      testNameId,
                      tn_id)
    GET_PG_IDENTIFIER(request_suite_id(void),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_SUITE_ID(" << this->testSuiteNameId << ", " << this->sessionId << ", "
                          << this->testRunId << ");"
                          << "COMMIT; "
                          << "SELECT GET_SUITE_ID(" << this->testSuiteNameId << ", " << this->sessionId << ", "
                          << this->testRunId << ");",
                      testSuiteId,
                      sr_id)
    GET_PG_IDENTIFIER(request_suite_id(std::string query), query, testSuiteId, sr_id)
    GET_PG_IDENTIFIER(request_test_id(std::string query), query, testId, tr_id)
    GET_PG_IDENTIFIER(request_test_ext_id(std::string query),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_MISS_" << query << "; COMMIT;"
                          << "SELECT ON_START_" << query,
                      testExtId,
                      t_id)
    GET_PG_IDENTIFIER(update_test_ext_id(std::string query),
                      "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE; "
                          << "CALL ON_END_MISS_" << query << "; COMMIT;"
                          << "SELECT ON_END_" << query,
                      testExtId,
                      t_id)

    /// \brief Send an update query. Depends on mode in calls specific UPDATE query in case of default reporting
    /// and regular ON_START query in case of fast reporting.
    /// In such case UPDATE query is used only for updating changed values in runtime, when
    /// ON_START query in this place already has all updated data, that's why UPDATE query is skipping
    void update_test_results(const ::testing::TestInfo* test_info = nullptr) {
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
                if (compile_string(extQuery->second, testDictionary, query)) {
                    if (reportingLevel == REPORT_LVL_DEFAULT) {
                        if (!update_test_ext_id(query)) {
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

    /// \brief Internal call of delayed start process which puts information about
    /// TestCase start into the table
    void _internal_start(void) {
        // In case of results will be refused - do not do anything
        if (!this->isPostgresEnabled || !this->testRunId || !this->sessionId || !this->testSuiteNameId ||
            !this->testSuiteId || isRefusedResult) {
            return;
        }

        std::stringstream sstr;

        if (!isManualStart) {
            // Creates temporary record
            sstr << "INSERT INTO test_results_temp (tr_id, session_id, suite_id, run_id, app_id, test_id, test_result) "
                 << "VALUES (DEFAULT, " << this->sessionId << ", " << this->testSuiteId << ", " << this->testRunId
                 << ", " << this->appId << ", " << this->testNameId << ", 0::smallint) RETURNING tr_id";
        } else {
            // Creates record
            sstr << "INSERT INTO test_results (tr_id, session_id, suite_id, run_id, app_id, test_id, test_result) "
                 << "VALUES (DEFAULT, " << this->sessionId << ", " << this->testSuiteId << ", " << this->testRunId
                 << ", " << this->appId << ", " << this->testNameId << ", 0::smallint) RETURNING tr_id";
        }

        if (!request_test_id(sstr.str()))
            return;

        if (grpName != testDictionary.end()) {
            // Looks query with GroupName + "_ON_START", "ReadIR_ON_START" as example
            auto extQuery = ExtTestQueries.find(grpName->second + "_ON_START");
            if (extQuery != ExtTestQueries.end()) {
                std::string query;
                testDictionary["__test_id"] = std::to_string(this->testId);
                if (compile_string(extQuery->second, testDictionary, query)) {
                    if (!request_test_ext_id(query)) {
                        std::cerr << PG_WRN << "Failed extended query: " << query << std::endl;
                    } else {
                        testDictionary["__test_ext_id"] = std::to_string(this->testExtId);
                    }
                } else {
                    std::cerr << PG_WRN << "Preparing extended query is failed: " << fullTestName << std::endl;
                }
            }
        }
    }

    void OnTestSuiteStart(const ::testing::TestSuite& test_suite) override {
        if (!this->isPostgresEnabled || !this->testRunId || !this->sessionId)
            return;
        try {
            if (!request_suite_name_id(escape_string(test_suite.name()).c_str()))
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
        if (!request_suite_id())
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

        this->testId = 0;
        this->testExtId = 0;
        isRefusedResult = false;
        testDictionary.clear();

        grpName = testDictionary.end();
        testName = test_info.name();  // Could be changed later
        fullTestName = testName;      // Shouldn't be changed, used as global identifier

        if ((isTestNameParsed = parse_test_name(fullTestName.c_str(), testDictionary)) == true &&
            (grpName = testDictionary.find("__groupName__")) != testDictionary.end()) {
            auto nameConvertStr = ExtTestNames.find(grpName->second);
            if (nameConvertStr != ExtTestNames.end()) {
                if (!compile_string(nameConvertStr->second, testDictionary, testName)) {
                    std::cerr << PG_WRN << "Error compiling test name: " << fullTestName << std::endl;
                    testName = grpName->second;
                }
            } else {
                testName = grpName->second;
            }
        } else {
            std::cerr << PG_WRN << "Error parsing test name: " << fullTestName << std::endl;
        }

        std::stringstream sstr;
        try {
            if (reportingLevel == REPORT_LVL_DEFAULT) {
                sstr << "TEST_NAME(" << this->testSuiteNameId << ", '" << escape_string(testName) << "'";
            } else if (reportingLevel == REPORT_LVL_FAST) {
                sstr << "SELECT ADD_TEST_RESULT(" << this->appId << ", " << this->sessionId << ", " << this->testRunId
                     << ", " << this->testSuiteId << ", " << this->testSuiteNameId << ", '" << escape_string(testName)
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
        testDictionary["__session_id"] = std::to_string(this->sessionId);
        testDictionary["__run_id"] = std::to_string(this->testRunId);
        testDictionary["__is_temp"] =
            (reportingLevel == REPORT_LVL_DEFAULT && !isManualStart ? "1::boolean" : "0::boolean");
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

        if (!request_test_name_id(sstr.str()))
            return;

        // If manual start isn't requested - push information immediately to the table
        if (!isManualStart)
            _internal_start();
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
                        if (compile_string(extQuery->second, testDictionary, query)) {
                            auto pgresult = connectionKeeper->query((std::string("CALL ON_REFUSE_") + query).c_str(),
                                                                    PGRES_COMMAND_OK);
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
                auto pgresult = connectionKeeper->query(sstr.str().c_str(), PGRES_COMMAND_OK);
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

        // Need to use such order to be able simplify queries to database
        // State 0 - Incomplete state, worst possible
        uint32_t testResult = 32;  // Failed state
        if (test_info.result()->Passed())
            testResult = 128;
        else if (test_info.result()->Skipped())
            testResult = 64;

        if (reportingLevel == REPORT_LVL_DEFAULT) {
            std::stringstream sstr;
            uint64_t testTempId = this->testId;
            if (!isManualStart) {
                sstr << "INSERT INTO test_results (session_id, suite_id, run_id, app_id, test_id, started_at, "
                        "finished_at, duration, test_result) "
                     << "(SELECT session_id, suite_id, run_id, app_id, test_id, started_at, NOW(), "
                     << test_info.result()->elapsed_time() << ", " << testResult << "::smallint FROM test_results_temp "
                     << "WHERE tr_id=" << this->testId << ") RETURNING tr_id";
                if (!request_test_id(sstr.str())) {
                    return;
                } else {
                    sstr.str("");
                    sstr.clear();
                    sstr << "DELETE FROM test_results_temp WHERE tr_id=" << testTempId;
                    auto pgresult = connectionKeeper->query(sstr.str().c_str(), PGRES_COMMAND_OK);
                    CHECK_PGRESULT(pgresult, "Cannot remove temporary test results", /* no return */);

                    // Set correct information about
                    set_custom_field("__test_ext_id", std::to_string(this->testId), true);
                    set_custom_field("__test_id", std::to_string(testTempId), true);
                    // Force updating fields because in some conditions IDs might be equal
                    isFieldsUpdated = true;
                }
            } else {
                sstr << "UPDATE test_results SET duration = " << test_info.result()->elapsed_time()
                     << ", test_result = " << testResult << "::smallint, finished_at = NOW() "
                     << "WHERE tr_id=" << this->testId;
                auto pgresult = connectionKeeper->query(sstr.str().c_str(), PGRES_COMMAND_OK);
                CHECK_PGRESULT(pgresult, "Cannot update test results", /* no return */);

                // Set correct information about
                set_custom_field("__test_ext_id", std::to_string(this->testId), true);
                set_custom_field("__test_id", std::to_string(this->testId), true);
                // Force updating fields because in some conditions IDs might be equal
                isFieldsUpdated = true;
            }
        } else if (reportingLevel == REPORT_LVL_FAST) {
            joinedQuery << ", " << testResult << "::smallint, " << test_info.result()->elapsed_time() << ");\n";
        }

        update_test_results(&test_info);

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
        auto pgresult = connectionKeeper->query(sstr.str().c_str(), PGRES_COMMAND_OK);
        CHECK_PGRESULT(pgresult, "Cannot update test suite results", return);
        this->testSuiteId = 0;

        if (reportingLevel == REPORT_LVL_FAST) {
            pgresult = connectionKeeper->query(joinedQuery.str().c_str());
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
            connectionKeeper = PostgreSQLConnection::get_instance();
            bool connInitResult = connectionKeeper->initialize();

            if (!connInitResult)
                return;

            bin_version = std::to_string(OPENVINO_VERSION_MAJOR) + "." + std::to_string(OPENVINO_VERSION_MINOR) + "." +
                          std::to_string(OPENVINO_VERSION_PATCH);
            {
                ov::Version version = ov::get_openvino_version();
                lib_version = version.buildNumber;
            }

            isPostgresEnabled = connInitResult;

            if (isPostgresEnabled)
                isPostgresEnabled &= request_application_id();
            if (isPostgresEnabled)
                isPostgresEnabled &= request_host_id();
            if (isPostgresEnabled)
                isPostgresEnabled &= request_session_id();
            if (isPostgresEnabled)
                isPostgresEnabled &= request_run_id();

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
        auto pgresult = connectionKeeper->query(sstr.str().c_str(), PGRES_COMMAND_OK);
        CHECK_PGRESULT(pgresult, "Cannot update run finish info", return);

        sstr.str("");
        sstr.clear();
        sstr << "UPDATE sessions SET end_time=NOW() WHERE session_id=" << this->sessionId << " AND end_time<NOW()";
        pgresult = connectionKeeper->query(sstr.str().c_str(), PGRES_COMMAND_OK);
        CHECK_PGRESULT(pgresult, "Cannot update session finish info", return);
    }

    /* Prohobit creation outsize of class, need to make a Singleton */
    PostgreSQLEventListener(const PostgreSQLEventListener&) = delete;
    PostgreSQLEventListener& operator=(const PostgreSQLEventListener&) = delete;

    friend class PostgreSQLEnvironment;

public:
    void manual_start(void) {
        if (isManualStart && this->testId == 0 && reportingLevel == REPORT_LVL_DEFAULT) {
            _internal_start();
        }
    }

    void set_refuse_result(bool value = true) {
        isRefusedResult = value;
    }

    void set_manual_start(bool value = true) {
        isManualStart = value;
    }

    bool set_custom_field(const std::string fieldName, const std::string fieldValue, const bool rewrite) {
        auto field = this->testDictionary.find(fieldName);
        if (rewrite || field == this->testDictionary.end()) {
            isFieldsUpdated |= (field == this->testDictionary.end()) ||
                               (field->second != fieldValue);  // Signals only in case value not equal with existing
            // Some plugins may return corrupted strings with zero-character at the end, it may
            // corrupt queries
            size_t zero_pos = fieldValue.find_first_of('\0');
            this->testDictionary[fieldName] =
                zero_pos == std::string::npos ? fieldValue : fieldValue.substr(0, zero_pos);
            return true;
        }
        return false;
    }

    std::string get_custom_field(const std::string fieldName, const std::string defaultValue) const {
        auto field = this->testDictionary.find(fieldName);
        if (field != this->testDictionary.end()) {
            return field->second;
        }
        return defaultValue;
    }

    bool remove_custom_field(const std::string fieldName) {
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
/// \brief Used for lazy set_manual_start call
static bool pgEventListenerInitialManualStart = false;

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
                pgEventListener->set_manual_start(pgEventListenerInitialManualStart);
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

void PostgreSQLLink::set_refuse_result(bool value) const {
    if (pgEventListener) {
        pgEventListener->set_refuse_result(value);
    }
}

void PostgreSQLLink::set_manual_start(bool value) const {
    if (pgEventListener) {
        pgEventListener->set_manual_start(value);
    }
}

void PostgreSQLLink::manual_start(void) const {
    if (pgEventListener) {
        pgEventListener->manual_start();
    }
}

bool PostgreSQLLink::set_custom_field(const std::string fieldName,
                                      const std::string fieldValue,
                                      const bool rewrite) const {
    if (pgEventListener) {
        return pgEventListener->set_custom_field(fieldName, fieldValue, rewrite);
    }
    return false;
}

std::string PostgreSQLLink::get_custom_field(const std::string fieldName, const std::string defaultValue) const {
    if (pgEventListener) {
        return pgEventListener->get_custom_field(fieldName, defaultValue);
    }
    return defaultValue;
}

bool PostgreSQLLink::remove_custom_field(const std::string fieldName) const {
    if (pgEventListener) {
        pgEventListener->remove_custom_field(fieldName);
    }
    return false;
}

}  // namespace utils
}  // namespace test
}  // namespace ov

namespace PostgreSQLLink {
std::map<std::string, std::string>* get_ext_test_queries(void) {
    return &ExtTestQueries;
}

std::map<std::string, std::string>* get_ext_test_names(void) {
    return &ExtTestNames;
}

void set_manual_start(bool value) {
    if (::ov::test::utils::pgEventListener) {
        ::ov::test::utils::pgEventListener->set_manual_start(value);
    } else {
        ::ov::test::utils::pgEventListenerInitialManualStart = value;
    }
}

void manual_start(void) {
    if (::ov::test::utils::pgEventListener) {
        ::ov::test::utils::pgEventListener->manual_start();
    }
}
}  // namespace PostgreSQLLink
