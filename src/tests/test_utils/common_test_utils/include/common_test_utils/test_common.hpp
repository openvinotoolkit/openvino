// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <string>
#include "common_test_utils/test_assertions.hpp"

namespace CommonTestUtils {
class PostgreSQLLink;

class TestsCommon : virtual public ::testing::Test {
    /// \brief Holds a pointer on PostgreSQL interface implementation (see postgres_link.hpp).
    ///
    ///        Better to keep it without #ifdef condition to prevent complex issues with
    ///        unexpected stack corruptions/segmentation faults in case this header
    ///        uses in a project, which doesn't define expected definition.
    ///        But if no handler of the variable is linked to a final runtime, then it
    ///        will show an assert if some code tries to use it by a corresponding getter.
    PostgreSQLLink* PGLink;

protected:
    TestsCommon();
    ~TestsCommon() override;

    static std::string GetTimestamp();
    std::string GetTestName() const;
    std::string GetFullTestName() const;

    /// \brief Returns a pointer on a PostgreSQL interface implementation (use postgres_link.hpp to
    ///        get an access to a interface properties).
    ///
    ///        If project supported it - it should be mandatory available. Otherwise assert will
    ///        show an error. If project doesn't support it - it just will return a nullptr.
    ///        Behaviour might be carefully changed by removing #ifdef condition.
    ///        A removing the condition might be useful to find where GetPGLink is used
    ///        by a wrong behaviour.
    /// \returns If object supports PostgreSQL reporting, then the method returns a pointer on
    ///          PostgreSQL interface implementation, otherwise - shows an assert or return a nullptr.
    PostgreSQLLink* GetPGLink() {
#ifdef ENABLE_CONFORMANCE_PGQL
        assert(this->PGLink != nullptr);
#endif
        return this->PGLink;
    }
};

}  // namespace CommonTestUtils
