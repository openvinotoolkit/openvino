// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdlib.h>

#include <map>
#include <string>

namespace ov {
namespace test {
namespace utils {

/// \brief Class-container for PostgreSQLLink-specific data, declared and implemented
///        out of header. Definition mustn't be a part of this header.
class PostgreSQLCustomData;

/// \brief Class incapsulates logic for communication with PostgreSQL from test-side
///        Most logic should be implemented in corresponding postgres_link.cpp
class PostgreSQLLink {
    /// \brief parentObject should store pointer to a parent object to be able
    ///        access to some specific parent's fields, if needed. Isn't mandatory.
    void* parentObject;
    /// \brief customData contains internal object state. It mustn't be accessible
    ///        from outside.
    PostgreSQLCustomData* customData;

public:
    /// \brief Simple constructor
    PostgreSQLLink(void);
    /// \brief Constructor allows to store unsafe pointer to parent object. Might be user
    ///        as external identifier.
    /// \param[in] ptrParentObject Unsafe pointer to a parent object
    PostgreSQLLink(void* ptrParentObject);
    /// \brief Simple destructor
    ~PostgreSQLLink(void);

    /// \brief Returns pointer to stored parentObject.
    /// \returns Unsafe pointer to parent object.
    void* get_parent_object(void) {
        return this->parentObject;
    }
    /// \brief Replaces stored pointer on parent object. Might be nullptr to reset stored pointer.
    /// \param[in] ptrParentObject Unsafe pointer to a parent object
    void set_parent_object(void* ptrParentObject) {
        this->parentObject = ptrParentObject;
    }
    /// \brief Sets custom field for current test instance
    /// \param[in] fieldName Field name, any applicable string
    /// \param[in] fieldValue Value to store as field value, any applicable string
    /// \param[in] rewrite Flag defines behaviour in case field already exists. Rewrites if true.
    /// \returns True if value has been stored, false otherwise.
    bool set_custom_field(const std::string fieldName, const std::string fieldValue, const bool rewrite = false) const;
    /// \brief Gets custom field value for current test instance
    /// \param[in] fieldName Field name, any applicable string
    /// \param[in] defaultValue Value should be returned in case of value wasn't stored, any applicable string
    /// \returns Stored value or defaultValue otherwise.
    std::string get_custom_field(const std::string fieldName, const std::string defaultValue) const;
    /// \brief Removes custom field for current test instance
    /// \param[in] fieldName Field name, any applicable string
    /// \returns True if value has been removed, false otherwise.
    bool remove_custom_field(const std::string fieldName) const;
    /// \brief Sets waste result flag which means do not store results
    /// \param[in] value Value should be set, true is default
    void set_refuse_result(bool value = true) const;
    /// \brief Sets manual start flag which allows initiate storing results manually
    /// IMPORTANT: It cannot change workflow of current execution, only for
    /// further calls.
    /// \param[in] value Value should be set, true is default
    void set_manual_start(bool value = true) const;
    /// \brief Initiate a TestCase start procedure which puts information into the table
    /// Works only if previously called set_manual_start()
    void manual_start(void) const;
};
}  // namespace utils
}  // namespace test
}  // namespace ov

#ifdef ENABLE_CONFORMANCE_PGQL
namespace PostgreSQLLink {
/// \brief Returns pointer on a global map which contains pairs of Extended Test Queries
/// Each pair has test name as a key and SQL-query as a value.
/// Query can contain a variables started with $ and be replaced by an actual values
/// Variables are parsed from test name.
extern std::map<std::string, std::string>* get_ext_test_queries(void);
/// \brief Returns pointer on a global map which contains pairs of Extended Test Names
/// Each pair has test name as a key and string as a value.
/// Query can contain a variables started with $ and be replaced by an actual values
/// Variables are parsed from test name.
extern std::map<std::string, std::string>* get_ext_test_names(void);
/// \brief Sets manual start flag which allows initiate storing results manually.
/// It may affect workflow of further execution if called before OnStartTestCase event
/// will be called.
/// \param[in] value Value should be set, true is default
extern void set_manual_start(bool value = true);
/// \brief Initiate a TestCase start procedure which puts information into the table
/// Works only if previously called set_manual_start()
void manual_start(void);
};  // namespace PostgreSQLLink
#endif
