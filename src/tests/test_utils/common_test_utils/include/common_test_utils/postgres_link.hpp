// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "test_common.hpp"

namespace CommonTestUtils {

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
    PostgreSQLLink();
    /// \brief Constructor allows to store unsafe pointer to parent object. Might be user
    ///        as external identifier.
    /// \param[in] ptrParentObject Unsafe pointer to a parent object
    PostgreSQLLink(void* ptrParentObject) : PostgreSQLLink() {
        SetParentObject(ptrParentObject);
    }
    /// \brief Simple destructor
    ~PostgreSQLLink();

    /// \brief Returns pointer to stored parentObject.
    /// \returns Unsafe pointer to parent object.
    void* GetParentObject() {
        return this->parentObject;
    }
    /// \brief Replaces stored pointer on parent object. Might be nullptr to reset stored pointer.
    /// \param[in] ptrParentObject Unsafe pointer to a parent object
    void SetParentObject(void* ptrParentObject) {
        this->parentObject = ptrParentObject;
    }
    /// \brief Sets custom field for current test instance
    /// \param[in] fieldName Field name, any applicable string
    /// \param[in] fieldValue Value to store as field value, any applicable string
    /// \param[in] rewrite Flag defines behaviour in case field already exists. Rewrites if true.
    /// \returns True if value has been stored, false otherwise.
    bool SetCustomField(const std::string fieldName, const std::string fieldValue, const bool rewrite = false);
    /// \brief Gets custom field value for current test instance
    /// \param[in] fieldName Field name, any applicable string
    /// \param[in] defaultValue Value should be returned in case of value wasn't stored, any applicable string
    /// \returns Stored value or defaultValue otherwise.
    std::string GetCustomField(const std::string fieldName, const std::string defaultValue) const;
    /// \brief Removes custom field for current test instance
    /// \param[in] fieldName Field name, any applicable string
    /// \returns True if value has been removed, false otherwise.
    bool RemoveCustomField(const std::string fieldName);
};

}  // namespace CommonTestUtils
