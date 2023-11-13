// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief TODO: short file description
 * \file file_utils.h
 */
#pragma once

#include <string>

namespace testing { namespace FileUtils {
    /**
     * @brief TODO: description
     * @param file_name - TODO: param
     * @param buffer - TODO: param
     * @param maxSize - TODO: param
     */
    void readAllFile(const std::string& file_name, void* buffer, size_t maxSize);

    /**
     * @brief TODO: description
     * @param filepath - TODO: param
     * @return TODO: ret obj
     */
    std::string folderOf(const std::string &filepath);

    /**
     * @brief TODO: description
     * @param folder - TODO: param
     * @param file - TODO: param
     * @return TODO: ret obj
     */
    std::string makePath(const std::string& folder, const std::string& file);

    /**
     * @brief TODO: description
     * @param filepath - TODO: param
     * @return TODO: ret obj
     */
    std::string fileNameNoExt(const std::string &filepath);

    /**
     * @brief TODO: description
     * @param filename - TODO: param
     * @return TODO: ret obj
     */
    std::string fileExt(const char* filename);

    /**
     * @brief TODO: description
     * @param filename - TODO: param
     * @return TODO: ret obj
     */
    std::string fileExt(const std::string &filename);
}}

