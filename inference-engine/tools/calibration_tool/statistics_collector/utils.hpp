// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <string>

#include <ie_blob.h>

std::string getFullName(const std::string& name, const std::string& dir);

/**
 * @brief Looking for regular file in directory
 * @param dir - path to the target directory
 * @param obj_number - desired number of regular files
 * @param includePath - image file name
 * @return detected files paths as deque
 */
std::deque<std::string> getDirRegContents(const std::string& dir, size_t obj_number, bool includePath = true);

std::deque<std::string> getDatasetEntries(const std::string& path, size_t obj_number = 0lu);

InferenceEngine::Blob::Ptr convertBlobFP32toFP16(InferenceEngine::Blob::Ptr blob);

InferenceEngine::Blob::Ptr convertBlobFP16toFP32(InferenceEngine::Blob::Ptr blob);

bool isFile(const std::string& path);

bool isDirectory(const std::string& path);
