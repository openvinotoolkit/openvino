// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <deque>
#include <stdio.h>
#include <string>

#include "inference_engine/precision_utils.h"
#include "utils.hpp"
#include "user_exception.hpp"

#ifdef _WIN32
# include <os/windows/w_dirent.h>
#else
# include <sys/stat.h>
# include <dirent.h>
#endif


using namespace InferenceEngine;

std::string getFullName(const std::string& name, const std::string& dir) {
    return dir + "/" + name;
}

std::deque<std::string> getDirRegContents(const std::string& dir, size_t obj_number, bool includePath) {
    struct stat sb;
    if (stat(dir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
        THROW_USER_EXCEPTION(1)
                << "Cannot read contents of the directory '"
                << dir
                << "'. It is not a directory or not accessible";
    }

    std::deque<std::string> entries;
    DIR *dp;
    dp = opendir(dir.c_str());
    if (dp == nullptr) {
        THROW_USER_EXCEPTION(1) << "Cannot open the directory '" << dir << "'";
    }
    size_t rest = obj_number > 0lu ? obj_number : 1lu;

    struct dirent *ep;
    while (nullptr != (ep = readdir(dp)) && rest > 0lu) {
#ifdef _WIN32
        if (ep->d_name == nullptr) continue;
#endif
        struct stat sb_n;
        if (strncmp(ep->d_name, ".", 1lu) == 0
                || strncmp(ep->d_name, "..", 2lu) == 0) continue;
        std::string f_name = getFullName(ep->d_name, dir);
        if (stat(f_name.c_str(), &sb_n) != 0
                || S_ISDIR(sb_n.st_mode)) continue;
        entries.push_back(includePath ? f_name : ep->d_name);
        if (obj_number > 0lu) rest--;
    }
    closedir(dp);
    return entries;
}

std::deque<std::string> getDatasetEntries(const std::string& path, size_t obj_number) {
    struct stat sb;
    if (stat(path.c_str(), &sb) == 0) {
        if (S_ISDIR(sb.st_mode)) {
            return getDirRegContents(path, obj_number);
        } else if (S_ISREG(sb.st_mode)) {
            return {path};
        }
    } else {
        if (errno == ENOENT || errno == EINVAL || errno == EACCES) {
            THROW_USER_EXCEPTION(3) << "The specified path \"" << path << "\" cannot be found or accessed";
        }
    }
    return {};
}

Blob::Ptr convertBlobFP32toFP16(Blob::Ptr blob) {
    Blob::Ptr weightsBlob =
            make_shared_blob<short>({ Precision::FP16, blob->getTensorDesc().getDims(), blob->getTensorDesc().getLayout()});
    weightsBlob->allocate();
    short* target = weightsBlob->buffer().as<short*>();
    float* source = blob->buffer().as<float*>();
    PrecisionUtils::f32tof16Arrays(target, source, blob->size(), 1.0f, 0.0f);
    return weightsBlob;
}

Blob::Ptr convertBlobFP16toFP32(Blob::Ptr blob) {
    Blob::Ptr weightsBlob =
            make_shared_blob<float>({ Precision::FP32, blob->getTensorDesc().getDims(), blob->getTensorDesc().getLayout()});
    weightsBlob->allocate();
    float* target = weightsBlob->buffer().as<float*>();
    short* source = blob->buffer().as<short *>();
    PrecisionUtils::f16tof32Arrays(target, source, blob->size(), 1.0f, 0.0f);
    return weightsBlob;
}

bool isFile(const std::string& path) {
    struct stat sb;
    return stat(path.c_str(), &sb) == 0 && S_ISREG(sb.st_mode);
}

bool isDirectory(const std::string& path) {
    struct stat sb;
    return stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
}
