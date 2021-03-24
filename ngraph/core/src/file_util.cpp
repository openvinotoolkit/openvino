//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <ftw.h>
#include <sys/file.h>
#include <sys/time.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include "ngraph/env_util.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/log.hpp"

#ifdef _WIN32
#define RMDIR(a) RemoveDirectoryA(a)
#define RMFILE(a) DeleteFileA(a)
#else
#define RMDIR(a) rmdir(a)
#define RMFILE(a) remove(a)
#ifdef ENABLE_UNICODE_PATH_SUPPORT
#include <codecvt>
#include <locale>
#endif
#endif

using namespace std;
using namespace ngraph;

string file_util::get_file_name(const string& s)
{
    string rc = s;
    auto pos = s.find_last_of('/');
    if (pos != string::npos)
    {
        rc = s.substr(pos + 1);
    }
    return rc;
}

string file_util::get_file_ext(const string& s)
{
    string rc = get_file_name(s);
    auto pos = rc.find_last_of('.');
    if (pos != string::npos)
    {
        rc = rc.substr(pos);
    }
    else
    {
        rc = "";
    }
    return rc;
}

string file_util::get_directory(const string& s)
{
    string rc = s;
    // Linux-style separator
    auto pos = s.find_last_of('/');
    if (pos != string::npos)
    {
        rc = s.substr(0, pos);
        return rc;
    }
    // Windows-style separator
    pos = s.find_last_of('\\');
    if (pos != string::npos)
    {
        rc = s.substr(0, pos);
        return rc;
    }
    return rc;
}

string file_util::path_join(const string& s1, const string& s2, const string& s3)
{
    return path_join(path_join(s1, s2), s3);
}

string file_util::path_join(const string& s1, const string& s2, const string& s3, const string& s4)
{
    return path_join(path_join(path_join(s1, s2), s3), s4);
}

string file_util::path_join(const string& s1, const string& s2)
{
    string rc;
    if (s2.size() > 0)
    {
        if (s2[0] == '/')
        {
            rc = s2;
        }
        else if (s1.size() > 0)
        {
            rc = s1;
            if (rc[rc.size() - 1] != '/')
            {
                rc += "/";
            }
            rc += s2;
        }
        else
        {
            rc = s2;
        }
    }
    else
    {
        rc = s1;
    }
    return rc;
}

#ifndef _WIN32
static void iterate_files_worker(const string& path,
                                 function<void(const string& file, bool is_dir)> func,
                                 bool recurse,
                                 bool include_links)
{
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(path.c_str())) != nullptr)
    {
        try
        {
            while ((ent = readdir(dir)) != nullptr)
            {
                string name = ent->d_name;
                string path_name = file_util::path_join(path, name);
                switch (ent->d_type)
                {
                case DT_DIR:
                    if (name != "." && name != "..")
                    {
                        if (recurse)
                        {
                            file_util::iterate_files(path_name, func, recurse);
                        }
                        func(path_name, true);
                    }
                    break;
                case DT_LNK:
                    if (include_links)
                    {
                        func(path_name, false);
                    }
                    break;
                case DT_REG: func(path_name, false); break;
                default: break;
                }
            }
        }
        catch (...)
        {
            exception_ptr p = current_exception();
            closedir(dir);
            rethrow_exception(p);
        }
        closedir(dir);
    }
    else
    {
        throw runtime_error("error enumerating file " + path);
    }
}
#endif

void file_util::iterate_files(const string& path,
                              function<void(const string& file, bool is_dir)> func,
                              bool recurse,
                              bool include_links)
{
    vector<string> files;
    vector<string> dirs;
#ifdef _WIN32
    std::string file_match = path_join(path, "*");
    WIN32_FIND_DATAA data;
    HANDLE hFind = FindFirstFileA(file_match.c_str(), &data);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        do
        {
            bool is_dir = data.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY;
            if (is_dir)
            {
                if (string(data.cFileName) != "." && string(data.cFileName) != "..")
                {
                    string dir_path = path_join(path, data.cFileName);
                    if (recurse)
                    {
                        iterate_files(dir_path, func, recurse);
                    }
                    func(dir_path, true);
                }
            }
            else
            {
                string file_name = path_join(path, data.cFileName);
                func(file_name, false);
            }
        } while (FindNextFileA(hFind, &data));
        FindClose(hFind);
    }
#else
    iterate_files_worker(
        path,
        [&files, &dirs](const string& file, bool is_dir) {
            if (is_dir)
            {
                dirs.push_back(file);
            }
            else
            {
                files.push_back(file);
            }
        },
        recurse,
        include_links);
#endif

    for (auto f : files)
    {
        func(f, false);
    }
    for (auto f : dirs)
    {
        func(f, true);
    }
}

std::string file_util::sanitize_path(const std::string& path)
{
    const auto colon_pos = path.find(":");
    const auto sanitized_path = path.substr(colon_pos == std::string::npos ? 0 : colon_pos + 1);
    const std::string to_erase = "/.\\";
    const auto start = sanitized_path.find_first_not_of(to_erase);
    return (start == std::string::npos) ? "" : sanitized_path.substr(start);
}

NGRAPH_API void file_util::convert_path_win_style(std::string& path)
{
    std::replace(path.begin(), path.end(), '/', '\\');
}

#ifdef ENABLE_UNICODE_PATH_SUPPORT

std::string file_util::wstring_to_string(const std::wstring& wstr)
{
#ifdef _WIN32
    int size_needed =
        WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL); // NOLINT
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(
        CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL); // NOLINT
    return strTo;
#else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_decoder;
    return wstring_decoder.to_bytes(wstr);
#endif
}

std::wstring file_util::multi_byte_char_to_wstring(const char* str)
{
#ifdef _WIN32
    int strSize = static_cast<int>(std::strlen(str));
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str, strSize, NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str, strSize, &wstrTo[0], size_needed);
    return wstrTo;
#else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_encoder;
    std::wstring result = wstring_encoder.from_bytes(str);
    return result;
#endif
}
#endif // ENABLE_UNICODE_PATH_SUPPORT
