#!/usr/bin/env bash
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e
#set -x
#set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]-$0}" )" >/dev/null 2>&1 && pwd )"

url_root=https://storage.openvinotoolkit.org/repositories/openvino/packages/master/opencv
[ -n "${OPENVINO_OPENCV_DOWNLOAD_SERVER}" ] && url_root="${OPENVINO_OPENCV_DOWNLOAD_SERVER}"
grep -q "Ubuntu 18" /etc/os-release && fname=ubuntu18.tgz
grep -q "Ubuntu 20" /etc/os-release && fname=ubuntu20.tgz
grep -q "CentOS Linux 7" /etc/os-release && fname=centos7.tgz
grep -q "CentOS Linux 8" /etc/os-release && fname=ubuntu20.tgz # yes ubuntu20
[[ "$OSTYPE" == "darwin"* ]] && fname=osx.tgz
[ -z "${fname}" ] && echo "Unsupported OS" && exit
archive_file="${TMPDIR:-/tmp}/openvino_opencv_linux.tgz"

echo "=== Download"
wget "${url_root}/${fname}" -O "${archive_file}"
echo "=== Unpack"
(cd "${SCRIPT_DIR}/../.." && tar -xf "${archive_file}")
echo "=== Cleanup"
rm "${archive_file}"
