# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

# Basic Makefile for contrib/docker. This can be expanded later as more targets
# are added.

# Building LLVM from source has been observed to trigger the oom-killer
#    on systems with a large number of cores
#    running with make -j
#
# Default is to build with -j 22 for parallel cmake/make.
# Override with make PARALLEL="-j <num_parallel_processes>" where
#    <num_parallel_processes> = the number of make processes to run in parallel
# Turn off with make PARALLEL=
PARALLEL=22

# DIR is an internal variable that serves as an anchor to this cloned git
# repository.  DIR is mounted into the docker container, so that builds
# can occur within the container on this cloned git repository.  DIR should
# not be modified - if it is, then the build system will not work.
DIR = $(realpath ../..)

# DOCKUSER_HOME is the location of the home directory of the fabricated
# "dockuser" user, used only within the docker containers.  "dockuser" is
# created (from the passed-in RUN_UID) to map the docker-caller user's UID to a
# first-class user (/etc/passwd entry, member of sudo group, proper home dir)
# /home/dockuser is also used in other scripts, notably run_as_user.sh, so if
# changed it must be done in other areas for the builds to work.
DOCKUSER_HOME=/home/dockuser

# Use /home/dockuser/ngraph-test, because we run as the user (and not root)
# /root/ngraph-test is not used, because /root is not accessible to user
VOLUME = -v "${DIR}:${DOCKUSER_HOME}/ngraph-test"
GIT_COMMIT = $(shell git rev-parse HEAD)
DBUILD_VERSION = ${GIT_COMMIT}_${PYTHON_VERSION}

# Look for evidence if GPU backend is supported on the platform
NVIDIA_SMI = $(shell which nvidia-smi)

# Enable additional options to be added on the command line
ifndef CMAKE_OPTIONS_EXTRA
    CMAKE_OPTIONS_EXTRA=
endif

# Allow linking pre-built third-party cache files (future)
ifndef THIRD_PARTY_CACHE_DIR
    THIRD_PARTY_CACHE_DIR=
endif

# OS set to 'ubuntu1604' by default
# can be overridden on the command line with 'make <target> OS=centos74"
ifndef OS
    OS="ubuntu1604"
endif

DBUILD_DIR = ${DIR}/contrib/docker/.build-${DBUILD_VERSION}_${OS}

# Configuration for specific reference OS in Dockerfiles
ifeq ("$(shell echo ${OS} | grep centos)","centos74")
    RUN_AS_USER_SCRIPT=${DOCKUSER_HOME}/ngraph-test/contrib/docker/run_as_centos_user.sh
    CPU_DOCKERFILE=Dockerfile.ngraph.centos74
else
    CPU_DOCKERFILE="Dockerfile.ngraph.ubuntu1604"
    RUN_AS_USER_SCRIPT=${DOCKUSER_HOME}/ngraph-test/contrib/docker/run_as_ubuntu_user.sh
endif

# Build GPU backend if NVIDIA_SMI command is found
# Sets CMAKE_OPTIONS_EXTRA to introduce GPU build configuration to cmake
# Configuration for GPU backend in Dockerfiles with "_gpu" suffix
# The nvidia-docker command must be used for any targets that actually utilize GPU devices
ifneq ("$(shell echo ${NVIDIA_SMI} | grep nvidia-smi)","")
    DOCKERFILE=${CPU_DOCKERFILE}_gpu
    DOCKER_CMD=nvidia-docker
    DOCKER_CMD_MESG=GPU appears to be supported on this platform.  Building for GPU and CPU backend support.
else
    DOCKERFILE=${CPU_DOCKERFILE}
    DOCKER_CMD=docker
    DOCKER_CMD_MESG=GPU does not appear to be supported on this platform.  Building for CPU backend support only.
endif

# For gcc builds, we do NOT regard warnings as errors
# For clang builds, we DO make warnings into errors
CMAKE_OPTIONS_COMMON=-DNGRAPH_BUILD_DOXYGEN_DOCS=ON -DNGRAPH_BUILD_SPHINX_DOCS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo $(CMAKE_OPTIONS_EXTRA)
CMAKE_OPTIONS_GCC=$(CMAKE_OPTIONS_COMMON) -DCMAKE_INSTALL_PREFIX=${DOCKUSER_HOME}/ngraph-test/BUILD-GCC/ngraph_dist
CMAKE_OPTIONS_CLANG=$(MAKE_OPTIONS_COMMON)-DCMAKE_INSTALL_PREFIX=${DOCKUSER_HOME}/ngraph-test/BUILD-CLANG/ngraph_dist -DCMAKE_CXX_COMPILER=clang++-3.9 -DCMAKE_C_COMPILER=clang-3.9 -DNGRAPH_WARNINGS_AS_ERRORS=ON

CALLER_UID := $(shell id -u)
CALLER_GID := $(shell id -g)

# Default version is python 2, but can be switched to 3 from command
# line
PYTHON_VERSION = 2

# Some targets are DEPRECATED and will be removed at a later date: check_cpu build_ngraph_cpu
# These DEPRECATED targets are currently included for Jenkins job compatibility with older dev branches
# Please see comments for individual targets for more details
.PHONY: clean build_docker_image build_gcc check_gcc build_clang check_clang install_gcc install_clang shell check_cpu build_all build_ngraph_cpu

DOCKER_BUILD=docker build --rm=true

ifdef http_proxy
DOCKER_BUILD+=--build-arg http_proxy=$(http_proxy)
DOCKER_RUN_ENV+=--env "http_proxy=$(http_proxy)"
endif

ifdef https_proxy
DOCKER_BUILD+=--build-arg https_proxy=$(https_proxy)
DOCKER_RUN_ENV+=--env "https_proxy=$(https_proxy)"
endif

all: check_gcc check_clang

# Docker actions

# Isolate specific dockerfiles in a .build_* subdirectory
expand_dockerfile_templates:
	@echo "OS=${OS}"
	@echo "DOCKERFILE=${DOCKERFILE}"
	@echo "RUN_AS_USER_SCRIPT=${RUN_AS_USER_SCRIPT}"
	cd "${DIR}"/contrib/docker
	mkdir "${DBUILD_DIR}" || true
	sed -e 's/\(FROM ngraph.*\)/\1:${DBUILD_VERSION}/' ${DOCKERFILE} > "${DBUILD_DIR}/Dockerfile.build_ngraph_${OS}"

build_docker_image: expand_dockerfile_templates
	@echo "OS=${OS}"
	@echo ${DBUILD_DIR}
	export CONTEXTDIR=${DBUILD_DIR};export DOCKER_TAG=build_ngraph_${OS};./make-dimage.sh
	docker tag build_ngraph_${OS}:latest build_ngraph_${OS}:${DBUILD_VERSION}

build_docker: build_docker_image

# Build docs
docs: sphinx_doc

# Docs build does not depend on GPU dependencies
sphinx_doc: build_docker_image
	# sphinx html docs build
	docker run --rm --tty \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
            --env RUN_UID="$(shell id -u)" \
            --env RUN_CMD="${DOCKUSER_HOME}/ngraph-test/contrib/docker/build-ngraph-docs.sh" \
            "build_ngraph_${OS}:${DBUILD_VERSION}" \
	    sh -c "cd ${DOCKUSER_HOME}; ${RUN_AS_USER_SCRIPT}"

# Build
build_all: build_gcc build_clang

# Build targets ALWAYS clean build directories (BUILD-GCC, BUILD-CLANG) prior to building
# Always use docker command to build docker images
# nvidia-docker command is not appropriate
build_gcc: build_docker_image
	@echo ""
	@echo "${DOCKER_CMD_MESG}"
	@echo ""
	docker run --rm --tty \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
	    --env BUILD_SUBDIR=BUILD-GCC \
	    --env CMAKE_OPTIONS_EXTRA="${CMAKE_OPTIONS_EXTRA}" \
	    --env PARALLEL=${PARALLEL} \
	    --env THIRD_PARTY_CACHE_DIR=${THIRD_PARTY_CACHE_DIR} \
	    --env CMD_TO_RUN='build_gcc' \
            --env RUN_UID="$(shell id -u)" \
            --env RUN_CMD="${DOCKUSER_HOME}/ngraph-test/contrib/docker/build-ngraph-and-test.sh" \
            "build_ngraph_${OS}:${DBUILD_VERSION}" \
	    sh -c "cd ${DOCKUSER_HOME}; ${RUN_AS_USER_SCRIPT}"

# Build targets ALWAYS clean build directories (BUILD-GCC, BUILD-CLANG) prior to building
# Always use docker command to build docker images
# nvidia-docker command is not appropriate
build_clang: build_docker_image
	@echo ""
	@echo "${DOCKER_CMD_MESG}"
	@echo ""
	docker run --rm --tty \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
	    --env BUILD_SUBDIR=BUILD-CLANG \
	    --env CMAKE_OPTIONS_EXTRA="${CMAKE_OPTIONS_EXTRA}" \
	    --env PARALLEL=${PARALLEL} \
	    --env THIRD_PARTY_CACHE_DIR=${THIRD_PARTY_CACHE_DIR} \
	    --env CMD_TO_RUN='build_clang' \
            --env RUN_UID="$(shell id -u)" \
            --env RUN_CMD="${DOCKUSER_HOME}/ngraph-test/contrib/docker/build-ngraph-and-test.sh" \
            "build_ngraph_${OS}:${DBUILD_VERSION}" \
	    sh -c "cd ${DOCKUSER_HOME}; ${RUN_AS_USER_SCRIPT}"

# Check (run unit-tests)
check_all: check_gcc check_clang

# Always use the platform-specific docker command to run unit tests
# ngraph make check target executes unit-test-check and style-check
check_gcc: build_gcc
	${DOCKER_CMD} run --rm --tty \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
	    --env BUILD_SUBDIR=BUILD-GCC \
	    --env CMAKE_OPTIONS_EXTRA="${CMAKE_OPTIONS_EXTRA}" \
	    --env PARALLEL=${PARALLEL} \
	    --env THIRD_PARTY_CACHE_DIR=${THIRD_PARTY_CACHE_DIR} \
	    --env CMD_TO_RUN=check_gcc \
            --env RUN_UID="$(shell id -u)" \
            --env RUN_CMD="${DOCKUSER_HOME}/ngraph-test/contrib/docker/build-ngraph-and-test.sh" \
            "build_ngraph_${OS}:${DBUILD_VERSION}" \
	    sh -c "cd ${DOCKUSER_HOME}; ${RUN_AS_USER_SCRIPT}"

# Always use the platform-specific docker command to run unit tests
# ngraph make check target executes unit-test-check and style-check
check_clang: build_clang
	${DOCKER_CMD} run --rm --tty \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
	    --env BUILD_SUBDIR=BUILD-CLANG \
	    --env CMAKE_OPTIONS_EXTRA="${CMAKE_OPTIONS_EXTRA}" \
	    --env PARALLEL=${PARALLEL} \
	    --env THIRD_PARTY_CACHE_DIR=${THIRD_PARTY_CACHE_DIR} \
	    --env CMD_TO_RUN=check_clang \
            --env RUN_UID="$(shell id -u)" \
            --env RUN_CMD="${DOCKUSER_HOME}/ngraph-test/contrib/docker/build-ngraph-and-test.sh" \
            "build_ngraph_${OS}:${DBUILD_VERSION}" \
	    sh -c "cd ${DOCKUSER_HOME}; ${RUN_AS_USER_SCRIPT}"

# Always use the platform-specific docker command to run unit tests
unit_test_check_gcc: build_gcc
	${DOCKER_CMD} run --rm --tty \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
	    --env BUILD_SUBDIR=BUILD-GCC \
	    --env CMAKE_OPTIONS_EXTRA="${CMAKE_OPTIONS_EXTRA}" \
	    --env PARALLEL=${PARALLEL} \
	    --env THIRD_PARTY_CACHE_DIR=${THIRD_PARTY_CACHE_DIR} \
	    --env CMD_TO_RUN='unit-test-check_gcc' \
            --env RUN_UID="$(shell id -u)" \
            --env RUN_CMD="${DOCKUSER_HOME}/ngraph-test/contrib/docker/build-ngraph-and-test.sh" \
            "build_ngraph_${OS}:${DBUILD_VERSION}" \
	    sh -c "cd ${DOCKUSER_HOME}; ${RUN_AS_USER_SCRIPT}"

# Always use the platform-specific docker command to run unit tests
unit_test_check_clang: build_clang
	${DOCKER_CMD} run --rm --tty \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
	    --env BUILD_SUBDIR=BUILD-CLANG \
	    --env CMAKE_OPTIONS_EXTRA="${CMAKE_OPTIONS_EXTRA}" \
	    --env PARALLEL=${PARALLEL} \
	    --env THIRD_PARTY_CACHE_DIR=${THIRD_PARTY_CACHE_DIR} \
	    --env CMD_TO_RUN='unit-test-check_clang' \
            --env RUN_UID="$(shell id -u)" \
            --env RUN_CMD="${DOCKUSER_HOME}/ngraph-test/contrib/docker/build-ngraph-and-test.sh" \
            "build_ngraph_${OS}:${DBUILD_VERSION}" \
	    sh -c "cd ${DOCKUSER_HOME}; ${RUN_AS_USER_SCRIPT}"

style_check_clang: build_clang

# Install

install_all: install_gcc install_clang

# install targets do not depend on GPU dependencies
# no unit tests are executed
# build prerequisites include GPU dependencies in the docker image automatically
install_gcc: build_gcc
	docker run --rm --tty \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
	    --env BUILD_SUBDIR=BUILD-GCC \
	    --env CMAKE_OPTIONS_EXTRA="${CMAKE_OPTIONS_EXTRA}" \
	    --env PARALLEL=${PARALLEL} \
	    --env THIRD_PARTY_CACHE_DIR=${THIRD_PARTY_CACHE_DIR} \
	    --env CMD_TO_RUN=install_gcc \
            --env RUN_UID="$(shell id -u)" \
            --env RUN_CMD="${DOCKUSER_HOME}/ngraph-test/contrib/docker/build-ngraph-and-test.sh" \
            "build_ngraph_${OS}:${DBUILD_VERSION}" \
	    sh -c "cd ${DOCKUSER_HOME}; ${RUN_AS_USER_SCRIPT}"

# install targets do not depend on GPU dependencies
# no unit tests are executed
# build prerequisites include GPU dependencies in the docker image automatically
install_clang: build_clang
	docker run --rm --tty \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
	    --env BUILD_SUBDIR=BUILD-CLANG \
	    --env CMAKE_OPTIONS_EXTRA="${CMAKE_OPTIONS_EXTRA}" \
	    --env PARALLEL=${PARALLEL} \
	    --env THIRD_PARTY_CACHE_DIR=${THIRD_PARTY_CACHE_DIR} \
	    --env CMD_TO_RUN=install_clang \
            --env RUN_UID="$(shell id -u)" \
            --env RUN_CMD="${DOCKUSER_HOME}/ngraph-test/contrib/docker/build-ngraph-and-test.sh" \
            "build_ngraph_${OS}:${DBUILD_VERSION}" \
	    sh -c "cd ${DOCKUSER_HOME}; ${RUN_AS_USER_SCRIPT}"

# Interactive shell

# Always use the platform-specific docker command for the interactive shell
shell: build_docker_image
	# "make shell" runs an interactive shell in the docker image, for debugging
	@echo "${DOCKER_CMD_MESG}"
	${DOCKER_CMD} run --rm --tty --interactive \
            ${VOLUME} \
	    ${DOCKER_RUN_ENV} \
            --env RUN_UID="$(shell id -u)" \
            "build_ngraph_${OS}:${DBUILD_VERSION}" \
	    sh -c "cd ${DOCKUSER_HOME}; ${RUN_AS_USER_SCRIPT}"

# Clean

clean:
	rm -f "${DIR}"/contrib/docker/.build-*/Dockerfile.* || echo "keep going if files are not present"
	rmdir "${DIR}"/contrib/docker/.build-* || echo "keep going if directory is not present"
	rm -fr "${DIR}"/BUILD-GCC
	rm -fr "${DIR}"/BUILD-CLANG
	rm -fr "${DIR}"/BUILD-DOCS
