# Docker Builds for ngraph with a _Reference-OS_

## Introduction

This directory contains a basic build system for creating docker images of the _reference-OS_ on which ngraph builds and unit tests are run.  The purpose is to provide reference builds for _Continuous Integration_ used in developing and testing ngraph.

The `Makefile` provides targets for:

* Building the _reference-OS_ into a docker image
* Building ngraph and running unit tests in this cloned repo, mounted into the docker image of the _reference-OS_
* Starting an interactive shell in the _reference-OS_ docker image, with the cloned repo available for manual builds and unit testing

The _make_ targets are designed to handle all aspects of building the _reference-OS_ docker image, running ngraph builds and unit testing in it, and opening up a session in the docker image for interactive use.  You should not need to issue any manual commands (unless you want to).  In addition the `Dockerfile.ngraph.*` files provide a description of how each _reference-OS_ environment is built, should you want to build your own server or docker image.

## Prerequisites

In order to use the _make_ targets, you will need to do the following:

* Have *docker* installed on your computer with the docker daemon running.
* For GPU support, also install *nvidia-docker* and start the nvidia-docker daemon.
* These scripts assume that you are able to run the `docker` command without using `sudo`.  You will need to add your account to the `docker` group so this is possible.
* If your computer (running docker) sits behind a firewall, you will need to have the docker daemon properly configured to use proxies to get through the firewall, so that public docker registries and git repositories can be accessed.
* You should _not_ run `make check_*` targets from a directory in an NFS filesystem, if that NFS filesystem uses _root squash_ (see **Notes** section below).  Instead, run `make check_*` targets from a cloned repo in a local filesystem.

## Make Targets

The _make_ targets are designed to provide easy commands to run actions using the docker image.  All _make_ targets should be issued on the host OS, and _not_ in a docker image.

GPU support will automatically be included for _make_ targets if the path of the `nvidia-smi` command is returned in response to `which nvidia-smi` on the host OS.

Most _make_ targets are structured in the form `<action>_<compiler>`.  The `<action>` indicates what you want to do (e.g. build, check, install), while the `<compiler>` indicates what you want to build with (i.e. gcc or clang).

* In general, you simply need to run the command **`make check_all`**.  This first makes the `build_docker_ngraph` target as a dependency.  Then it makes the `build_*` and `check_*` targets, which will build ngraph using _cmake_ and _make_ and then run unit testing.  Please keep in mind that `make check_*` targets do not work when your working directory is in an NFS filesystem that uses _root squash_ (see **Notes** section below).

* Two builds types are supported: building with `gcc` and `clang`.  Targets are named `*_gcc` and `*_clang` when they refer to building with a specific compiler, and the `*_all` targets are available to build with both compilers.  Output directories are BUILD-GCC and BUILD-CLANG, at the top level.

* You can also run the command **`make shell`** to start an interactive bash shell inside the docker image.  While this is not required for normal builds and unit testing, it allows you to run interactively within the docker image with the cloned repo mounted.  Again, `build_docker_ngraph` is made first as a dependency.  Please keep in mind that `make shell` does not work when your working directory is in an NFS filesystem that uses _root squash_ (see **Notes** section below).

* Running the command **`make build_docker_ngraph`** is also available, if you simply want to build the docker image.  This target does work properly when your working directory is in an NFS filesystem.

* Finally, **`make clean`** is available to clean up the BUILD-* and docker build directories.

Note that all operations performed inside the docker image are run as a regular user, using the `run-as-user.sh` script.  This is done to avoid writing root-owned files in mounted filesystems.

## Examples/Hints

* To build an Ubuntu 16.04 docker container, compile with gcc 5.4, and run unit tests:

```
cd contrib/docker
make check_gcc
```

* To build an Ubuntu 16.04 docker container, compile with clang 3.9, and run unit tests:

```
cd contrib/docker
make check_clang
```

* To build a CentOS 7.4 docker container, compile with gcc 4.8.5, and run unit tests:

```
cd contrib/docker
make check_gcc OS=centos74
```

## Helper Scripts

These helper scripts are included for use in the `Makefile` and automated (Jenkins) jobs.  **These scripts should _not_ be called directly unless you understand what they do.**

#### `build-ngraph-docs.sh`

A helper script to simplify implentation of the make_docs target using docker images.

#### `build-ngraph-and-test.sh`

A helper script to simplify implementation of make targets with multiple reference OS environments with different compilers using docker images.

#### `docker_cleanup.sh`

A helper script for Jenkins jobs to clean up old exited docker containers and `ngraph_*` docker images.

#### `make-dimage.sh`

A helper script to simplify building of docker images for multiple reference OS environments.

#### `run_as_user.sh`

A helper script to run as a normal user within the docker container.  This is done to avoid writing root-owned files in mounted filesystems.

#### `run_as_ubuntu_user.sh`

Same as `run_as_user.sh`, specifically called for _make_ targets with Ubuntu 16.04 docker containers.

#### `run_as_centos_user.sh`

A helper script to run as a normal user within a CentOS 7.4 docker container.

## Notes

* The top-level `Makefile` in this cloned repo can be used to build and unit-test ngraph _outside_ of docker.  This directory is only for building and running unit tests for ngraph in the _reference-OS_ docker image.

* Due to limitations in how docker mounts work, `make check_*` and `make shell` targets will fail if you try to run them while in a working directory that is in an NFS-mount that has _root squash_ enabled.  The cause results from the process in the docker container running as root.  When a file or directory is created by root in the mounted directory tree, from within the docker image, the NFS-mount (in the host OS) does not allow a root-created file, leading to a permissions error.  This is dependent on whether the host OS performs "root squash" when mounting NFS filesystems.  The fix to this is easy: run `make check_*` and `make shell` from a local filesystem.

* The _make_ targets have been tested with the following docker containers on an Ubuntu 16.04 host OS with docker installed and the docker daemon properly configured.  Some adjustments may be needed to run these on other OSes.

#### Ubuntu 16.04 (default)

```
Dockerfile: Dockerfile.ngraph.ubuntu1604_gpu
Reference-OS: Ubuntu 16.04
GPU Support: Yes
BUILD-GCC: gcc 5.4
BUILD-CLANG: clang 3.9
pre-built LLVM
```
```
Dockerfile: Dockerfile.ngraph.ubuntu1604
Reference-OS: Ubuntu 16.04
GPU Support: No
BUILD-GCC: gcc 5.4
BUILD-CLANG: clang 3.9
pre-built LLVM
```

#### CentOS 7.4

```
Dockerfile: Dockerfile.ngraph.centos74_gpu
Reference-OS: Centos 7.4.1708
GPU Support: Yes
BUILD-GCC: gcc 4.8
BUILD-CLANG: not supported
pre-built cmake3
LLVM built from source
```
```
Dockerfile: Dockerfile.ngraph.centos74
Reference-OS: Centos 7.4.1708
GPU Support: No
BUILD-GCC: gcc 4.8
BUILD-CLANG: not supported
pre-built cmake3
LLVM built from source
```
