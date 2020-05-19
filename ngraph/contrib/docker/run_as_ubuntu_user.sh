#!  /bin/bash

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

# ******************************************************************************
# This script is designed to simulate running as a user with a particular UID
# within a docker container.
#
# Normally a docker container runs as root, which can cause problems with file
# ownership when a host directory tree is mounted into the docker container.
# There are other problems with building and running software as root as
# well.  Good practice when validating software builds in a docker container
# is to run as a normal user, since many (most?) end users will not be building
# and installing software as root.
#
# This script should be run using "docker run", with RUN_UID (set to the user
# you want to run as) passed into the docker container as an environment
# variable.  The script will then add the UID as user "dockuser" to
# /etc/passwd (important for some software, like bazel), add the new dockuser
# to the sudo group (whether or not sudo is installed), and su to a new shell
# as the dockuser (passing in the existing environment, which is important).
#
# If the environment variable RUN_CMD is passed into the docker container, then
# this script will use RUN_CMD as a command to run when su'ing.  If RUN_CMD is
# not defined, then /bin/bash will run, which effectively provides an
# interactive shell in the docker container, for debugging.
# ******************************************************************************

set -e  # Make sure we exit on any command that returns non-zero
set -u  # No unset variables

if [ -z "$RUN_UID" ] ; then

    # >&2 redirects echo output to stderr.
    # See: https://stackoverflow.com/questions/2990414/echo-that-outputs-to-stderr
    ( >&2 echo 'ERROR: Environment variable RUN_UID was not set when run-as-user.sh was run' )
    ( >&2 echo '       Running as default user (root, in docker)' )
    ( >&2 echo ' ' )

    exit 1

else

    # The username used in the docker container to map the caller UID to
    #
    # Note 'dockuser' is used in other scripts, notably Makefile.  If you
    # choose to change it here, then you need to change it in all other
    # scripts, or else the builds will break.
    #
    DOCK_USER='dockuser'

    # We will be su'ing using a non-login shell or command, and preserving
    # the environment.  This is done so that env. variables passed in with
    # "docker run --env ..." are honored.
    # Therefore, we need to reset at least HOME=/root ...
    #
    # Note also that /home/dockuser is used in other scripts, notably
    # Makefile.  If you choose to change it here, then you need to change it
    # in all other scripts, or else the builds will break.
    #
    export HOME="/home/${DOCK_USER}"

    # Make sure the home directory is owned by the new user
    if [ -d "${HOME}" ] ; then
      chown "${RUN_UID}" "${HOME}"
    fi

    # Add a user with UID of person running docker (in ${RUN_UID})
    # If $HOME does not yet exist, then it will be created
    adduser --disabled-password --gecos 'Docker-User' -u "${RUN_UID}" "${DOCK_USER}"
    # Add dockuser to the sudo group
    adduser "${DOCK_USER}" sudo

    # If root access is needed in the docker image while running as a normal
    # user, uncomment this and add 'sudo' as a package installed in Dockerfile
    # echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

    if [ -z "${RUN_CMD+x}" ] ; then  # Launch a shell as dockuser
      su -m "${DOCK_USER}" -c /bin/bash
    else                         # Run command as dockuser
      su -m "${DOCK_USER}" -c "${RUN_CMD}"
    fi

fi
