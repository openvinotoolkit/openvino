#!  /bin/bash

#set -x
set -e
set -o pipefail


# Debugging
if [ -f "/etc/centos-release" ]; then
    cat /etc/centos-release
fi

if [ -f "/etc/lsb-release" ]; then
    cat /etc/lsb-release
fi

uname -a
cat /etc/os-release || true

echo ' '
echo 'Contents of /home:'
ls -la /home
echo ' '
echo 'Contents of /home/dockuser:'
ls -la /home/dockuser
echo ' '

if [ -z ${CMD_TO_RUN} ] ; then
    CMD_TO_RUN="make html"
fi

export NGRAPH_REPO=/home/dockuser/ngraph-test

if [ -z ${BUILD_SUBDIR} ] ; then
    BUILD_DIR="${NGRAPH_REPO}/doc/sphinx"
else
    BUILD_DIR="${NGRAPH_REPO}/${BUILD_SUBDIR}"
fi

if [ -z ${OUTPUT_DIR} ]; then
    OUTPUT_DIR="${NGRAPH_REPO}/BUILD-DOCS"
fi

# Print the environment, for debugging
echo ' '
echo 'Environment:'
export
echo ' '

# Remove old OUTPUT_DIR directory if present
( test -d ${OUTPUT_DIR} && rm -fr ${OUTPUT_DIR} && echo "Removed old ${OUTPUT_DIR} directory" ) || echo "Previous ${OUTPUT_DIR} directory not found"
# Make OUTPUT_DIR directory as user
mkdir -p ${OUTPUT_DIR}
chmod ug+rwx ${OUTPUT_DIR}

# build html docs
cd ${BUILD_DIR}
echo "Building docs in `pwd`:"
env VERBOSE=1 ${CMD_TO_RUN} 2>&1 | tee ${OUTPUT_DIR}/make_docs.log
ls -l build/html/* || true
mv build/ ${OUTPUT_DIR}
