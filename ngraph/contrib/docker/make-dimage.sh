#!  /bin/bash

###
#
# Create a docker image that includes dependencies for building ngraph
#
# Uses CONTEXTDIR as the docker build context directory
#   Default value is '.'
#
# Uses ./Dockerfile.${DOCKER_TAG}
#   DOCKER_TAG is set to 'ngraph' if not set 
#
# Sets the docker image name as ${DOCKER_IMAGE_NAME}
#   DOCKER_IMAGE_NAME is set to the ${DOCKER_TAG} if not set in the environment
#   The datestamp tag is automatically appended to the DOCKER_IMAGE_NAME to create the DIMAGE_ID
#   The ${DIMAGE_ID} docker image is created on the local server
#   The ${DOCKER_IMAGE_NAME}:latest tag is also created by default for reference
#
###

set -e
#set -u
set -o pipefail

if [ -z $DOCKER_TAG ]; then
    DOCKER_TAG=build_ngraph
fi

if [ -z $DOCKER_IMAGE_NAME ]; then
    DOCKER_IMAGE_NAME=${DOCKER_TAG}
fi

echo "CONTEXTDIR=${CONTEXTDIR}"

if [ -z ${CONTEXTDIR} ]; then
    CONTEXTDIR='.'  # Docker image build context
fi

echo "CONTEXTDIR=${CONTEXTDIR}"

if [ -n $DFILE ]; then
    DFILE="${CONTEXTDIR}/Dockerfile.${DOCKER_TAG}"
fi

CONTEXT='.'

DIMAGE_NAME="${DOCKER_IMAGE_NAME}"
DIMAGE_VERSION=`date -Iseconds | sed -e 's/:/-/g'`

DIMAGE_ID="${DIMAGE_NAME}:${DIMAGE_VERSION}"

# If proxy settings are detected in the environment, make sure they are
# included on the docker-build command-line.  This mirrors a similar system
# in the Makefile.

if [ ! -z "${http_proxy}" ] ; then
    DOCKER_HTTP_PROXY="--build-arg http_proxy=${http_proxy}"
else
    DOCKER_HTTP_PROXY=' '
fi

if [ ! -z "${https_proxy}" ] ; then
    DOCKER_HTTPS_PROXY="--build-arg https_proxy=${https_proxy}"
else
    DOCKER_HTTPS_PROXY=' '
fi

cd ${CONTEXTDIR}

echo ' '
echo "Building docker image ${DIMAGE_ID} from Dockerfile ${DFILE}, context ${CONTEXT}"
echo ' '

# build the docker base image
docker build  --rm=true \
       ${DOCKER_HTTP_PROXY} ${DOCKER_HTTPS_PROXY} \
       -f="${DFILE}" \
       -t="${DIMAGE_ID}" \
       ${CONTEXT}

docker tag  "${DIMAGE_ID}"  "${DIMAGE_NAME}:latest"

echo ' '
echo 'Docker image build completed'
echo ' '
