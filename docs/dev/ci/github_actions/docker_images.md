# Docker Images

Most of the workflows in the OpenVINO GHA CI use [self-hosted machines with dynamically spawned runners](./runners.md)
to execute jobs.

To avoid the corruption of runners and machines, workflows use various Docker images
which introduce an additional layer of protection for self-hosted machines.

Docker images are specified for each job using the `container` key. See the
[GHA documentation](https://docs.github.com/en/actions/using-jobs/running-jobs-in-a-container)
for syntax reference.

An example `Build` job from the [`ubuntu_22.yml`](./../../../../.github/workflows/ubuntu_22.yml) workflow:
```yaml
Build:
  ...
  container:
    image: openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04
    volumes:
      - /mount:/mount
  ...
```

The `openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04` Docker image is used for this job.

Additionally, you can make the caches available in Docker containers using the `volumes` key.
For more details, refer to the [caches](./caches.md) page.

## Available Docker Images

The jobs in the workflows use Docker images based on each job's requirements.

Since self-hosted machines are hosted in [Microsoft Azure](https://azure.microsoft.com/en-us),
it is recommended to use Docker images hosted in an instance of [Azure Container Registry (ACR)](https://azure.microsoft.com/en-us/products/container-registry).

The ACR used for OpenVINO GHA CI is `openvinogithubactions.azurecr.io`.

Here are some pros and cons of having your own container registry:
* Pros:
  * No pull limits for images: Docker Hub imposes [limits](https://docs.docker.com/docker-hub/download-rate-limit/) on image pulls
  * Fast pulling speeds
* Cons:
  * The registry needs to be populated with the required images before usage
    * The registry does not automatically mirror images available on Docker Hub
    * The necessary images must be manually added to the registry

The available Docker images use the following pattern for their names: `openvinogithubactions.azurecr.io/<image-name>:<image-tag>` or `openvinogithubactions.azurecr.io/library/<image-name>:<image-tag>`

Most of the images on the OpenVINO ACR are either mirrors of the images from Docker Hub or based on them.

Examples:
* `openvinogithubactions.azurecr.io/library/ubuntu:22.04` corresponds to `ubuntu:22.04` from Docker Hub
* `openvinogithubactions.azurecr.io/nvidia/cuda:11.8.0-runtime-ubuntu20.04` corresponds to `nvidia/cuda:11.8.0-runtime-ubuntu20.04` from Docker Hub

Image mirroring is performed automatically via [ACR artifact cache rules](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-artifact-cache),
so no manual upload from a public DockerHub to OpenVINO ACR instance is required. 

## Custom docker images and handle_docker action

To optimize the time required to install dependencies in workflows and to make local reproduction of workflow steps easier, 
we create custom Docker images for different types of validation and use them in our workflows. 
The dockerfiles for these images are stored in the OpenVINO repository in [`.github/dockerfiles`](./../../../../.github/dockerfiles) 
folder. Dockerfiles are organized as follows:

* [`.github/dockerfiles/ov_build`](./../../../../.github/dockerfiles/ov_build) - contains environment setup for 
build workflows for different OSes (e.g., installation of [install_build_dependencies.sh](./../../../../install_build_dependencies.sh))
* [`.github/dockerfiles/ov_test`](./../../../../.github/dockerfiles/ov_test) - contains environment setup for 
test workflows for different OSes (e.g., installation of [scripts/install_dependencies/install_openvino_dependencies.sh](./../../../../scripts/install_dependencies/install_openvino_dependencies.sh))

The changes to these dockerfiles are getting checked and applied automatically in pre-commits via a custom action
[handle_docker](./../../../../.github/actions/handle_docker), which is executed in workflows before starting actual 
validation in a separate job called Docker. The action checks if a pull request changes either dockerfiles or 
environment setup scripts, and if so - triggers affected docker images build and validation with the updated images.
The images are tagged with the ID of the pull request that changes them, and the tag must be updated in git by changing 
PR ID in [.github/dockerfiles/docker_tag](./../../../../.github/dockerfiles/docker_tag). The action will prompt you 
to do that once you change something that alters docker environment.

**Important**: If you add a new environment configuration script to be used in dockerfiles, please, add its path under 
`category: docker_env` key in [.github/labeler.yml](./../../../../.github/labeler.yml) (this will 
ensure that the changes to this script are detected by handle_docker action and applied automatically), and exclude 
the path to this script from [.dockerignore](./../../../../.dockerignore) (this will make sure that Docker itself 
detects a script file).

The action accepts a list of the desired images to build as an input and outputs fully-qualified Docker image references
to use in workflow jobs.

### Using custom images in workflow jobs

* Make sure that Docker job is called in your workflow. Pass a path or multiple paths to the folders with dockerfiles, 
that are going to be used further in a workflow, to `images` parameter of the `handle_docker` action. Example:
```yaml
  Docker:
    needs: Smart_CI
    runs-on: aks-linux-4-cores-16gb-docker-build
    container:
      image: openvinogithubactions.azurecr.io/docker_build:0.2
      volumes:
        - /mount:/mount
    outputs:
      images: "${{ steps.handle_docker.outputs.images }}"
    steps:
      - name: Checkout
        uses: actions/checkout@692973e3d937129bcbf40652eb9f2f61becf3332 # v4.1.7

      - uses: ./.github/actions/handle_docker
        id: handle_docker
        with:
          images: |
            ov_build/ubuntu_22_04_x64
            ov_build/ubuntu_22_04_x64_nvidia
            ov_test/ubuntu_22_04_x64
          registry: 'openvinogithubactions.azurecr.io'
          dockerfiles_root_dir: '.github/dockerfiles'
          changed_components: ${{ needs.smart_ci.outputs.changed_components }}
```
* Add "Docker" to the `needs:` block of the job that will be executed with the desired custom image and set 
`container.image` key in this job to point to the docker image taken from `handle_docker`'s outputs, like that: 
```yaml
  Build:
    needs: [Smart_CI, Docker]
    ...
    runs-on: aks-linux-16-cores-32gb
    container:
      image: ${{ fromJSON(needs.docker.outputs.images).ov_build.ubuntu_22_04_x64 }}
```
**Note**, that Azure-based runner (aks-***) is required to use custom Docker images, since the access to our 
Azure Container Registry is enabled only from this type of runners.


## How to Choose an Image

The Docker image choice depends on the nature of the job and the configuration being tested. Please, review the images 
in [.github/dockerfiles](./../../../../.github/dockerfiles). Typically for jobs that require compilation you'll
choose **ov_build** images, and **ov_test** for others, as described in the previous section.

If no existing images suit your needs, and you need a separate Docker image with their own dependencies installed, 
feel free to add a new dockerfile under `.github/dockerfiles/<validation_type>/<platform>` folder and refer to it 
in a workflow as described above.

If you don't need any system dependencies to be installed in your pipeline, it might be enough to use a plain
image from DockerHub. In this case, refer to the mirror of this DockerHub image in our 
private Azure Container Registry. For example, if you need to use the `ubuntu:22.04` image, which you locally pull like that:
```
docker pull ubuntu:22.04
...
Status: Downloaded newer image for ubuntu:22.04
docker.io/library/ubuntu:22.04
```
Note the last line starting with `docker.io`, replace `docker.io` with `openvinogithubactions.azurecr.io` and use the 
resulting image reference in your workflow:
```yaml
  SomeJob:
    ...
    container:
      image: openvinogithubactions.azurecr.io/library/ubuntu:22.04
```
The image will be mirrored from DockerHub to our Azure Container Registry automatically.


## Local reproducibility

Custom docker images can also be built locally to reproduce CI results in the same environment:
1. Install [Docker](https://docs.docker.com/engine/install) on your local machine (Linux is the best option).
2. If you're running behind proxy on a host where you run a docker build, 
[add proxy](https://docs.docker.com/engine/cli/proxy/#configure-the-docker-client) to your `~/.docker/config.json`.
3. Clone openvino repo, go to the root of the repository, and execute `docker build` for a desired image, for example:
```
cd _git/openvino
docker build -f .github/dockerfiles/ov_test/ubuntu_22_04_x64/Dockerfile -t my_local_ov_test_build .
```
4. To run an interactive shell session within a built image:
```
docker run --entrypoint bash -it my_local_ov_test_build
```

Please, contact a member of the CI team for any assistance regarding Docker environment.
