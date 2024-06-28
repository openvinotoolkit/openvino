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

The available Docker images use the following pattern for their names: `openvinogithubactions.azurecr.io/dockerhub/<image-name>:<image-tag>`.

Most of the images on the OpenVINO ACR are mirrors of the images with the same names on Docker Hub.

Examples:
* `openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04` corresponds to `ubuntu:20.04` from Docker Hub
* `openvinogithubactions.azurecr.io/dockerhub/ubuntu:22.04` corresponds to `ubuntu:22.04` from Docker Hub
* `openvinogithubactions.azurecr.io/dockerhub/nvidia/cuda:11.8.0-runtime-ubuntu20.04` corresponds to `nvidia/cuda:11.8.0-runtime-ubuntu20.04` from Docker Hub

## How to Choose an Image

The Docker image choice depends on the nature of the job and the configuration being tested.

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

The `openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04` Docker image is used for this job
and **for other jobs in the workflow**.
Usually, if one Docker image is used for the building job, other jobs use the same image for testing.

If the tests do not require any specific OS or distribution, it is recommended to use
the available images, for example, `openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04`.

If testing requires a specific OS or distribution (for example, `fedora`),
the Docker image for this flavor should be uploaded to the OpenVINO ACR first and
then used in a workflow.

Contact a member of the CI team for assistance with the image uploading.
