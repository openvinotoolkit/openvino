# Overview of the Docker Images used in the OpenVINO GitHub Actions CI

Most of the workflows in the OpenVINO GHA CI are using [self-hosted machines with dynamically spawned runners](./runners.md) to execute jobs. 

To avoid corruption of the runners and machines, the workflows utilize various Docker images that introduce a layer of protection for the self-hosted machines.

The Docker images are specified for each job using the `container` key. See the [GHA documentation](https://docs.github.com/en/actions/using-jobs/running-jobs-in-a-container) for the syntax reference.

An example `Build` job from the [`linux.yml`](./../../../../.github/workflows/linux.yml) workflow:
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

Additionally, it is possible to make the caches available in the Docker containers using the `volumes` key. 
Read more about the available caches and how to choose one [here](./caches.md).

## Available Docker Images

The jobs in the workflows utilize appropriate Docker images based on a job's needs. 

As the self-hosted machines are hosted in [Microsoft Azure](https://azure.microsoft.com/en-us), 
it is optimal to use the Docker images hosted in an instance of [Azure Container Registry (ACR)](https://azure.microsoft.com/en-us/products/container-registry).

The ACR used for the OpenVINO GHA CI is `openvinogithubactions.azurecr.io`.

Some pros and cons of having own container registry are:
* pros:
  * No pull limits for the images
    * There are [limits](https://docs.docker.com/docker-hub/download-rate-limit/) for the pulls from Docker Hub
  * Fast pulling speeds
* cons:
  * The registry should be populated with needed images before usage
    * The registry does not mirror the images available on Docker Hub automatically
    * The needed images should be added manually to the registry

As the number of enabled workflows grew, so did the number of available Docker images.

The available Docker images are using the following pattern for their names: `openvinogithubactions.azurecr.io/dockerhub/<image-name>:<image-tag>`.

Most of the images on the OpenVINO ACR are mirrors of the images with the same names on Docker Hub.

The examples:
* `openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04` corresponds to `ubuntu:20.04` from Docker Hub
* `openvinogithubactions.azurecr.io/dockerhub/ubuntu:22.04` corresponds to `ubuntu:22.04` from Docker Hub
* `openvinogithubactions.azurecr.io/dockerhub/nvidia/cuda:11.8.0-runtime-ubuntu20.04` corresponds to `nvidia/cuda:11.8.0-runtime-ubuntu20.04` from Docker Hub

## How to choose an Image

The Docker image required for a job stems from the nature of the job and configuration that is being tested.

An example `Build` job from the [`linux.yml`](./../../../../.github/workflows/linux.yml) workflow:
```yaml
Build:
  ...
  container:
    image: openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04
    volumes:
      - /mount:/mount
  ...
```

The `openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04` Docker image is used for this job and **for other jobs in the workflow**.
Usually, if one Docker image is used for the building job, the other jobs would use the same image for testing. 

If the tests do not require any specific OS or distribution, it would be best to use the already available images: e.g., `openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04`.

If the plan is to test some specific OS or distribution (e.g., `fedora`), 
the Docker image for this flavour should be first uploaded to the OpenVINO ACR and only then used in a workflow. 

Contact someone from the CI team for assistance with the image uploading. 
