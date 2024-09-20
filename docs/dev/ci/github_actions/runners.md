# OpenVINO Runners used by GitHub Actions CI

The machines that execute workflow commands are referred to as _runners_ in GitHub Actions.

Two types of runners are available in this repository:

* [GitHub Actions Runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners) - runners provided and managed by GitHub
* [Self-hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners) - runners created and managed by the OpenVINO CI team and linked to the OpenVINO repositories

Generally, it is advised to use the GitHub Actions runners for light jobs, like labelers, code style checks, etc, whereas
longer workflows (such as builds or functional tests) should use the self-hosted runners.

The runners are specified for each job using the `runs-on` key.

An example `Build` job from the [`ubuntu_22.yml`](./../../../../.github/workflows/ubuntu_22.yml)
workflow, using the `aks-linux-16-cores-32gb` runner group:

```yaml
Build:
  ...
  runs-on: aks-linux-16-cores-32gb
  ...
```


## Available GitHub Actions Runners

GitHub provides runners with different combinations of available resources and software.
OpenVINO repositories make use of the following runners:

* [default runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources),
  used for not-so-intensive memory and CPU tasks: `ubuntu-22/20.04`, `windows-2019/2022`,
  `macos-12/13`, etc.

* [larger runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-larger-runners/about-larger-runners#machine-sizes-for-larger-runners),
  used for memory and CPU-intensive tasks, listed in [the runners page](https://github.com/openvinotoolkit/openvino/actions/runners).


## Available Self-hosted Runners

Two groups of self-hosted runners are available:

* Dynamically-spawned Linux and Windows runners with CPU-only capabilities
* Dedicated Linux runners with GPU capabilities

### Dynamically-spawned Linux and Windows Runners

These self-hosted runners are dynamically spawned for each requested pipeline.
Several configurations are available, which are identified by different group names.
The group names generally follow the pattern:
`aks-{OS}-{CORES_N}-cores-|{RAM_SIZE}gb|-|{ARCH}|`, where:

* `{OS}` - the operating system: `win`/`linux` (currently, only the Windows and Linux
  self-hosted runners are available).
* `{CORES_N}` - the number of cores available to the runners in the group: `4`/`8`/etc.
* `|{RAM_SIZE}gb|` - **_optional_**, the RAM size in GB available to the runners in the group: `8`/`16`/etc.
  * **Note**: The groups with unspecified `{RAM_SIZE}` consist of the runners with 32 GB of RAM
* `|{ARCH}|` - **_optional_**, the architecture of the runners in the group: `arm`
  * **Note**: The groups with unspecified `{ARCH}` consist of the `x86_64` runners

Examples:
* `aks-win-16-cores-32gb` - the Windows x86_64 runners with 16 cores and 32 GB of RAM available
* `aks-linux-16-cores-arm` - the Linux ARM64 runners with 16 cores and 32 GB of RAM available

The available configurations are:

|             | Group Name        | CPU Cores    | RAM in GB        | Architecture         | Examples                                           |
|-------------|-------------------|--------------|------------------|----------------------|----------------------------------------------------|
| Windows     | `aks-win-*`       | `4`/`8`/`16` | `8`/`16`/`32`    | `x86_64`<sup>*</sup> | `aks-win-4-cores-8gb`/`aks-win-16-cores-32gb`      |
| Linux       | `aks-linux-*`     | `4`/`8`/`16` | `16`/`32`        | `x86_64`<sup>*</sup> | `aks-linux-4-cores-16gb`/`aks-linux-16-cores-32gb` |
| Linux ARM64 | `aks-linux-*-arm` | `16`         | `32`<sup>*</sup> | `arm`                | `aks-linux-16-cores-arm`                           |

* `*` - Not specified in the group name

### Dedicated GPU Runners

Eighteen runners with GPU capabilities (both integrated and discrete, iGPU and dGPU) 
are available to all repositories in the OpenVINO organisation's GitHub Actions. 

These runners are virtual machines with Ubuntu 22.04 and have the following specifications:

* CPU: i9-12900k
* All of them have iGPU (UHD 770 Graphics)
* Twelve of them have iGPU (UHD 770 Graphics) and dGPU (Arc A770)

These runners may be selected using labels provided in the `runs-on` field in a job configuration.
The available labels are:
* `gpu` - encapsulates all the 18 runners
* `igpu` - encapsulates all the 18 runners as all of them have iGPU
* `dgpu` - encapsulates 12 runners that have dGPU

Here is an example, a `GPU Tests` job that uses the `gpu` label and runs on any available GPU runner:
```yaml
  GPU:
    name: GPU Tests
    needs: [ Build, Smart_CI ]
    runs-on: [ self-hosted, gpu ]
    container:
      image: ubuntu:20.04
      options: --device /dev/dri:/dev/dri --group-add 109 --group-add 44
      volumes:
        - /dev/dri:/dev/dri
  ...
```

If, for example, a job requires a dGPU, it should use the `dgpu` label, instead:
```yaml
  GPU:
    name: GPU Tests
    needs: [ Build, Smart_CI ]
    runs-on: [ self-hosted, dgpu ]
  ...
```

**NOTE**: as these are persistent runners, Docker should be used for the jobs that utilise the GPU runners.
Learn more about the
available images and how to choose one in the [OpenVINO Docker Image Overview](./docker_images.md).  

## How to Choose a Runner

The configuration of a runner required for a job (building, testing, other tasks) depends on the
nature of the job. Jobs that are more memory and/or CPU-intensive require a more robust configuration.

The `Build` job in the [`ubuntu_22.yml`](./../../../../.github/workflows/ubuntu_22.yml) workflow uses
the `aks-linux-16-cores-32gb` group as specified in the `runs-on` key:
```yaml
Build:
  ...
  runs-on: aks-linux-16-cores-32gb
  ...
```

The `aks-linux-16-cores-32gb` group has machines with 16-core CPU and 32 GB of RAM.
These resources are suitable for using in parallel by the build tools in the `Build` job.

The `C++ unit tests` job in the [`ubuntu_22.yml`](./../../../../.github/workflows/ubuntu_22.yml) workflow uses the `aks-linux-4-cores-16gb` group:
```yaml
CXX_Unit_Tests:
  name: C++ unit tests
  ...
  with:
    runner: aks-linux-4-cores-16gb
    ...
```

As the C++ tests can not use a large number of cores for parallel execution like
the build tools in the `Build` job, it is not beneficial to use the `aks-linux-16-cores-32gb` group for them.

Instead, it is advisable to use runners with more cores/RAM for tasks that can load them.

It is possible to experiment with different configurations before making a decision. You can
run a job on runners from different groups and compare the gains. If they are significant,
for example, 60 minutes on a 4-core runner compared to 15 minutes on a 16-core runner,
it is better to use the configuration with more cores.
