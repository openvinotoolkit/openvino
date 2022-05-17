# Contributing guidelines

If you have improvements to the oneDNN code, please send us your pull
requests! To get started, see the GitHub
[howto](https://help.github.com/en/articles/about-pull-requests).

You can:

- Submit your changes directly with a
  [pull request](https://github.com/oneapi-src/oneDNN/pulls)
- Log a bug or feedback with an [issue](https://github.com/oneapi-src/oneDNN/issues)

**See also:** [Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

## Pull request checklist

Before sending your pull requests, make sure that you have followed this list:

* Check the [library functionality guidelines](CONTRIBUTING.md#library-functionality-guidelines).
  If you are contributing a new compute primitive or propose changes to the
  external API, it is strongly advised to first open an [RFC pull request](CONTRIBUTING.md#RFC-pull-requests)
  with a detailed explanation of expected use cases and performance benefits.

* Ensure that the changes are consistent with the
  [code contribution guidelines](CONTRIBUTING.md#code-contribution-guidelines)
  and [coding standards](CONTRIBUTING.md#coding-standards).

* Check that [unit tests](CONTRIBUTING.md#unit-tests) pass.

## Library functionality guidelines

oneDNN focuses on functionality that satisfies all of the following
criteria:

1. *Performance*: the functionality has material impact on a workload level.
   In other words, this means that for a new primitive it should be
   demonstrated that it brings visible performance improvement to some
   workload.

2. *Generality*: the functionality is useful in a wide range of deep learning
   applications. This implies that when introducing a new primitive, its API
   needs to be general enough to be integrated into multiple deep learning
   frameworks that have similar functionality.

3. *Complexity*: it is not trivial to implement the functionality directly in
   a deep learning application.

### RFC pull requests

Significant library changes (new primitives, library architecture changes,
API modifications, etc) require approval from oneDNN maintainers before
opening a pull request with such implementation. For that we use the Request
For Comments (RFC) process, which consists of opening, discussing, and
accepting (promoting) RFC pull requests.

More information about the process can be found in the dedicated
[`rfcs`](https://github.com/oneapi-src/oneDNN/tree/rfcs) branch.

## Code contribution guidelines

When submitting your contribution, please make sure that it is:

* *Tested*: oneDNN uses gtests for lightweight functional testing and
  benchdnn for functionality that requires both performance and functional
  testing.

* *Documented*: oneDNN uses Doxygen for inline comments in public header
  files that is used to build reference manual and markdown (also processed by
  Doxygen) for user guide.

* *Portable*: oneDNN supports different operating systems, CPU and GPU
  architectures, compilers, and run-times. The new code should be compliant
  with the [System Requirements](README.md#system-requirements).

All code in oneDNN gets promoted to product branches (`master`, `rls-`, and
`mnt-`) only through GitHub pull requests. Requirements for promotion:

- The request is reviewed and approved by maintainers for all affected
  components.
- All discussions in the pull request are resolved.
- Continuous integration pipeline passed without errors.
- Promotion to release (`rls-`) branches can be done only by maintainers
  (enforced by GitHub)
- The pull request author is responsible for collecting all the necessary
  approvals, rebasing of the changes, and resolving the discussions.

To simplify the work of reviewers, make sure that the commits in the pull
request adhere to the following requirements:

- Commit message should be fit into 50 (at most 72) characters and have the
  imperative mood.
- Commit message should follow the format:
  `<scope>:[scope: ..] <short description>`
  Scope examples:
  * Top level: `build`, `api`, `doc`, `tests`, `common`, `cpu`, `gpu`
  * Second level: `convolution`, `pooling`, `utils`, `verbose`
  * Example commit message:
~~~git
common: verbose: fix crash when prim_iface_t is empty
~~~

- Commit body should also fit 72 characters. Think of it as a standard e-mail
  body or a markdown document in terms of styling - write sentences from the
  very left border keeping capital letters and punctuation in place.
- oneDNN branches maintain linear history. Rebase the changes on top of target
  branch before creating a pull request. Rebase again after resolving all the
  discussions, as well as in case of merge conflicts.
- Use `git add -p`  and `git rebase -i` liberally to split unrelated changes
  into multiple self-contained commits. This is a courtesy to reviewers: smaller
  commits are easier to comprehend. It also helps with bisecting in the future.
  Of course judgement needs to be applied whether to split changes or not. For
  example, split code cleanup and the actual fix into two separate patches.

## Coding Standards

Contributions to oneDNN must follow the [Coding Standards](CODING_STANDARDS.md)
in order to simplify development and review processes. The general principle is
to follow the style of existing/surrounding code.

The Coding Standards are subject to change and contributions to the Coding
Standards are welcome.

If you wish to propose changes to the Coding Standards (including `clang-tidy`
checks and `clang-format` options), please submit the proposal via an [RFC pull
request](CONTRIBUTING.md#RFC-pull-requests). The proposal should contain the
following information:
* *Motivation*: Why should the proposed standard be introduced and applied?
* *Enforcement*: Can the proposed standard be applied via an automated process
  or other practical means?
* *Example*: What does the code base look like with the proposed standard
  applied?
  * For instance, in case of a `clang-tidy` check, please open a separate PR
    with the check applied to the code base alongside the RFC PR.

## Unit tests

oneDNN uses gtests for lightweight functional testing and benchdnn for
performance and functional testing.

Be sure to extend the existing tests when fixing an issue.

Developing new benchdnn tests can be hard, so it is a good idea to start with
gtests first.
