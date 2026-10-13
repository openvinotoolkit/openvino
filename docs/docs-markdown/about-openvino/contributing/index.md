---
sidebar_position: 4
sidebar_label: "Contribute to OpenVINO"
---

# Contribute to OpenVINO


OpenVINO™ is always looking for opportunities to improve and your contributions
play a big role in this process. Here are four ways you can make OpenVINO better:

- [Provide feedback](#provide-feedback)
- [Contribute code changes](#contribute-code-changes)
- [Improve documentation](#improve-documentation)
- [Promote and support OpenVINO](#promote-and-support-openvino)

##  Provide feedback

Report bugs / issues

If you notice unexpected behavior in OpenVINO or its components, you can
[create a new issue](https://github.com/openvinotoolkit/openvino/issues)
in the GitHub issue tracker.

Propose improvements

If you want to share your ideas for improving OpenVINO:

- Open a new [GitHub Discussion](https://github.com/openvinotoolkit/openvino/discussions).
- Create a [Feature Request Issue](https://github.com/openvinotoolkit/openvino/issues/new?assignees=octocat&labels=enhancement%2Cfeature&projects=&template=feature_request.yml&title=%5BFeature+Request%5D%3A+)
  if your idea is already well defined.

In both cases, provide a detailed description and list potential use cases,
benefits, and challenges. Keep in mind that even if your input is not immediately
prioritized, it may be used at a later or undertaken by the community.

##  Contribute code changes

Always check if the change is still needed! Verify if
[the issue](https://github.com/openvinotoolkit/openvino/issues) or
[request](https://github.com/openvinotoolkit/openvino/pulls) is still open
and nobody has started working on it. If the ticket is already work in progress,
you can always ask if you can help.

**Address only the issues that affect the master or**
[LTS release branches](../release-notes-openvino/release-policy.md).

**Do not start work on contributions, if a proper issue/ request has not been created.**

:::tip
If you want to start with something simple, check out
[first-time contributions](https://github.com/orgs/openvinotoolkit/projects/3).
:::

Fix bugs

Choose one of the issues reported in
[GitHub Issue Tracker](https://github.com/openvinotoolkit/openvino/issues) and
[create a Pull Request](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_PR.md)
(PR) addressing it.

If you find a new bug and want to fix it, you should still
create a new issue before working on the PR. This way, it will be easier for other
developers to track changes.

Develop new features

If you find a [Feature Request](https://github.com/openvinotoolkit/openvino/issues/new?assignees=octocat&labels=enhancement%2Cfeature&projects=&template=feature_request.yml&title=%5BFeature+Request%5D%3A+)
you want to work on, make sure it is clearly defined. If you have any doubts,
or the change is complex, [discuss it](https://github.com/openvinotoolkit/openvino/discussions)
with OpenVINO developers first.

If you have an idea for a new feature and want
to develop it, you should still create a Feature Request  before working on the
PR. This way, it will be easier for other developers to track changes.

Develop a new device plugin

If you want to run inference on a device that is currently not supported, you
can see how to develop a new plugin for it in the
[Plugin Developer Guide](https://docs.openvino.ai/2026/documentation/openvino-extensibility/openvino-plugin-library.html).

##  Improve documentation

OpenVINO user documentation is built from several sources, mainly the files in
the [docs/articles\_en](https://github.com/openvinotoolkit/openvino/tree/master/docs/articles_en)
folder, using [Sphinx](https://www.sphinx-doc.org/) and the
[reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)
markup language.

OpenVINO [developer documentation](https://github.com/openvinotoolkit/openvino/tree/master/docs/dev)
is available only in markdown in the [docs/dev](https://github.com/openvinotoolkit/openvino/tree/master/docs/dev)
folder.

To edit docs, consider using the Editor’s
[guide](https://github.com/openvinotoolkit/openvino/blob/master/CONTRIBUTING_DOCS.md)
and contacting [documentation maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-docs-maintainers),
who will help you with information architecture and formatting, as well as
review, adjust, and merge the PR.

Review user documentation

In most cases, creating a PR is enough to correct a documentation mistake, improve
the language, and update or extend the information. For your convenience, the
top-right panel of most pages includes the “Edit on GitHub” button that will
take you to the source file of the given article.

Write new content

For more extensive changes in docs, reach out to any of the
[documentation maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-docs-maintainers)
to discuss the new content.

##  Promote and support OpenVINO

Popularize OpenVINO

Articles, tutorials, blog posts, demos, videos, and any other involvement in the
OpenVINO community is more than welcome. If you discuss or present OpenVINO on
various social platforms, you are raising awareness of the product among AI
enthusiasts and enabling other people to discover the toolkit.

Feel free to reach out to OpenVINO developers if you need help with making a
contribution. You can also contact
[documentation maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-docs-maintainers)
, if you need help with visuals, brand materials, or content creation in general.

Help other community members

If you are an experienced OpenVINO user and want to help, you can share your
expertise with the community at any time. Check GitHub
[Discussions](https://github.com/openvinotoolkit/openvino/discussions)
and [Issues](https://github.com/openvinotoolkit/openvino/issues) to see if
you can help someone.

:::note
By contributing to the OpenVINO project, you agree that your contributions
will be licensed under [the terms of the OpenVINO repository](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE).
:::

## Additional Resources

- [Code Contribution Guide](code-contribution-guide.md)
- Choose a ["Good First Issue"](https://github.com/orgs/openvinotoolkit/projects/3).
- Learn more about [OpenVINO architecture](https://github.com/openvinotoolkit/openvino/blob/master/src/docs/architecture.md).
- Check out a [blog post on contributing to OpenVINO](https://medium.com/openvino-toolkit/how-to-contribute-to-an-ai-open-source-project-c741f48e009e).
- Visit [Intel DevHub Discord server](https://discord.gg/7pVRxUwdWG) to join
  discussions and talk to OpenVINO developers.
