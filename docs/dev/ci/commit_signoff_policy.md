# How to sign-off commits

We require a sign-off commit message in the following format on each commit in pull request.

```
This is a commit message.

Signed-off-by: Author Name <author-email@example.com>
```

## How to sign-off new commits

In a local Git environment, the sign-off message can be added to a commit either manually (as a text) 
or via the **-s** flag used with the “git commit” command, for example:

`git commit -s -m "My commit message"`

To avoid manually adding the flag for each commit, we recommend setting up a Git hook with the following steps:

1.	Navigate to the `<openvino repository root>/.git/hooks` folder.
2.	Open the `prepare-commit-msg.sample`  file and paste the following content:

```
#!/bin/sh

COMMIT_MSG_FILE=$1
COMMIT_SOURCE=$2
SHA1=$3

NAME=$(git config user.name)
EMAIL=$(git config user.email)

if [ -z "$NAME" ]; then
    echo "empty git config user.name"
    exit 1
fi

if [ -z "$EMAIL" ]; then
    echo "empty git config user.email"
    exit 1
fi

git interpret-trailers --if-exists doNothing --trailer \
    "Signed-off-by: $NAME <$EMAIL>" \
    --in-place "$1"
```

3.	Save the file with the name `prepare-commit-msg` (remove the .sample extension).
4.	Make the file executable (on Linux / Git Bash: `chmod +x <openvino repository root>/.git/hooks/prepare-commit-msg`).

**Note**: For both sign-off approaches, ensure your user name and email address are configured in Git first:

```
git config user.name 'FIRST_NAME LAST_NAME'
git config user.email 'MY_EMAIL@example.com'
```

### Sign-off web-based commits

To enable automatic sign-off of commits made via GitHub web interface, make sure that 
[Require contributors to sign off on web-based commits](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-the-commit-signoff-policy-for-your-repository#enabling-or-disabling-compulsory-commit-signoffs-for-your-repository) 
setting is selected in the Settings menu of your OpenVINO repository fork.

## How to sign-off older commits in the history

If you forget to add the sign-off to your last commit, you can amend it and force-push to GitHub:

```
git commit --amend --signoff
```

To sign off on even older commits, use an interactive rebase, edit unsigned commits, and execute 
`git commit --amend --signoff` for each. However, please note that if others have already started working based on 
the commits in this branch, this will rewrite history and may cause issues for collaborators.
