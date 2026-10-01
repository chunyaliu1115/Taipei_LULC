# Publishing to GitHub

Run everything below in Terminal on the Mac that holds this folder. Nothing
here touches Earth Engine.

Commands are shown indented. Copy only the indented lines — if you select a
whole block including its surrounding text you will paste something the shell
cannot run.

## 1. Create the SSH key

Check first whether a key already exists, so you do not overwrite one that
other services use:

    ls -l ~/.ssh/id_ed25519.pub

If that file exists, skip to step 2 and use it. Otherwise:

    ssh-keygen -t ed25519 -C "mac-aman" -f ~/.ssh/id_ed25519

Press Return at the file prompt. At the passphrase prompt, type one — an
unprotected private key is a plain-text credential sitting on the disk. macOS
will remember it for you in step 2, so you type it once.

## 2. Load it into the agent and the Keychain

Start the agent:

    eval "$(ssh-agent -s)"

Append the host block to your SSH config. Paste all five lines together,
including the closing `EOF`; the shell will not return a prompt until it sees
that line.

    cat >> ~/.ssh/config <<'EOF'
    Host github.com
      HostName github.com
      User git
      IdentityFile ~/.ssh/id_ed25519
      AddKeysToAgent yes
      UseKeychain yes
    EOF

Then add the key:

    ssh-add --apple-use-keychain ~/.ssh/id_ed25519

`UseKeychain` is what stops the passphrase prompt returning after every
reboot.

Check the config took:

    cat ~/.ssh/config

## 3. Register the public key on GitHub

    pbcopy < ~/.ssh/id_ed25519.pub

The **public** key is now on the clipboard. Go to
<https://github.com/settings/keys> → *New SSH key*, title `mac-aman`, type
*Authentication Key*, paste, save.

Never paste `~/.ssh/id_ed25519` — the file without `.pub` is the private half
and must not leave the machine.

Verify:

    ssh -T git@github.com

The first connection asks you to confirm GitHub's host fingerprint. Compare
it against the published list at
<https://docs.github.com/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints>
before typing `yes`. Success looks like:
`Hi ami01! You've successfully authenticated, but GitHub does not provide shell access.`

## 4. Push

    cd ~/Taipei_LULC/Taipei_LULC
    git init -b main
    git add .
    git status

Read `git status` before committing. It should list around 124 files and no
`.venv`, no `archive/`, no `manuscript/`, and no `s2_composite_2020.tif`. If
any of those appear, stop — `.gitignore` has not been picked up.

    git config user.name  "Chun-ya Liu"
    git config user.email "chunyaliu@hotmail.com"
    git commit -m "Code, data and figures for the Taipei six-classifier comparison"
    git remote add origin git@github.com:ami01/taipei-lulc.git
    git push -u origin main

If the repository was created with a README on GitHub, the first push will be
rejected as non-fast-forward. Either delete and recreate it empty, or run
`git pull --rebase origin main` first.

## 5. Mint the DOI

Sign in at <https://zenodo.org> with GitHub, go to *GitHub* in the account
menu, flip the switch next to `ami01/taipei-lulc`, then create a release on
GitHub (`v1.0.0`). Zenodo archives that tag and issues a DOI. Attach
`s2_composite_2020.tif` to the GitHub release so the archived version is
complete even though the file is not in the Git history.

Put the DOI in the manuscript's Data Availability Statement and in the
response to the editor before submitting.

## If the prompt changes to `bash-3.2$`

You pasted a line that started a nested shell — usually a stray triple-backtick
from a markdown block. Type `exit` and press Return; you are back in your
normal `%` prompt with nothing broken. Then check what actually ran:

    ls -l ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub
    cat ~/.ssh/config

If the key files exist and the config shows the `Host github.com` block once,
carry on from step 3. If the block appears twice, open `~/.ssh/config` in a
text editor and delete the duplicate.
