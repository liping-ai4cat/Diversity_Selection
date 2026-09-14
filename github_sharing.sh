#!/usr/bin/env bash
#
# github_sharing.sh -- publish the divsel package to GitHub, in one command.
#
# QUICK START (first time)
#   1. Create an EMPTY repository on github.com -- no README, no .gitignore, no
#      licence, or the first push is rejected:
#         https://github.com/new   ->  owner liping-ai4cat, name Diversity_Selection
#   2. ./github_sharing.sh push
#      One command does the lot: git init if needed, sets the commit e-mail,
#      puts the branch on 'main', adds origin, runs the preflight, commits
#      everything and pushes.
#
# EVERY TIME AFTER THAT
#   ./github_sharing.sh push -m "what changed"
#
# USAGE
#   ./github_sharing.sh push [-m MSG] [-y] [url]  # setup + check + commit + push
#   ./github_sharing.sh setup [url]               # all of that except publishing
#   ./github_sharing.sh check                     # preflight only, writes nothing
#   ./github_sharing.sh commit [-m MSG]           # check + commit, no push
#   ./github_sharing.sh remote [url]              # set or replace origin
#   ./github_sharing.sh tag v0.1.0                # annotated tag for a release
#   ./github_sharing.sh status
#
#   -y  skip the push confirmation (needed when stdin is not a terminal)
#
# WHAT IS NOT PUSHED
#   orginal_code_v1/ and orginal_code_v2/ -- the pre-package original scripts --
#   stay on this machine. They are listed in .gitignore, and `check` FAILS
#   outright if anything under them ever reaches the index, so a stray
#   `git add -f` cannot publish them by accident. Everything else .gitignore
#   covers (caches, .npy/.traj/.png outputs, demo runs) also stays local.
#
# AUTHENTICATION (https remote)
#   GitHub does not accept account passwords. Use a Personal Access Token:
#     github.com -> Settings -> Developer settings -> Personal access tokens
#     -> Fine-grained tokens -> repository Diversity_Selection, Contents: Read and write
#   At the push prompt:  Username = liping-ai4cat,  Password = that token.
#   `setup` enables git's in-memory credential cache (8 h), so a day's pushes
#   ask once. The token is never written to disk by this script -- /projects is
#   a group-readable shared filesystem, so keep it out of files.
#
#   Or use SSH and never be prompted again:
#     ssh-keygen -t ed25519 -C lipingliu@uchicago.edu
#     cat ~/.ssh/id_ed25519.pub     # paste into github.com -> Settings -> SSH keys
#     ./github_sharing.sh remote git@github.com:liping-ai4cat/Diversity_Selection.git
#
# SAFETY
#   * `check` runs before every commit and every push; a failure blocks both.
#   * `push` asks to confirm -- typed confirmation the first time, y/N after.
#   * Force-pushing is refused outright; there is no flag to enable it.
#   * Publishing is not undoable: anything pushed to a public repository can be
#     cached, forked or indexed by third parties even if you delete it later.
#
# TAGGED RELEASE / CODE DOI (optional, after the first push)
#   ./github_sharing.sh tag v0.1.0 && git push origin v0.1.0
#   Then draft a GitHub release from that tag. To mint a DOI, switch the
#   repository ON at zenodo.org -> Settings -> GitHub *before* the release;
#   Zenodo only archives releases created after the switch is on.
#
set -euo pipefail

SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SELF_DIR"

DEFAULT_REMOTE="https://github.com/liping-ai4cat/Diversity_Selection"  # matches pyproject [project.urls]
COMMIT_EMAIL="lipingliu@uchicago.edu"                         # baked into every commit, publicly
BRANCH="main"

# Directories that must never leave this machine.
NEVER_PUSH=(orginal_code_v1 orginal_code_v2)

MAX_FILE_BYTES=$((100*1024*1024))   # GitHub hard limit, per file
WARN_FILE_BYTES=$((50*1024*1024))   # GitHub starts warning here
WARN_REPO_BYTES=$((1024*1024*1024)) # GitHub recommends staying under ~1 GB

say()  { printf '%s\n' "$*" >&2; }
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$*" >&2; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*" >&2; WARNS=$((WARNS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*" >&2; FAILS=$((FAILS+1)); }
rule() { printf '%s\n' "------------------------------------------------------------" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
human(){ numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 bytes"; }
usage(){ sed -n '2,60p' "$0" >&2; }

# Files git would put in a commit: the index plus everything .gitignore leaves.
tracked_list() {
  if [[ -d .git ]]; then
    git ls-files -z --cached --others --exclude-standard
  else
    # Probe without writing anything into the work tree: the throwaway index
    # and object store live in a temp GIT_DIR that is deleted immediately.
    local tmp; tmp="$(mktemp -d)"
    GIT_DIR="$tmp/git" GIT_WORK_TREE="$PWD" git init -q >/dev/null 2>&1 || true
    GIT_DIR="$tmp/git" GIT_WORK_TREE="$PWD" git ls-files -z --others --exclude-standard
    rm -rf "$tmp"
  fi
}

# --------------------------------------------------------------------------
# setup: idempotent. Safe to run at any time; publishes nothing.
# --------------------------------------------------------------------------
ensure_repo() {
  if [[ ! -d .git ]]; then
    git init -q -b "$BRANCH"
    say "Initialised an empty repository on '$BRANCH'."
  fi
}

ensure_identity() {
  local local_email; local_email=$(git config --local --get user.email 2>/dev/null || echo "")
  if [[ -z "$local_email" ]]; then
    # Repo-local on purpose: the author e-mail is public and permanent in every
    # commit, and your global config is left untouched.
    git config user.email "$COMMIT_EMAIL"
    say "Commit e-mail for this repository: $COMMIT_EMAIL"
  elif [[ "$local_email" != "$COMMIT_EMAIL" ]]; then
    say "Note: repo-local user.email is $local_email, not $COMMIT_EMAIL. Leaving it."
    say "      To change it:  git config user.email '$COMMIT_EMAIL'"
  fi
  [[ -n "$(git config --get user.name 2>/dev/null || echo '')" ]] \
    || git config user.name "Liping Liu"
  # In-memory only, 8 h. Nothing is written to this shared filesystem.
  git config --local --get credential.helper >/dev/null 2>&1 \
    || git config credential.helper 'cache --timeout=28800'
}

ensure_branch() {
  local cur; cur=$(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo "")
  [[ "$cur" == "$BRANCH" ]] && return 0
  if ! git rev-parse --verify -q HEAD >/dev/null 2>&1; then
    git symbolic-ref HEAD "refs/heads/$BRANCH"        # no commits yet: just repoint
    say "Branch set to '$BRANCH' (was '${cur:-none}')."
  elif git show-ref -q --verify "refs/heads/$BRANCH" 2>/dev/null; then
    git switch -q "$BRANCH"
    say "Switched to existing branch '$BRANCH'."
  else
    git branch -m "$BRANCH"                            # rename, keeping history
    say "Renamed branch '$cur' -> '$BRANCH'."
  fi
}

ensure_ignored() {
  [[ -f .gitignore ]] || : > .gitignore
  local d added=0
  for d in "${NEVER_PUSH[@]}"; do
    grep -qxF "$d/" .gitignore && continue
    (( added )) || printf '\n# kept local, never published (pre-package originals)\n' >> .gitignore
    printf '%s/\n' "$d" >> .gitignore
    added=1
    say "Added '$d/' to .gitignore."
  done
  # Belt and braces: if they were committed before being ignored, ignoring is
  # not enough -- git keeps tracking them. Untrack, keeping the files on disk.
  if [[ -d .git ]]; then
    for d in "${NEVER_PUSH[@]}"; do
      if git ls-files --error-unmatch "$d" >/dev/null 2>&1; then
        git rm -r -q --cached "$d"
        say "Untracked '$d/' (files left on disk)."
      fi
    done
  fi
}

cmd_setup() {
  ensure_repo
  ensure_identity
  ensure_branch
  ensure_ignored
  if [[ -n "${1:-}" ]]; then
    cmd_remote "$1"
  elif ! git remote get-url origin >/dev/null 2>&1; then
    cmd_remote "$DEFAULT_REMOTE"
  else
    say "origin -> $(git remote get-url origin)"
  fi
  say "Setup complete. Preflight:"
  cmd_check
}

# --------------------------------------------------------------------------
cmd_check() {
  FAILS=0; WARNS=0
  say "Preflight for: $SELF_DIR"
  rule

  [[ -f .gitignore ]] && ok ".gitignore present" || bad ".gitignore MISSING -- run '$0 setup'"
  [[ -f README.md  ]] && ok "README.md present"  || bad "README.md missing"
  [[ -f LICENSE    ]] && ok "LICENSE present"    || warn "LICENSE missing"

  # Collect exactly what a commit would contain, once, and reuse it below.
  local -a files=(); local f
  while IFS= read -r -d '' f; do files+=("$f"); done < <(tracked_list)

  # The one hard rule: the original-code directories must not be publishable.
  local -a leaked=() d
  for f in "${files[@]}"; do
    for d in "${NEVER_PUSH[@]}"; do
      [[ "$f" == "$d/"* ]] && leaked+=("$f")
    done
  done
  if (( ${#leaked[@]} )); then
    bad "${#leaked[@]} file(s) from ${NEVER_PUSH[*]} would be published:"
    printf '           %s\n' "${leaked[@]}" >&2
    say  "           Fix with:  $0 setup"
  else
    ok "${NEVER_PUSH[*]} excluded from the push"
  fi

  local n=0 total=0 big=() warnbig=()
  for f in "${files[@]}"; do
    [[ -f "$f" ]] || continue
    local sz; sz=$(stat -c%s "$f")
    n=$((n+1)); total=$((total+sz))
    (( sz > MAX_FILE_BYTES )) && big+=("$f ($(human "$sz"))")
    (( sz > WARN_FILE_BYTES && sz <= MAX_FILE_BYTES )) && warnbig+=("$f ($(human "$sz"))")
  done

  say "  ....  $n files, $(human "$total") would be committed"
  if (( ${#big[@]} )); then
    bad "${#big[@]} file(s) exceed GitHub's 100 MB hard limit -- the push WILL be rejected:"
    printf '           %s\n' "${big[@]}" >&2
  else
    ok "no file exceeds 100 MB"
  fi
  (( ${#warnbig[@]} )) && { warn "${#warnbig[@]} file(s) over 50 MB (GitHub will warn):"; printf '           %s\n' "${warnbig[@]}" >&2; } || ok "no file over 50 MB"
  (( total > WARN_REPO_BYTES )) && warn "repository over 1 GB ($(human "$total")) -- GitHub recommends staying below this" || ok "repository size is comfortable"

  # symlinks: git stores the link TEXT, not the target -- a clone elsewhere
  # gets a dangling link
  local nsym; nsym=$(find . -path ./.git -prune -o -type l -print | wc -l)
  (( nsym == 0 )) && ok "no symlinks (git would commit these as text, not data)" \
                  || { bad "$nsym symlink(s) found -- they would commit as dangling links:"; find . -path ./.git -prune -o -type l -print | sed 's/^/           /' >&2; }

  # secrets, scanned over the exact file list that would be pushed
  local hits
  hits=$(printf '%s\0' "${files[@]}" | xargs -0 --no-run-if-empty \
           grep -IlE 'ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}|BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|(ZENODO|GITHUB)_TOKEN=[A-Za-z0-9]' \
           --exclude=github_sharing.sh 2>/dev/null || true)
  [[ -z "$hits" ]] && ok "no API tokens or private keys" || { bad "possible secrets:"; printf '%s\n' "$hits" | sed 's/^/           /' >&2; }

  # e-mail addresses other than the ones you chose to publish
  hits=$(printf '%s\0' "${files[@]}" | xargs -0 --no-run-if-empty \
           grep -IoE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' 2>/dev/null \
         | grep -v '@users\.noreply\.github\.com' | grep -v 'git@github\.com' \
         | grep -vF "$COMMIT_EMAIL" \
         | cut -d: -f1 | sort -u || true)
  [[ -z "$hits" ]] && ok "no third-party e-mail addresses in the content" \
    || { warn "e-mail address(es) other than $COMMIT_EMAIL in:"; printf '%s\n' "$hits" | sed 's/^/           /' >&2; }

  # the commit author e-mail is public and permanent
  local ce; ce=$(git config --get user.email 2>/dev/null || echo "")
  if [[ -z "$ce" ]]; then
    bad "git user.email is not set -- run '$0 setup'"
  else
    ok "commits are authored as $ce (public and permanent in every commit)"
  fi

  # placeholders still to fill
  local ph; ph=$(grep -noE '<TODO>|XXXXXXX|/xxx|TBD' README.md pyproject.toml 2>/dev/null | head -8 || true)
  [[ -z "$ph" ]] && ok "no unfilled placeholders in README.md / pyproject.toml" \
    || { warn "unfilled placeholder(s):"; printf '%s\n' "$ph" | sed 's/^/           /' >&2; }

  rule
  if (( FAILS )); then say "$FAILS failure(s), $WARNS warning(s) -- fix the failures before pushing."; return 1; fi
  say "0 failures, $WARNS warning(s). Safe to commit and push."
}

# --------------------------------------------------------------------------
first_commit_message() {
  cat <<'MSG'
divsel: diversity selection of atomic structures

SOAP descriptors with k-means or farthest-point sampling, streaming for
datasets that do not fit in memory, and seed sets so active-learning rounds
never re-select structures you already hold. See README.md.
MSG
}

do_commit() {
  local msg="${1:-}"
  ensure_ignored
  git add -A
  if git diff --cached --quiet 2>/dev/null && git rev-parse --verify -q HEAD >/dev/null 2>&1; then
    say "Nothing new to commit."
    return 0
  fi
  if [[ -z "$msg" ]]; then
    if git rev-parse --verify -q HEAD >/dev/null 2>&1; then
      msg="Update divsel"
    else
      msg="$(first_commit_message)"
    fi
  fi
  git -c core.pager=cat commit -q -m "$msg"
  say "Committed: $(git log -1 --oneline)"
  say "Tracking $(git ls-files | wc -l) files."
}

cmd_commit() {
  ensure_repo; ensure_identity; ensure_branch
  cmd_check || die "Preflight failed; nothing committed."
  do_commit "${1:-}"
}

cmd_remote() {
  ensure_repo
  local url="${1:-$DEFAULT_REMOTE}"
  if git remote get-url origin >/dev/null 2>&1; then
    local old; old=$(git remote get-url origin)
    if [[ "$old" != "$url" ]]; then
      git remote set-url origin "$url"
      say "origin: $old -> $url"
    else
      say "origin -> $url"
    fi
  else
    git remote add origin "$url"
    say "origin -> $url"
  fi
}

cmd_push() {
  local msg="${1:-}" yes="${2:-0}" url="${3:-}"

  ensure_repo; ensure_identity; ensure_branch; ensure_ignored
  if [[ -n "$url" ]]; then
    cmd_remote "$url"
  elif ! git remote get-url origin >/dev/null 2>&1; then
    cmd_remote "$DEFAULT_REMOTE"
  fi

  cmd_check || die "Preflight failed; refusing to push."
  do_commit "$msg"
  git rev-parse --verify -q HEAD >/dev/null 2>&1 || die "No commits to push."

  local remote_url n sz first=0
  remote_url=$(git remote get-url origin)
  n=$(git ls-files | wc -l)
  sz=$(git ls-files -z | xargs -0 --no-run-if-empty stat -c%s 2>/dev/null | awk '{s+=$1} END {print s+0}')
  git rev-parse --verify -q "refs/remotes/origin/$BRANCH" >/dev/null 2>&1 || first=1

  rule
  say "About to PUSH"
  say "  remote: $remote_url"
  say "  branch: $BRANCH"
  say "  files:  $n  ($(human "$sz"))"
  say "  commit: $(git log -1 --oneline)"
  say "  local:  ${NEVER_PUSH[*]} (not pushed)"
  rule

  if (( yes )); then
    say "Confirmation skipped (-y)."
  elif (( first )); then
    say "This publishes for the first time. Anything pushed to a public"
    say "repository may be cached, forked or indexed by third parties even if"
    say "you delete it afterwards."
    say ""
    [[ -t 0 ]] || die "Not a terminal; re-run with -y if you are sure."
    say "Type exactly:  PUSH TO GITHUB"
    local reply; read -r reply
    [[ "$reply" == "PUSH TO GITHUB" ]] || die "Not confirmed; nothing pushed."
  else
    [[ -t 0 ]] || die "Not a terminal; re-run with -y if you are sure."
    printf 'Push to %s ? [y/N] ' "$remote_url" >&2
    local reply; read -r reply
    [[ "$reply" == [yY]* ]] || die "Not confirmed; nothing pushed."
  fi

  # never --force, by design
  if ! git push -u origin "$BRANCH"; then
    rule
    say "Push failed. The usual causes:"
    say "  * repository does not exist yet -- create it EMPTY at https://github.com/new"
    say "  * authentication -- use a Personal Access Token as the password,"
    say "    not your GitHub account password (see AUTHENTICATION at the top)."
    say "  * remote has commits you do not have (e.g. GitHub added a README):"
    say "      git pull --rebase origin $BRANCH   then re-run '$0 push'"
    exit 1
  fi
  rule
  say "Pushed to $remote_url ($BRANCH)."
  say "Next push:  $0 push -m 'what changed'"
}

cmd_tag() {
  [[ -d .git ]] || die "Not a git repository. Run: $0 setup"
  local t="${1:?usage: $0 tag v0.1.0}"
  git tag -a "$t" -m "divsel $t"
  say "Created tag $t. Publish it with:  git push origin $t"
  say "Then draft a GitHub release from that tag."
}

cmd_status() {
  [[ -d .git ]] || { say "Not a git repository yet. Run: $0 setup"; return 0; }
  say "branch:  $(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo '(detached)')"
  say "origin:  $(git remote get-url origin 2>/dev/null || echo '(none)')"
  say "e-mail:  $(git config --get user.email 2>/dev/null || echo '(unset)')"
  say "tracked: $(git ls-files | wc -l) files"
  say "commits: $(git rev-list --count HEAD 2>/dev/null || echo 0)"
  say "local:   ${NEVER_PUSH[*]} (never pushed)"
  git -c core.pager=cat status --short | head -20 >&2
}

# --------------------------------------------------------------------------
sub="${1:-check}"; shift || true
MSG=""; YES=0; POS=()
while (( $# )); do
  case "$1" in
    -m|--message) MSG="${2:?-m needs a message}"; shift 2 ;;
    -y|--yes)     YES=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    -f|--force)   die "Force-pushing is refused by design." ;;
    -*)           die "unknown option: $1" ;;
    *)            POS+=("$1"); shift ;;
  esac
done

case "$sub" in
  check)  cmd_check ;;
  setup)  cmd_setup "${POS[0]:-}" ;;
  init)   cmd_setup "${POS[0]:-}" ;;            # old name, kept working
  commit) cmd_commit "$MSG" ;;
  remote) cmd_remote "${POS[0]:-}" ;;
  push)   cmd_push "$MSG" "$YES" "${POS[0]:-}" ;;
  tag)    cmd_tag "${POS[0]:-}" ;;
  status) cmd_status ;;
  -h|--help|help) usage ;;
  *) say "unknown command: $sub"; usage; exit 1 ;;
esac
