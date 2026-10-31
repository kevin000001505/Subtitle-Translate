import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README_PATH = os.path.join(ROOT, "README.md")


def readme():
    assert os.path.exists(README_PATH), "README.md is missing"
    return open(README_PATH).read()


def compose_text():
    return open(os.path.join(ROOT, "docker-compose.yml")).read()


def compose_services():
    return re.findall(r"^  (\w+):$", compose_text(), re.M)


def compose_section(name):
    m = re.search(rf"^  {name}:\n((?:^    .*$\n?)+)", compose_text(), re.M)
    return m.group(1) if m else ""


def test_setup_gets_a_fresh_machine_to_a_serving_ui():
    text = readme()
    # Clone from the repo's own remote, not an invented URL.
    remote = subprocess.run(["git", "-C", ROOT, "remote", "get-url", "origin"],
                           capture_output=True, text=True).stdout.strip()
    url = re.sub(r"\.git$", "", remote.split("://")[-1] if "://" in remote
                 else remote.split(":")[-1])
    assert url in text
    # The .env step must reference a file that actually exists.
    assert os.path.exists(os.path.join(ROOT, ".env.example"))
    assert ".env.example" in text
    # The exact launch command the spec names.
    assert "docker compose up -d --build" in text
    # The UI address must use the port the compose file actually publishes.
    m = re.search(r'- "(\d+):80"', compose_section("frontend"))
    assert m, "frontend must publish a host port in the compose file"
    assert f":{m.group(1)}" in text


# Prose words, not configuration variables.
ENV_TOKEN_ALLOWLIST = {"API", "URL"}


def test_setup_invents_no_env_variables():
    declared = set(re.findall(r"^\s*([A-Z][A-Z0-9_]+)\s*=",
                             open(os.path.join(ROOT, ".env.example")).read(), re.M))
    used = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", readme()))
    assert used - declared - ENV_TOKEN_ALLOWLIST == set(), (
        f"README names variables that are not in .env.example: "
        f"{sorted(used - declared - ENV_TOKEN_ALLOWLIST)}")


def test_redeploy_is_an_in_place_pull_with_the_stale_bind_mount_explained():
    text = readme().lower()
    assert "git pull" in text
    assert "docker compose up -d" in text
    # The failure mode is explained in the project's own vocabulary.
    assert "stale bind mount" in open(os.path.join(ROOT, "GLOSSARY.md")).read().lower()
    assert "stale bind mount" in text
    # The mechanism: a fresh copy replaces the directory the running container
    # mounts, so it serves an empty web root and 403s.
    for keyword in ("replac", "web root", "403"):
        assert keyword in text, f"stale bind mount explanation must cover: {keyword}"


def test_recovery_and_known_limitation_are_documented():
    assert "frontend" in compose_services()
    text = readme()
    # Recovery from a fresh-copy deploy: recreate the frontend container
    # (compose service name from the compose file, not hardcoded).
    service = next(s for s in compose_services() if "front" in s)
    line = next((l for l in text.splitlines()
                if "recreate" in l.lower() and service in l.lower()), None)
    assert line, f"README must say to recreate the '{service}' container"
    # Known limitation: a directory replaced while the container runs is
    # caught at the next (re)start. Ground truth: the check runs once before
    # nginx starts (test_packaging) and the script starts nothing, so nothing
    # watches the web root while the container is up.
    assert re.search(r"next \(?re\)?start|every (re)?start|on start", text.lower())


def test_webroot_check_description_matches_the_script(tmp_path):
    script_path = os.path.join(ROOT, "frontend", "check-webroot.sh")
    script = open(script_path).read()
    checked = re.search(r'-f "\$webroot/(\S+)"', script)
    assert checked, "the check script must name the file it validates"
    # The README names the same file the script checks.
    assert checked.group(1) in readme()
    # Failure behavior, verified by running the script: non-zero exit, and a
    # message that names the missing file and its directory.
    result = subprocess.run(["sh", script_path, str(tmp_path)],
                           capture_output=True, text=True)
    assert result.returncode != 0
    assert checked.group(1) in result.stderr
    assert str(tmp_path) in result.stderr
    # The README describes that same behavior: the service exits and the
    # message names the missing file.
    low = readme().lower()
    assert "exit" in low
    assert "missing" in low
    # And that the check runs at start, before nginx comes up.
    assert re.search(r"before (nginx )?starts|at (every |each )?start|on start", low)


def test_restart_policy_description_matches_compose():
    values = {}
    for name in compose_services():
        m = re.search(r"^    restart: (\S+)$", compose_section(name), re.M)
        values[name] = m.group(1) if m else None
    assert values["frontend"] == values["backend"], \
        "restart policies must match across services"
    policy = values["frontend"]
    text = readme()
    # The README names the policy the compose file declares.
    assert policy in text
    low = text.lower()
    # ... and describes its two behaviors: start on host boot, and a service
    # the operator stopped stays stopped.
    assert "reboot" in low
    assert "stays stopped" in low
