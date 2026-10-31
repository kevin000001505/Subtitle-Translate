import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def local_modules():
    source = open(os.path.join(ROOT, "main.py")).read()
    imported = set(re.findall(r"^(?:from|import) (\w+)", source, re.M))
    return {m for m in imported if os.path.exists(os.path.join(ROOT, f"{m}.py"))} | {"main"}


def test_the_image_copies_every_backend_module():
    copy_line = next(l for l in open(os.path.join(ROOT, "Dockerfile")) if l.startswith("COPY"))
    assert local_modules() <= set(re.findall(r"(\w+)\.py", copy_line))


def test_the_dev_override_mounts_every_backend_module():
    override = open(os.path.join(ROOT, "docker-compose.override.yml")).read()
    assert local_modules() <= set(re.findall(r"\./(\w+)\.py:", override))


def compose_section(name):
    compose = open(os.path.join(ROOT, "docker-compose.yml")).read()
    m = re.search(rf"^  {name}:\n((?:^    .*$\n?)+)", compose, re.M)
    return m.group(1) if m else ""


def test_the_compose_wires_the_webroot_check_and_restart_policies():
    frontend = compose_section("frontend")
    backend = compose_section("backend")
    assert "restart: unless-stopped" in backend
    assert "restart: unless-stopped" in frontend

    # The check script is mounted read-only at a path outside the web root,
    # so it is not publicly served.
    mount = re.search(r"^\s*- \./frontend/check-webroot\.sh:([\w./-]+):ro\s*$", frontend, re.M)
    assert mount and not mount.group(1).startswith("/usr/share/nginx/html")

    # The command runs the check first, then starts nginx in the foreground
    # through the official entrypoint, so its configuration hooks still run.
    command = re.search(r"^\s*command:\s*(.+)$", frontend, re.M).group(1)
    assert command.index("check-webroot.sh") < command.index("&&") < command.index("nginx")
    assert "docker-entrypoint.sh" in command
