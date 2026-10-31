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
