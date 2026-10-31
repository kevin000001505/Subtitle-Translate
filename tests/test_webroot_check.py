import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECK_SCRIPT = os.path.join(ROOT, "frontend", "check-webroot.sh")


def run_check(directory):
    return subprocess.run(["sh", CHECK_SCRIPT, str(directory)],
                         capture_output=True, text=True)


def test_the_webroot_check_fails_when_index_html_is_missing(tmp_path):
    result = run_check(tmp_path)
    assert result.returncode != 0
    assert "index.html" in result.stderr
    assert str(tmp_path) in result.stderr


def test_the_webroot_check_passes_when_index_html_is_present(tmp_path):
    (tmp_path / "index.html").write_text("<html></html>")
    assert run_check(tmp_path).returncode == 0
