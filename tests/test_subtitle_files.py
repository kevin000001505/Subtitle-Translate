import pytest

from subtitle_files import output_subtitle_path, sidecar_subtitle_path


@pytest.mark.parametrize("lang", ["zh-TW", "ja"])
def test_the_translation_of_an_english_srt_sits_next_to_the_video(lang):
    assert output_subtitle_path("/media/M/Movie.en.srt", lang) == f"/media/M/Movie.{lang}.srt"


@pytest.mark.parametrize("lang", ["zh-TW", "ja"])
def test_the_translation_of_an_untagged_srt_is_marked_as_ai(lang):
    assert output_subtitle_path("/media/M/Movie.srt", lang) == f"/media/M/Movie(AI).{lang}.srt"


def test_the_translation_of_a_non_srt_subtitle_keeps_its_full_name():
    assert output_subtitle_path("/media/M/Movie.ass", "ja") == "/media/M/Movie.ass.ja.srt"


@pytest.mark.parametrize("lang", ["zh-TW", "ja"])
def test_a_videos_subtitle_is_named_after_the_video(lang):
    assert sidecar_subtitle_path("/media/S/S01E01.mkv", lang) == f"/media/S/S01E01.{lang}.srt"
