import os

DEFAULT_TARGET_LANG = "zh-TW"


def subtitle_suffix(lang: str) -> str:
    """The `.<lang>.srt` ending that marks a subtitle's language."""
    return f".{lang}.srt"


def sidecar_subtitle_path(video_path: str, lang: str = DEFAULT_TARGET_LANG) -> str:
    """`Video.<lang>.srt` next to the video: the name Jellyfin and Plex pick up."""
    return f"{os.path.splitext(video_path)[0]}{subtitle_suffix(lang)}"


def output_subtitle_path(source_sub_path: str, lang: str = DEFAULT_TARGET_LANG) -> str:
    """Where the translation of `source_sub_path` is written, and where an
    existing translation is looked for before translating again."""
    if source_sub_path.lower().endswith(".en.srt"):
        return f"{source_sub_path[:-len('.en.srt')]}{subtitle_suffix(lang)}"
    if source_sub_path.endswith(".srt"):
        return f"{source_sub_path[:-len('.srt')]}(AI){subtitle_suffix(lang)}"
    return f"{source_sub_path}{subtitle_suffix(lang)}"
