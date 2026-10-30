import os

DEFAULT_TARGET_LANG = "zh-TW"


def sidecar_subtitle_path(video_path: str, lang: str = DEFAULT_TARGET_LANG) -> str:
    """`Video.<lang>.srt` next to the video: the name Jellyfin and Plex pick up."""
    return f"{os.path.splitext(video_path)[0]}.{lang}.srt"


def output_subtitle_path(source_sub_path: str, lang: str = DEFAULT_TARGET_LANG) -> str:
    """Where the translation of `source_sub_path` is written, and where an
    existing translation is looked for before translating again."""
    if source_sub_path.lower().endswith(".en.srt"):
        return f"{source_sub_path[:-len('.en.srt')]}.{lang}.srt"
    if source_sub_path.endswith(".srt"):
        return f"{source_sub_path[:-len('.srt')]}(AI).{lang}.srt"
    return f"{source_sub_path}.{lang}.srt"
