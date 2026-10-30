import os

DEFAULT_TARGET_LANG = "zh-TW"
SOURCE_LANG = "en"

# The target languages the picker offers, default first.
LANGUAGES = {
    "zh-TW": "Traditional Chinese (Taiwan)",
    "zh-CN": "Simplified Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "en": "English",
}


def language_name(code: str) -> str:
    return LANGUAGES.get(code, code)


def canonical_lang(code: str) -> str:
    """BCP-47 casing, so "zh-tw" and "ZH-TW" name the same file as "zh-TW":
    language lower case, 4-letter script title case, 2-letter region upper case."""
    language, *subtags = code.split("-")
    cased = [language.lower()]
    for tag in subtags:
        if len(tag) == 4 and tag.isalpha():
            cased.append(tag.title())
        elif len(tag) == 2 and tag.isalpha():
            cased.append(tag.upper())
        else:
            cased.append(tag.lower())
    return "-".join(cased)


def subtitle_suffix(lang: str) -> str:
    """The `.<lang>.srt` ending that marks a subtitle's language."""
    return f".{lang}.srt"


def is_source_subtitle(path: str) -> bool:
    """Whether `path` is the English `.en.srt` (any case) that gets translated."""
    return path.lower().endswith(subtitle_suffix(SOURCE_LANG))


def strip_source_suffix(path: str) -> str:
    """`Movie.en.srt` -> `Movie`, for a path is_source_subtitle accepts."""
    return path[:-len(subtitle_suffix(SOURCE_LANG))]


def sidecar_subtitle_path(video_path: str, lang: str = DEFAULT_TARGET_LANG) -> str:
    """`Video.<lang>.srt` next to the video: the name Jellyfin and Plex pick up."""
    return f"{os.path.splitext(video_path)[0]}{subtitle_suffix(lang)}"


def output_subtitle_path(source_sub_path: str, lang: str = DEFAULT_TARGET_LANG) -> str:
    """Where the translation of `source_sub_path` is written, and where an
    existing translation is looked for before translating again."""
    if is_source_subtitle(source_sub_path):
        return f"{strip_source_suffix(source_sub_path)}{subtitle_suffix(lang)}"
    if source_sub_path.endswith(".srt"):
        return f"{source_sub_path[:-len('.srt')]}(AI){subtitle_suffix(lang)}"
    return f"{source_sub_path}{subtitle_suffix(lang)}"
