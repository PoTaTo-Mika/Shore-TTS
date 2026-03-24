from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Iterable


_DEFAULT_TEXT_SUFFIX = ".txt"
_DEFAULT_OUTPUT_PATH = "checkpoints/vocab/vocab.json"
_CUSTOM_TRANS = str.maketrans(
    {
        ";": ",",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)


def _require_pinyin_deps():
    try:
        import rjieba  # type: ignore
        from pypinyin import Style, lazy_pinyin  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PinyinTokenizer requires `rjieba` and `pypinyin`. "
            "Install them before using shore_tts/text/tokenizer.py."
        ) from exc

    return rjieba, Style, lazy_pinyin


def _is_chinese(char: str) -> bool:
    return "\u3100" <= char <= "\u9fff"


def _discover_tar_files(data_path: str | Path) -> list[Path]:
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    tar_files = sorted(path for path in root.rglob("*.tar") if path.is_file())
    if not tar_files:
        raise FileNotFoundError(f"No *.tar files found under: {root}")
    return tar_files


def _iter_tar_texts(data_path: str | Path, text_suffix: str = _DEFAULT_TEXT_SUFFIX) -> Iterable[str]:
    for tar_path in _discover_tar_files(data_path):
        with tarfile.open(tar_path, mode="r:*") as archive:
            for member in archive:
                if not member.isreg():
                    continue
                if Path(member.name).suffix.lower() != text_suffix:
                    continue

                extracted = archive.extractfile(member)
                if extracted is None:
                    continue

                try:
                    yield extracted.read().decode("utf-8").strip()
                finally:
                    extracted.close()


class PinyinTokenizer:
    def __init__(self, token_to_id: dict[str, int] | None = None, polyphone: bool = True):
        self.polyphone = bool(polyphone)
        self.token_to_id = self._normalize_vocab(token_to_id or {" ": 0})

    @staticmethod
    def _normalize_vocab(token_to_id: dict[str, int]) -> dict[str, int]:
        if not token_to_id:
            return {" ": 0}

        if " " not in token_to_id:
            token_to_id = {" ": 0, **token_to_id}

        sorted_items = sorted(token_to_id.items(), key=lambda item: item[1])
        vocab = [token for token, _ in sorted_items if token != " "]
        normalized = {" ": 0}
        for token in vocab:
            if token not in normalized:
                normalized[token] = len(normalized)
        return normalized

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def unk_id(self) -> int:
        return 0

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def load(cls, path: str | Path, polyphone: bool = True) -> "PinyinTokenizer":
        with Path(path).open("r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        if not isinstance(token_to_id, dict):
            raise ValueError(f"Expected JSON dict vocab in {path}, but got {type(token_to_id).__name__}")
        return cls(token_to_id={str(k): int(v) for k, v in token_to_id.items()}, polyphone=polyphone)

    def text_to_pinyin(self, text: str) -> list[str]:
        rjieba, Style, lazy_pinyin = _require_pinyin_deps()

        tokens: list[str] = []
        text = text.translate(_CUSTOM_TRANS)

        for segment in rjieba.cut(text):
            segment_byte_len = len(segment.encode("utf-8"))

            if segment_byte_len == len(segment):
                if tokens and segment_byte_len > 1 and tokens[-1] not in " :'\"":
                    tokens.append(" ")
                tokens.extend(segment)
                continue

            if self.polyphone and segment_byte_len == 3 * len(segment):
                segment_pinyin = lazy_pinyin(segment, style=Style.TONE3, tone_sandhi=True)
                for idx, char in enumerate(segment):
                    if _is_chinese(char):
                        tokens.append(" ")
                    tokens.append(segment_pinyin[idx])
                continue

            for char in segment:
                if ord(char) < 256:
                    tokens.extend(char)
                elif _is_chinese(char):
                    tokens.append(" ")
                    tokens.extend(lazy_pinyin(char, style=Style.TONE3, tone_sandhi=True))
                else:
                    tokens.append(char)

        return tokens

    def pinyin_to_ids(self, pinyin_tokens: list[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in pinyin_tokens]

    def encode(self, text: str) -> list[int]:
        return self.pinyin_to_ids(self.text_to_pinyin(text))

    @classmethod
    def train_from_tar_dir(
        cls,
        data_path: str | Path,
        output_path: str | Path | None = None,
        polyphone: bool = True,
        text_suffix: str = _DEFAULT_TEXT_SUFFIX,
    ) -> "PinyinTokenizer":
        tokenizer = cls(polyphone=polyphone)
        vocab_tokens: set[str] = set()

        for text in _iter_tar_texts(data_path, text_suffix=text_suffix):
            vocab_tokens.update(tokenizer.text_to_pinyin(text))

        ordered_tokens = [" "]
        ordered_tokens.extend(sorted(token for token in vocab_tokens if token != " "))
        tokenizer.token_to_id = {token: idx for idx, token in enumerate(ordered_tokens)}

        if output_path is not None:
            tokenizer.save(output_path)

        return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Shore-TTS pinyin vocabulary by streaming .txt entries from tar shards."
    )
    parser.add_argument("--data-path", required=True, help="Root directory containing *.tar shards.")
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT_PATH,
        help=f"Output JSON vocab path. Default: {_DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--text-suffix",
        default=_DEFAULT_TEXT_SUFFIX,
        help="Text file suffix inside tar shards. Default: .txt",
    )
    parser.add_argument(
        "--disable-polyphone",
        action="store_true",
        help="Disable polyphone-aware pinyin conversion.",
    )
    args = parser.parse_args()

    tokenizer = PinyinTokenizer.train_from_tar_dir(
        data_path=args.data_path,
        output_path=args.output,
        polyphone=not args.disable_polyphone,
        text_suffix=args.text_suffix,
    )

    tar_count = len(_discover_tar_files(args.data_path))
    print(f"[tokenizer] tar_files={tar_count}")
    print(f"[tokenizer] vocab_size={tokenizer.vocab_size}")
    print(f"[tokenizer] saved_to={Path(args.output)}")


if __name__ == "__main__":
    main()
