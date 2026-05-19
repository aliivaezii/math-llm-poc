# Character-level tokenizer for arithmetic sequences; vocabulary: digits, operators, special tokens.
# Character-level is sufficient here — with only 16 tokens, subword tokenisation adds complexity
# without benefit for this fixed-domain task.

_VOCAB = ["<PAD>", "<BOS>", "<EOS>"] + list("0123456789+-=")


class Tokenizer:
    """Shared vocabulary and encode/decode helpers for arithmetic strings.

    All attributes are class-level so multiple instances share the same
    lookup tables without copying.
    """
    token_to_id: dict[str, int] = {tok: idx for idx, tok in enumerate(_VOCAB)}
    id_to_token: dict[int, str] = {idx: tok for idx, tok in enumerate(_VOCAB)}

    pad_id: int = token_to_id["<PAD>"]
    bos_id: int = token_to_id["<BOS>"]
    eos_id: int = token_to_id["<EOS>"]
    vocab_size: int = len(_VOCAB)

    _special_ids: frozenset[int] = frozenset({pad_id, bos_id, eos_id})

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text:
            if ch not in self.token_to_id:
                raise ValueError(f"Unknown character: {ch!r}")
            ids.append(self.token_to_id[ch])
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self.id_to_token[i] for i in ids if i not in self._special_ids
        )


if __name__ == "__main__":
    tok = Tokenizer()

    print(f"vocab_size : {tok.vocab_size}")
    print(f"pad_id={tok.pad_id}  bos_id={tok.bos_id}  eos_id={tok.eos_id}")
    print()

    text = "951+11=962"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    recovered = tok.decode(ids)

    print(f"encode({text!r}) → {ids}")
    print(f"decode(...)      → {recovered!r}")
    assert recovered == text, f"Round-trip failed: {recovered!r} != {text!r}"
    print("Round-trip OK")
