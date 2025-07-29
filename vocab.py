class SentencePieceVocabulary(object):
    def __init__(self, sp_model_path: str):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)

        # Set padding token.
        self.padding_idx: int = self.sp.pad_id()
        self.pad: str = self.sp.id_to_piece(self.padding_idx)

        # Set unk token.
        self.unk_id: int = self.sp.unk_id()
        self.unktoken: str = self.sp.id_to_piece(self.unk_id)

        # Set eos token.
        self.eos_idx: int = self.sp.eos_id()
        self.eos: str = self.sp.id_to_piece(self.eos_idx)

    def id_to_word(self, i: int) -> str:
        return self.sp.id_to_piece(i)

    def size(self) -> int:
        return self.sp.get_piece_size()

    def get_id(self, w) -> int:
        return self.sp.piece_to_id(w)
