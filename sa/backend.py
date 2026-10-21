"""Model backend (MLX): final hidden states H and the LM head W, nothing else.

SA only touches the last layer, so everything downstream needs just:
- hidden(ids): H [T, d], the final hidden states AFTER the model's last norm
- head(x):     logits [n, V] = x @ W^T, using the model's own (possibly 4-bit) head
"""
import mlx.core as mx
from mlx_lm import load


class Backend:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model, self.tokenizer = load(model_path)
        self.model.eval()
        args = self.model.args
        self.tied = bool(getattr(args, "tie_word_embeddings", False))
        self.d = args.hidden_size
        self.vocab_size = args.vocab_size

    def encode(self, text):
        """Token ids for raw text. Chat markers like <|im_start|> are parsed as special tokens."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def hidden(self, ids):
        """H [T, d] in float32: final hidden states after the last RMSNorm."""
        h = self.model.model(mx.array(ids)[None])[0]
        h = h.astype(mx.float32)
        mx.eval(h)
        return h

    def head(self, x):
        """Logits [n, V] for hidden vectors x [n, d], through the model's own LM head."""
        x = x.astype(self.head_dtype)
        if self.tied:
            out = self.model.model.embed_tokens.as_linear(x)
        else:
            out = self.model.lm_head(x)
        return out.astype(mx.float32)

    @property
    def head_dtype(self):
        layer = self.model.model.embed_tokens if self.tied else self.model.lm_head
        scales = getattr(layer, "scales", None)  # present on 4-bit layers
        return (scales if scales is not None else layer.weight).dtype

    def full_logits(self, ids):
        """Reference: the model's own forward pass (for the Step 2 check)."""
        out = self.model(mx.array(ids)[None])[0].astype(mx.float32)
        mx.eval(out)
        return out
