"""文本编码：mobileclip2_b.ts + YOLOE get_tpe -> W [N,512]（容器内运行）"""
import os, json, gzip, html, functools, numpy as np, torch, torch.nn.functional as F
from functools import lru_cache
from ftfy import fix_text
import regex as re
_TS = os.environ.get("MOBILECLIP_TS", "/cix/mobileclip2_b.ts")
_PT = os.environ.get("YOLOE_PT", "/cix/yoloe/yoloe-26l-seg.pt")

@functools.lru_cache(maxsize=1)
def _assets():
    import torch
    enc = torch.jit.load(_TS, map_location="cpu").eval()
    ck = torch.load(_PT, map_location="cpu", weights_only=False)
    net = (ck["model"] if isinstance(ck, dict) and "model" in ck else ck).float().eval()
    return enc, net.model[-1]

def encode(texts):
    """texts: list[str] -> np.ndarray [N,512] (L2 normalized)"""
    if isinstance(texts, str):
        texts = [texts]
    tk = SimpleTokenizer(default_bpe())
    enc, head = _assets()
    ids = torch.tensor([tk.encode(t) for t in texts])
    with torch.no_grad():
        feats = enc(ids).float()
        tpe = head.get_tpe(feats.unsqueeze(0))
    W = torch.nn.functional.normalize(tpe[0].float(), dim=-1).numpy()
    return W

"""词表工具：mobileclip2_b.ts + YOLOE 的 reprta/savpe 头 → W[C,512]\n\n用法: PYTHONUSERBASE=.pyuser /usr/bin/python3.10 make_W.py "person" "car" ...\n产物: W.npy + W_classes.json"""
from functools import lru_cache
from ftfy import fix_text
import regex as re

# ---------- OpenAI CLIP 的 BPE 分词器（本地 vocab/merges） ----------
def bytes_to_unicode():
    bs=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1))
    cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return dict(zip(bs,[chr(c) for c in cs]))

@lru_cache()
def default_bpe():
    return "/cix/clipL"

def get_pairs(word):
    return {(word[i],word[i+1]) for i in range(len(word)-1)}

class SimpleTokenizer:
    def __init__(self, bpe_path):
        self.byte_encoder=bytes_to_unicode()
        merges=open(f"{bpe_path}/merges.txt",encoding="utf-8").read().split("\n")[1:-1]
        merges=[tuple(m.split()) for m in merges]
        self.bpe_ranks=dict(zip(merges,range(len(merges))))
        self.cache={}
        vocab=json.load(open(f"{bpe_path}/vocab.json",encoding="utf-8"))
        self.encoder={t:i for t,i in vocab.items()}
        self.pat=re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",re.IGNORECASE)
    def bpe(self, token):
        if token in self.cache: return self.cache[token]
        word=tuple(token[:-1])+(token[-1]+"</w>",)
        pairs=get_pairs(word)
        if not pairs: return token+"</w>"
        while True:
            bigram=min(pairs,key=lambda p:self.bpe_ranks.get(p,float("inf")))
            if bigram not in self.bpe_ranks: break
            a,b=bigram; new=[]
            i=0
            while i<len(word):
                try: j=word.index(a,i)
                except ValueError: new.extend(word[i:]); break
                new.extend(word[i:j]); i=j
                if i<len(word)-1 and word[i]==a and word[i+1]==b: new.append(a+b); i+=2
                else: new.append(word[i]); i+=1
            word=tuple(new)
            if len(word)==1: break
            pairs=get_pairs(word)
        out=" ".join(word); self.cache[token]=out; return out
    def encode(self, text):
        bpe_tokens=[]
        for token in re.findall(self.pat, html.unescape(fix_text(text)).lower().strip()):
            tok="".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[t] for t in self.bpe(tok).split(" "))
        sot=self.encoder["<|startoftext|>"]; eot=self.encoder["<|endoftext|>"]
        ids=[sot]+bpe_tokens[:75]+[eot]
        return ids+[0]*(77-len(ids))

