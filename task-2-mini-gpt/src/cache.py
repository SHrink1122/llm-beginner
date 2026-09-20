import hashlib
from pathlib import Path

import numpy as np
# 加速encode 不然每次调参都要encode，cpu跑得慢

def _fingerprint(tokenizer, text):
    h = hashlib.sha256()
    h.update(repr(tokenizer.merges).encode())
    h.update(b"\x00")                      
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def encode_cached(tokenizer, text, cache_path):

    cache_path = Path(cache_path)
    key = _fingerprint(tokenizer, text)

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                if str(data["key"].item()) == key:
                    ids = data["ids"].tolist()
                    print(f"缓存命中 {cache_path.name}")
                    return ids
        except Exception:
            pass

    print(f"缓存未命中，encode {len(text)} 字符中……")
    ids = tokenizer.encode(text)
    dtype = np.uint16 if tokenizer.vocab_size <= 65536 else np.uint32
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, ids=np.asarray(ids, dtype=dtype), key=np.array(key))
    print(f"缓存已写入 {cache_path}：{len(ids)} token")
    return ids
