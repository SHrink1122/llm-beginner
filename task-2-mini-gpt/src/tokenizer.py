import json
import time
from collections import defaultdict
from pathlib import Path


class BPETokenizer:
    def __init__(self):
        self.merges: list[tuple[int, int]] = []        
        self._id2bytes: list[bytes] = []               
        self._merge2rank: dict[tuple[int, int], int] = {}  
        self._rebuild_cache()

    @property
    def vocab_size(self):
        return 256 + len(self.merges)                  

    def train(self, text, vocab_size=1024):
        ids = list(text.encode("utf-8"))               # 原文转成 token id 序列
        num_merges = vocab_size - 256
        self.merges = []
        print(f"[BPE] 语料 {len(text)} 字符 / {len(ids)} 字节，vocab_size={vocab_size}（{num_merges} 次 merge）")

        report_every = max(1, num_merges // 10)        # 每 10% 报一次进度
        t0 = time.perf_counter()

        for step in range(num_merges):
            # 统计当前序列里相邻 token 对的频率（defaultdict 第一次出现即为 0）
            pairs: dict[tuple[int, int], int] = defaultdict(int)
            for i in range(len(ids) - 1):
                pairs[(ids[i], ids[i + 1])] += 1

            best_pair = max(pairs, key=pairs.get)
            best_count = pairs[best_pair]
            new_id = 256 + len(self.merges)
            self.merges.append(best_pair)

            # best_pair 换成新 token id
            merged = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged

            # 进度报告
            if (step + 1) % report_every == 0:
                print(f"[BPE] merge {step + 1}/{num_merges}"
                      f"  最高频对 {best_pair}×{best_count}"
                      f"  序列长度 {len(ids)}"
                      f"  已用 {time.perf_counter() - t0:.1f}s")

        self._rebuild_cache()
        print(f"[BPE] 训练结束：{len(self.merges)} 次 merge，序列长 {len(ids)}，"
              f"耗时 {time.perf_counter() - t0:.1f}s")

    def _rebuild_cache(self):
        self._id2bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self._id2bytes.append(self._id2bytes[a] + self._id2bytes[b])
        self._merge2rank = {pair: rank for rank, pair in enumerate(self.merges)}

    def encode(self, text):
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            # 找出当前序列里 rank 最小（优先级最高）的相邻对
            best_idx = -1
            best_rank = len(self.merges)
            for i in range(len(ids) - 1):
                rank = self._merge2rank.get((ids[i], ids[i + 1]))
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break

            merged_id = 256 + best_rank
            ids = ids[:best_idx] + [merged_id] + ids[best_idx + 2:]

        return ids

    def decode(self, ids):
        return b"".join(self._id2bytes[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        payload = {"vocab_size": self.vocab_size, "merges": self.merges}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.merges = [tuple(pair) for pair in data["merges"]]
        tok._rebuild_cache()
        return tok


_ROOT = Path(__file__).resolve().parent.parent


def train_tokenizer(corpus_path, out_path, vocab_size=347):
    text = Path(corpus_path).read_text(encoding="utf-8")

    tok = BPETokenizer()
    tok.train(text, vocab_size=vocab_size)
    tok.save(out_path)
    print(f"[BPE] 已保存 {out_path}（vocab_size={tok.vocab_size}）")

    # 算压缩率
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tok.encode(text))
    print(f"[BPE] 编码 train.txt：{n_tokens} tokens，压缩率 {n_bytes / n_tokens:.2f} 字节/token")
    return tok


if __name__ == "__main__":
    train_tokenizer(_ROOT / "data" / "train.txt", _ROOT / "ckpt" / "tokenizer.json")
