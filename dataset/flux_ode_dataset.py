import os
import pickle
from pathlib import Path

import lmdb
import torch
from torch.utils.data import Dataset

from dataset.realesrgan import RealESRGAN_degradation


class LMDBDataset(Dataset):
    """Read one legacy LMDB or a directory of generated rank*.lmdb shards.

    LMDB handles are opened lazily in each DataLoader process. Initialization
    reads keys only, so it does not copy the image dataset into host memory.
    """

    def __init__(self, args):
        self.args = args
        self.lmdb_path = args.lmdb_path
        root = Path(args.lmdb_path)
        if root.is_file() or (root / "data.mdb").is_file():
            self.shard_paths = [root]
        else:
            self.shard_paths = sorted(p for p in root.glob("rank*.lmdb") if (p / "data.mdb").is_file())
        if not self.shard_paths:
            raise FileNotFoundError(f"No LMDB database or rank*.lmdb shards at {root}")

        self.records = []
        seen = set()
        for shard_idx, path in enumerate(self.shard_paths):
            env = self._open(path)
            try:
                with env.begin() as txn:
                    for key in txn.cursor().iternext(keys=True, values=False):
                        if key in seen:
                            raise ValueError(f"Duplicate sample key across LMDB shards: {key!r}")
                        seen.add(key)
                        self.records.append((shard_idx, key))
            finally:
                env.close()
        self.records.sort(key=lambda record: record[1])
        self.keys = [key for _, key in self.records]
        self.length = len(self.records)
        if not self.length:
            raise ValueError(f"LMDB dataset is empty: {root}")
        self._envs = {}
        self._pid = os.getpid()
        print(f"Total samples in LMDB: {self.length} ({len(self.shard_paths)} shards)")
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device="cpu")

    @staticmethod
    def _open(path):
        return lmdb.open(
            str(path), subdir=path.is_dir(), readonly=True, lock=False,
            readahead=False, meminit=False,
        )

    def _get_env(self, shard_idx):
        pid = os.getpid()
        if self._pid != pid:
            self.close()
            self._pid = pid
        if shard_idx not in self._envs:
            self._envs[shard_idx] = self._open(self.shard_paths[shard_idx])
        return self._envs[shard_idx]

    def close(self):
        for env in getattr(self, "_envs", {}).values():
            env.close()
        self._envs = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_envs"] = {}
        state["_pid"] = None
        return state

    def __del__(self):
        self.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range (0, {self.length})")
        shard_idx, key = self.records[idx]
        with self._get_env(shard_idx).begin() as txn:
            value = txn.get(key)
        if value is None:
            raise KeyError(f"Key {key!r} not found in {self.shard_paths[shard_idx]}")
        sample = pickle.loads(value)
        input_latents = torch.from_numpy(sample["input_latents"]).float()
        output_latents = torch.from_numpy(sample["output_latents"]).float()
        images = torch.from_numpy(sample["images"]).float()
        hr_images, lr_images = self.degradation.degrade_process(images.unsqueeze(0), resize_bak=True)
        return {
            "input_latents": input_latents,
            "output_latents": output_latents,
            "hr_images": hr_images.squeeze(0),
            "lr_images": lr_images.squeeze(0),
            "prompt_idx": sample.get("prompt_idx", idx),
            "seed": sample.get("seed", -1),
        }
