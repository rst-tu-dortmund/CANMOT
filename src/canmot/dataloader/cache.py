import multiprocessing
import os.path
from pickle import UnpicklingError


class CacheLoader:
    def __init__(self, loader, cache_path):
        self.loader = loader
        self.path = cache_path
        self._sub_cache_path = None
        os.makedirs(self.path, exist_ok=True)
        self.cache = {}
        self._hash_checked = False

    def __getitem__(self, item):
        if item > len(self.loader) - 1:
            raise IndexError(
                f"Index {item} out of range for loader with length {len(self.loader)}"
            )
        if item not in self.cache or not self._hash_checked:
            if os.path.isfile(f"{self.sub_cache_path}/{item}.pkl"):
                with open(f"{self.sub_cache_path}/{item}.pkl", "rb") as f:
                    import pickle

                    try:
                        self.cache[item] = pickle.load(f)
                    except (UnpicklingError, EOFError):
                        self.cache[item] = self.loader[item]
                        self._hash_checked = False
                        with open(f"{self.sub_cache_path}/{item}.pkl", "wb") as f_w:
                            pickle.dump(self.cache[item], f_w)
            else:
                self.cache[item] = self.loader[item]
                with open(f"{self.sub_cache_path}/{item}.pkl", "wb") as f:
                    import pickle

                    pickle.dump(self.cache[item], f)
        return self.cache[item]

    @property
    def sub_cache_path(self):
        if self._sub_cache_path is None:
            h = self.hash()
            self._sub_cache_path = f"{self.path}/{h}"
            os.makedirs(self._sub_cache_path, exist_ok=True)
        return self._sub_cache_path

    def _build_cache_item(self, index, force=False):
        filepath = f"{self.sub_cache_path}/{index}.pkl"
        if force or (index not in self.cache and not os.path.exists(filepath)):
            item = self.loader[index]
            self.cache[index] = item
            with open(filepath, "wb") as f:
                import pickle

                pickle.dump(item, f)

    def hash(self):
        import hashlib

        hasher = hashlib.sha256()
        for item, value in self.loader.detector.items():
            hasher.update(str((item, value)).encode("utf-8"))
        return hasher.hexdigest()

    def build_cache(self, n_processes=1, force=False):
        if not force:
            # check hash
            if os.path.isfile(f"{self.sub_cache_path}/hash.txt"):
                with open(f"{self.sub_cache_path}/hash.txt", "r") as f:
                    cached_hash = f.read().strip()
                current_hash = self.hash()
                if cached_hash != current_hash:
                    force = True
                else:
                    self._hash_checked = True
            else:
                force = True
                self._hash_checked = True
        if n_processes > 1:
            with multiprocessing.Pool(n_processes) as pool:
                pool.starmap(
                    self._build_cache_item,
                    zip(range(len(self.loader)), [force] * len(self.loader)),
                )
        else:
            for i in range(len(self.loader)):
                self._build_cache_item(i, force=force)
        if force:
            with open(f"{self.sub_cache_path}/hash.txt", "w") as f:
                f.write(self.hash())

    def __getattr__(self, item):
        if item in {"path", "cache", "loader", "lock"}:
            return getattr(super, item)
        if hasattr(self.loader, item):
            return getattr(self.loader, item)
        raise AttributeError(f"{type(self).__name__} has no attribute '{item}'")

    def __len__(self):
        return len(self.loader)
