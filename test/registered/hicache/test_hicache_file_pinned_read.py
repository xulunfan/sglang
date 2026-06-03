"""
Unit test for HiCacheFile.get() pinned memory fix (issue #26886).

Verifies that reading from file storage correctly populates the target tensor
even when it uses pinned (page-locked) memory.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig


def _make_config(tmpdir: str) -> HiCacheStorageConfig:
    """Create a minimal HiCacheStorageConfig for testing."""
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="test-model",
    )


class TestHiCacheFilePinnedMemoryRead(unittest.TestCase):
    """Test that HiCacheFile.get() correctly reads into pinned memory tensors."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Patch the env var so HiCacheFile uses our tmpdir
        self.env_patcher = patch.dict(
            os.environ, {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": self.tmpdir}
        )
        self.env_patcher.start()
        config = _make_config(self.tmpdir)
        self.storage = HiCacheFile(storage_config=config, file_path=self.tmpdir)

    def tearDown(self):
        self.env_patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_target_tensor(self, pin: bool = True) -> torch.Tensor:
        """Create a zero-filled target tensor matching the write shape."""
        # Shape must match what set() writes: (2, layer_num, page_size, head_num, head_dim)
        return torch.zeros(
            (2, 4, 64, 8, 16),
            dtype=torch.float32,
            device="cpu",
            pin_memory=pin,
        )

    def _make_data_tensor(self) -> torch.Tensor:
        """Create some non-zero test data."""
        return torch.randn((2, 4, 64, 8, 16), dtype=torch.float32)

    def test_get_pinned_memory(self):
        """Test that get() correctly reads into a pinned memory tensor."""
        data = self._make_data_tensor()
        key = "test_pinned_001"

        # Write data to file
        ok = self.storage.set(key, data)
        self.assertTrue(ok, "set() should succeed")

        # Create a pinned zero tensor and read into it
        target = self._make_target_tensor(pin=True)
        self.assertTrue((target == 0).all(), "Target should start as zeros")

        result = self.storage.get(key, target)

        self.assertIsNotNone(result, "get() should return the tensor on success")
        torch.testing.assert_close(
            result.cpu(), data.cpu(), msg="Data should match after read"
        )

    def test_get_non_pinned_memory(self):
        """Test that get() also works with non-pinned memory tensors."""
        data = self._make_data_tensor()
        key = "test_nonpinned_001"

        self.storage.set(key, data)

        target = self._make_target_tensor(pin=False)
        result = self.storage.get(key, target)

        self.assertIsNotNone(result)
        torch.testing.assert_close(result.cpu(), data.cpu())

    def test_get_file_not_found(self):
        """Test that get() returns None for missing keys."""
        target = self._make_target_tensor(pin=True)
        result = self.storage.get("nonexistent_key", target)
        self.assertIsNone(result)

    def test_batch_get_pinned_memory(self):
        """Test batch_get() with pinned memory tensors."""
        keys = ["batch_001", "batch_002", "batch_003"]
        data_list = [self._make_data_tensor() for _ in keys]

        for key, data in zip(keys, data_list):
            self.storage.set(key, data)

        targets = [self._make_target_tensor(pin=True) for _ in keys]
        results = self.storage.batch_get(keys, targets)

        self.assertEqual(len(results), len(keys))
        for i, (result, expected) in enumerate(zip(results, data_list)):
            self.assertIsNotNone(result, f"Result {i} should not be None")
            torch.testing.assert_close(result.cpu(), expected.cpu())

    def test_short_read_raises_io_error(self):
        """Test that a short read raises IOError with details."""
        key = "short_read_test"
        # Write directly to the file with a suffixed key to bypass set()
        suffixed_key = key + self.storage.config_suffix
        path = os.path.join(self.tmpdir, f"{suffixed_key}.bin")

        # Write only 10 bytes (much less than expected ~131072 bytes)
        with open(path, "wb") as f:
            f.write(b"\x00" * 10)

        target = self._make_target_tensor(pin=True)
        with self.assertRaises(IOError) as ctx:
            self.storage.get(key, target)
        self.assertIn("Short read", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
