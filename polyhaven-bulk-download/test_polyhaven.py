"""Offline check of polyhaven.py: md5-checked download, resume and mismatch handling (through file:// URLs), and the
file list per asset kind with the resolution fallback (with the API call replaced by a fixed answer).
    python -m unittest test_polyhaven -v"""
import contextlib, hashlib, io, json, os, tempfile, unittest
from unittest.mock import patch
from pathlib import Path

import polyhaven


class Fetch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = Path(self.tmp.name)
        self.src = d / "src.bin"; self.src.write_bytes(b"poly" * 1000)
        self.url, self.md5 = self.src.as_uri(), hashlib.md5(self.src.read_bytes()).hexdigest()
        self.dst = str(d / "out" / "a" / "src.bin")

    def tearDown(self):
        self.tmp.cleanup()

    def test_download_then_resume(self):
        self.assertEqual(polyhaven.fetch(self.url, self.dst, self.md5), "ok")
        self.assertEqual(Path(self.dst).read_bytes(), self.src.read_bytes())
        self.assertEqual(polyhaven.fetch(self.url, self.dst, self.md5), "have")

    def test_mismatch_keeps_nothing(self):
        r = polyhaven.fetch(self.url, self.dst, "0" * 32)
        self.assertTrue(r.startswith("MD5 MISMATCH"))
        self.assertFalse(os.path.exists(self.dst) or os.path.exists(self.dst + ".part"))

    def test_bad_file_on_disk_is_replaced(self):
        os.makedirs(os.path.dirname(self.dst)); Path(self.dst).write_bytes(b"truncated")
        self.assertEqual(polyhaven.fetch(self.url, self.dst, self.md5), "ok")
        self.assertEqual(Path(self.dst).read_bytes(), self.src.read_bytes())


def entry(name):
    return {"url": f"https://dl.example/{name}", "md5": name, "size": 10}


FILES = {  # /files/<id> answers, one per asset kind
    "m": {"fbx": {"2k": {"fbx": {**entry("m_2k.fbx"), "include": {"textures/m_diff_2k.jpg": entry("d")}}}},
          "blend": {"4k": {"blend": entry("m_4k.blend")}}},
    "h": {"hdri": {"4k": {"hdr": entry("h_4k.hdr"), "exr": entry("h_4k.exr")}}},
    "t": {"Diffuse": {"4k": {"jpg": entry("t_diff_4k.jpg"), "png": entry("t_diff_4k.png")}},
          "Rough": {"1k": {"png": entry("t_rough_1k.png")}},
          "Displacement": {"8k": {"png": entry("t_disp_8k.png")}},
          "gltf": {"4k": {"gltf": entry("skip.gltf")}},
          "thumb": "not a dict"},
}


class FilesFor(unittest.TestCase):
    def setUp(self):
        self.js = polyhaven.js
        polyhaven.js = lambda path: FILES[path.rsplit("/", 1)[1]]

    def tearDown(self):
        polyhaven.js = self.js

    def test_model_falls_back_to_lower_res_with_textures(self):
        got = polyhaven.files_for("m", "models", "4k")
        self.assertEqual([x[1] for x in got], ["m_2k.fbx", "textures/m_diff_2k.jpg"])

    def test_hdri_prefers_hdr(self):
        self.assertEqual([x[1] for x in polyhaven.files_for("h", "hdris", "4k")], ["h_4k.hdr"])

    def test_textures_jpg_first_skip_formats(self):
        got = [x[1] for x in polyhaven.files_for("t", "textures", "4k")]
        self.assertEqual(got, ["t_diff_4k.jpg", "t_rough_1k.png"])  # no 8k-only map, no gltf, no non-map keys

    def test_no_higher_res_fallback(self):
        self.assertEqual([x[1] for x in polyhaven.files_for("t", "textures", "1k")], ["t_rough_1k.png"])


class Main(unittest.TestCase):
    def run_cli(self, catalog, dry):
        def respond(path):
            return catalog if path.startswith("/assets?") else FILES[path.rsplit("/", 1)[1]]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            argv = ["polyhaven.py", "--out", str(out), "--res", "1k", "--models", "nature"]
            if dry:
                argv.append("--dry")
            stream = io.StringIO()
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch("sys.argv", argv))
                stack.enter_context(patch.object(polyhaven, "js", side_effect=respond))
                fetch = stack.enter_context(patch.object(polyhaven, "fetch", return_value="ok"))
                stack.enter_context(contextlib.redirect_stdout(stream))
                try:
                    polyhaven.main()
                except SystemExit as exc:
                    status = exc.code
                else:
                    status = 0
            manifest = out / "manifest.jsonl"
            plan = [json.loads(row) for row in manifest.read_text().splitlines()] if manifest.exists() else []
            return status, stream.getvalue(), fetch.call_count, plan, out.exists()

    def test_all_skipped_reports_asset_and_fails_in_both_modes(self):
        for dry in (True, False):
            with self.subTest(dry=dry):
                status, output, downloads, plan, out_exists = self.run_cli({"m": {}}, dry)
                self.assertEqual(status, 1)
                self.assertIn("skipped models/m", output)
                self.assertIn("skipped assets 1", output)
                self.assertEqual(downloads, 0)
                self.assertEqual(plan, [])
                self.assertFalse(out_exists)

    def test_mixed_batch_downloads_only_supported_assets(self):
        ready = {"fbx": {"1k": {"fbx": entry("ready_1k.fbx")}}}
        with patch.dict(FILES, {"ready": ready}):
            for dry in (True, False):
                with self.subTest(dry=dry):
                    status, output, downloads, plan, out_exists = self.run_cli({"m": {}, "ready": {}}, dry)
                    self.assertEqual(status, 0)
                    self.assertIn("skipped models/m", output)
                    self.assertIn("skipped assets 1", output)
                    self.assertEqual(downloads, 0 if dry else 1)
                    self.assertEqual([row["id"] for row in plan], [] if dry else ["ready"])
                    self.assertEqual(out_exists, not dry)

    def test_empty_selection_fails_without_creating_output(self):
        for dry in (True, False):
            with self.subTest(dry=dry):
                status, output, downloads, plan, out_exists = self.run_cli({}, dry)
                self.assertEqual(status, 1)
                self.assertIn("no assets matched", output)
                self.assertEqual(downloads, 0)
                self.assertEqual(plan, [])
                self.assertFalse(out_exists)


if __name__ == "__main__":
    unittest.main()
