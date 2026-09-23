#!/usr/bin/env python3
"""
Regression anchor for the multi-scene refactor.

The scene/rig randomiser was added on top of a single hand-built scene that had already
produced results. The refactor is only trustworthy if `seed=None` still reproduces that
scene EXACTLY -- otherwise every previously reported number silently moved. These tests
assert that, plus the invariants the stratified metrics depend on.

Run:  python -m unittest code.synthetic.test_scene_regression -v
  or: cd code/synthetic && python -m unittest test_scene_regression -v
"""
import os, sys, json, tempfile, unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

from scene import default_scene, scene_params, build_scene, Scene   # noqa: E402
from capture import trajectory, rig_params                           # noqa: E402


def _probe_rays(n=5000, seed=0):
    o = np.array([0.0, 0.0, 1.5])
    d = np.random.default_rng(seed).normal(size=(n, 3))
    return o, d / np.linalg.norm(d, axis=1, keepdims=True)


class TestSceneSeedNoneIsByteIdentical(unittest.TestCase):
    """seed=None must reproduce the original hand-built layout exactly."""

    def test_traced_geometry_matches_default_scene(self):
        o, d = _probe_rays()
        for wall in (24.0, 28.0, 38.0):
            a = default_scene(wall)
            b = build_scene(scene_params(None, wall))
            ta, ia = a.trace(o, d)
            tb, ib = b.trace(o, d)
            self.assertTrue(np.array_equal(ta, tb), f"distances differ at wall={wall}")
            self.assertTrue(np.array_equal(ia, ib), f"primitive ids differ at wall={wall}")

    def test_default_rig_matches_original_hardcoded_trajectory(self):
        rigs = trajectory(n=24)
        for i, r in enumerate(rigs):
            self.assertTrue(np.allclose(
                r.eye, [-6.0 + 0.45 * i, 0.35 * np.sin(i * 0.22), 1.55], atol=0, rtol=0))
        # focal length recorded in every pre-refactor dataset meta.json
        self.assertAlmostEqual(rigs[0].f, 228.50368107873834, places=10)


class TestPrimitiveSchemaIsStable(unittest.TestCase):
    """The stratified metrics address primitives BY INDEX. If a randomised scene ever
    reorders or changes the primitive count, mae_wall / mae_pole / mae_car silently
    start measuring different objects."""

    EXPECTED = ["road", "wall", "carL", "carR", "pole"]

    def test_schema_order_and_count_fixed_across_seeds(self):
        for seed in [None] + list(range(25)):
            p = scene_params(seed, 24.0)
            self.assertEqual(list(p["schema"]), self.EXPECTED, f"schema moved at seed={seed}")
            sc = build_scene(p)
            self.assertEqual(len(sc.prims), 5, f"primitive count moved at seed={seed}")

    def test_boxes_rest_on_the_ground_plane(self):
        for seed in range(25):
            p = scene_params(seed, 24.0)
            for k in ("carL", "carR"):
                self.assertAlmostEqual(p[k]["c"][2], p[k]["h"][2], places=12,
                                       msg=f"{k} floats or sinks at seed={seed}")


class TestSeededScenesAreDistinct(unittest.TestCase):
    def test_seeds_give_different_geometry(self):
        o, d = _probe_rays()
        sigs = []
        for seed in range(10):
            t, _ = build_scene(scene_params(seed, 24.0)).trace(o, d)
            sigs.append(np.round(t[np.isfinite(t) & (t < 1e8)][:200], 6).tobytes())
        self.assertEqual(len(set(sigs)), 10, "two scene seeds produced identical geometry")

    def test_controlled_vertical_baseline_is_preserved(self):
        """Only the lateral mounting offset may be randomised; the vertical baseline is
        the controlled variable the occlusion argument turns on."""
        for seed in range(25):
            rp = rig_params(seed, (0.0, 0.0, 0.75))
            self.assertAlmostEqual(rp["baseline"][2], 0.75, places=12,
                                   msg=f"vertical baseline drifted at seed={seed}")


class TestReleasedSceneMetadataMatchesGenerator(unittest.TestCase):
    """The ten released scenes must still be reproducible from their recorded seeds."""

    def test_meta_matches_regenerated_params(self):
        for k in range(10):
            mp = os.path.join(ROOT, "data", "multiscene", f"sc{k:02d}", "meta.json")
            if not os.path.exists(mp):
                self.skipTest("released scenes not present in this checkout")
            meta = json.load(open(mp))
            self.assertEqual(meta["scene_seed"], k)
            self.assertEqual(meta["scene_params"], scene_params(k, meta["wall_dist_requested"]))
            self.assertEqual(meta["rig_params"],
                             rig_params(k, (0.0, 0.0, 0.75))
                             if meta["rig_params"]["rig_seed"] == k else meta["rig_params"])


class TestSeededSceneRegeneratesByteIdentically(unittest.TestCase):
    """The Data-availability statement says the analytic scene arrays are rebuilt from
    their seeds rather than shipped. That is only honest if it is true, so assert it.

    Slow (it runs the generator), so it is opt-in: set ARS_REGEN_TEST=1.
    """

    def test_scene_00_regenerates(self):
        if os.environ.get("ARS_REGEN_TEST") != "1":
            self.skipTest("set ARS_REGEN_TEST=1 to run the full regeneration check")
        import subprocess, tempfile, glob
        ref = os.path.join(ROOT, "data", "multiscene_v3", "sc00")
        if not os.path.exists(os.path.join(ref, "meta.json")):
            self.skipTest("released scenes not present in this checkout")
        here = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "sc00")
            for cmd in (["gen_dataset.py", "--out", out, "--wall-dist", "24.0",
                         "--baseline", "0.0", "0.0", "0.75", "--views", "24",
                         "--seed", "0", "--scene-seed", "0"],
                        ["add_native.py", "--data", out]):
                r = subprocess.run([sys.executable, os.path.join(here, cmd[0])] + cmd[1:],
                                   cwd=ROOT, capture_output=True)
                self.assertEqual(r.returncode, 0, r.stderr.decode()[-500:])
            n = 0
            for f in sorted(glob.glob(f"{ref}/**/*.npy", recursive=True) +
                            glob.glob(f"{ref}/*.npz")):
                rel = os.path.relpath(f, ref)
                g = os.path.join(out, rel)
                self.assertTrue(os.path.exists(g), f"missing after regeneration: {rel}")
                if f.endswith(".npz"):
                    a, b = np.load(f), np.load(g)
                    for k in a.files:
                        self.assertTrue(np.array_equal(a[k], b[k]), f"{rel}[{k}] differs")
                else:
                    self.assertTrue(np.array_equal(np.load(f), np.load(g)),
                                    f"{rel} differs")
                n += 1
            self.assertEqual(n, 120, f"expected 120 arrays, compared {n}")
            a = json.load(open(os.path.join(ref, "meta.json")))
            b = json.load(open(os.path.join(out, "meta.json")))
            for k in ("scene_params", "rig_params", "wall_dist_requested", "stats"):
                self.assertEqual(a.get(k), b.get(k), f"meta[{k}] differs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
