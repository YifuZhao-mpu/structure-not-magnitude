"""
GATE A: an analytic synthetic scene where TRUE visibility is COMPUTABLE, not estimated.

This exists because of the identification problem Codex raised: on real data, a
visibility "certificate" built from accumulated LiDAR + boxes cannot distinguish a
scene with a thin unannotated occluder from one without. No agreement rule fixes
missing evidence. So the causal claim has to be established where geometry is known.

Everything here is closed-form ray-primitive intersection:
  - exact camera-visible depth at every pixel
  - exact LiDAR return along every LiDAR ray, from its OWN origin
  - therefore exact co-visibility, per return, with no estimation anywhere

Scene layout is deliberately driving-like and contains the structures the whole
argument is about: a thin pole (easy to miss / easy to wrongly certify), box
occluders with sharp silhouettes, and a far wall that supplies the WRONG depth
when a background return is misprojected onto a foreground object.
"""
import numpy as np

INF = 1e9


# ---------------------------------------------------------------- primitives
class Plane:
    """point p0, unit normal n, optional finite extent in two tangent dirs"""
    def __init__(self, p0, n, color, half=None, u=None, v=None):
        self.p0 = np.asarray(p0, float); self.n = _unit(np.asarray(n, float))
        self.color = np.asarray(color, float); self.half = half
        self.u = _unit(np.asarray(u, float)) if u is not None else None
        self.v = _unit(np.asarray(v, float)) if v is not None else None

    def intersect(self, o, d):
        denom = d @ self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.p0 - o) @ self.n) / denom
        t = np.where(np.abs(denom) < 1e-9, INF, t)
        t = np.where(t > 1e-4, t, INF)
        if self.half is not None:
            p = o + t[:, None] * d
            loc = p - self.p0
            a, b = loc @ self.u, loc @ self.v
            t = np.where((np.abs(a) <= self.half[0]) & (np.abs(b) <= self.half[1]), t, INF)
        return t


class Box:
    """axis-aligned box, centre c, half-extents h"""
    def __init__(self, c, h, color):
        self.c = np.asarray(c, float); self.h = np.asarray(h, float)
        self.color = np.asarray(color, float)

    def intersect(self, o, d):
        lo, hi = self.c - self.h, self.c + self.h
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o) / d, (hi - o) / d
        tmin = np.maximum.reduce(np.minimum(t1, t2), axis=1)
        tmax = np.minimum.reduce(np.maximum(t1, t2), axis=1)
        hit = (tmax >= np.maximum(tmin, 1e-4))
        return np.where(hit, np.where(tmin > 1e-4, tmin, tmax), INF)


class Cylinder:
    """vertical (z-axis) finite cylinder -- the 'thin pole' case"""
    def __init__(self, cx, cy, r, z0, z1, color):
        self.cx, self.cy, self.r, self.z0, self.z1 = cx, cy, r, z0, z1
        self.color = np.asarray(color, float)

    def intersect(self, o, d):
        ox, oy = o[:, 0] - self.cx, o[:, 1] - self.cy
        dx, dy = d[:, 0], d[:, 1]
        a = dx * dx + dy * dy
        b = 2 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - self.r * self.r
        disc = b * b - 4 * a * c
        ok = (disc > 0) & (a > 1e-12)
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t = INF * np.ones(len(d))
        for root in ((-b - sq) / (2 * np.where(a > 1e-12, a, 1)),
                     (-b + sq) / (2 * np.where(a > 1e-12, a, 1))):
            z = o[:, 2] + root * d[:, 2]
            good = ok & (root > 1e-4) & (z >= self.z0) & (z <= self.z1) & (root < t)
            t = np.where(good, root, t)
        return t


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------- the scene
class Scene:
    def __init__(self, prims):
        self.prims = prims

    def trace(self, o, d):
        """returns (t, prim_index). o,d: [N,3]. d must be unit."""
        N = len(d)
        if o.ndim == 1:
            o = np.broadcast_to(o, (N, 3))
        best = INF * np.ones(N); idx = -np.ones(N, np.int64)
        for i, p in enumerate(self.prims):
            t = p.intersect(o, d)
            m = t < best
            best, idx = np.where(m, t, best), np.where(m, i, idx)
        return best, idx

    def shade(self, o, d):
        """simple lambert-ish shading so the RGB images have usable texture"""
        t, idx = self.trace(o, d)
        col = np.zeros((len(d), 3))
        hit = idx >= 0
        p = (o if o.ndim > 1 else np.broadcast_to(o, (len(d), 3))) + np.where(t < INF, t, 0)[:, None] * d
        for i, pr in enumerate(self.prims):
            m = idx == i
            if not m.any():
                continue
            base = pr.color
            # procedural checker so geometry is actually observable from RGB
            q = p[m]
            chk = ((np.floor(q[:, 0] * 0.7) + np.floor(q[:, 1] * 0.7)
                    + np.floor(q[:, 2] * 0.7)) % 2)
            col[m] = base * (0.68 + 0.32 * chk[:, None])
        col[~hit] = np.array([0.45, 0.55, 0.72])   # sky
        return np.clip(col, 0, 1), t, idx


def scene_params(seed=None, wall_dist=38.0):
    """
    Parameters for ONE scene instance.

    `seed=None` reproduces the original hand-built layout EXACTLY -- that is the
    regression anchor: every dataset generated before multi-scene support must be
    byte-reproducible through this path, otherwise the refactor has silently moved
    the results it is supposed to preserve.

    A seed draws an INDEPENDENT scene. The five-primitive schema (road, wall, carL,
    carR, pole) is deliberately held fixed while its parameters vary, because the
    stratified metrics address primitives by POSITIONAL INDEX (0..4). Randomising the
    schema itself would silently re-label `mae_wall` / `mae_pole` / `mae_car`.

    Why this exists: with one hand-built scene, seeds are training-noise replicates,
    not independent scenes, so a paired t over seeds has the wrong statistical unit.
    Scene-level replication is what licenses a scene-level claim.
    """
    if seed is None:
        return dict(
            schema=["road", "wall", "carL", "carR", "pole"], scene_seed=None,
            road=dict(half=[45.0, 18.0], color=[0.32, 0.32, 0.35]),
            wall=dict(dist=float(wall_dist), half=[14.0, 7.0], color=[0.60, 0.55, 0.48]),
            carL=dict(c=[14.0, 2.4, 0.85], h=[2.2, 0.95, 0.85], color=[0.70, 0.22, 0.20]),
            carR=dict(c=[19.5, -2.6, 0.80], h=[2.1, 0.90, 0.80], color=[0.20, 0.32, 0.68]),
            pole=dict(cx=11.0, cy=-1.5, r=0.085, z0=0.0, z1=3.2, color=[0.85, 0.80, 0.25]),
        )

    rng = np.random.default_rng(90000 + int(seed))
    U = rng.uniform

    def jitter(c, d=0.10):
        return [float(np.clip(v + U(-d, d), 0.05, 0.95)) for v in c]

    # wall distance sets the occlusion residual magnitude, i.e. the regime the claim is
    # about. It is jittered only +/-15% so every scene stays in the SAME regime as the
    # caller asked for; the achieved residual is recorded per scene in meta.json.
    wd = float(wall_dist) * U(0.85, 1.15)

    hL = [U(1.8, 2.6), U(0.80, 1.10), U(0.70, 1.00)]
    hR = [U(1.8, 2.6), U(0.80, 1.10), U(0.70, 1.00)]
    # a box must rest ON the road, so its centre height equals its half-height
    return dict(
        schema=["road", "wall", "carL", "carR", "pole"], scene_seed=int(seed),
        road=dict(half=[U(38.0, 50.0), U(14.0, 22.0)], color=jitter([0.32, 0.32, 0.35])),
        wall=dict(dist=wd, half=[U(11.0, 17.0), U(6.0, 9.0)], color=jitter([0.60, 0.55, 0.48])),
        carL=dict(c=[U(10.0, 18.0), U(1.5, 3.5), hL[2]], h=hL, color=jitter([0.70, 0.22, 0.20], 0.15)),
        carR=dict(c=[U(16.0, 24.0), U(-3.6, -1.6), hR[2]], h=hR, color=jitter([0.20, 0.32, 0.68], 0.15)),
        pole=dict(cx=U(8.0, 14.0), cy=U(-2.5, 2.5), r=U(0.060, 0.120),
                  z0=0.0, z1=U(2.6, 3.8), color=jitter([0.85, 0.80, 0.25], 0.12)),
    )


def build_scene(p):
    """Rebuild a Scene from a `scene_params` dict. Primitive ORDER is the contract:
    0=road 1=wall 2=carL 3=carR 4=pole -- consumed by the stratified metrics."""
    return Scene([
        Plane([0, 0, 0], [0, 0, 1], p["road"]["color"],
              half=list(p["road"]["half"]), u=[1, 0, 0], v=[0, 1, 0]),
        Plane([p["wall"]["dist"], 0, 0], [-1, 0, 0], p["wall"]["color"],
              half=list(p["wall"]["half"]), u=[0, 1, 0], v=[0, 0, 1]),
        Box(p["carL"]["c"], p["carL"]["h"], p["carL"]["color"]),
        Box(p["carR"]["c"], p["carR"]["h"], p["carR"]["color"]),
        Cylinder(p["pole"]["cx"], p["pole"]["cy"], p["pole"]["r"],
                 p["pole"]["z0"], p["pole"]["z1"], p["pole"]["color"]),
    ])


def default_scene(wall_dist=38.0):
    """
    driving-like: road, far wall, two box occluders, one thin pole.

    `wall_dist` controls the occlusion residual magnitude (residual = wall - occluder
    distance). The default 38 m yields a median residual of ~26 m, which is ~2.5x the
    8-13 m measured on real nuScenes. Pulling the wall in lets us re-test the effect at
    a REALISTIC residual magnitude -- residuals stay physically generated by geometry
    rather than rescaled by hand, which would destroy the occlusion's physical meaning.

    Kept as the seed=None path of `scene_params` so every pre-existing dataset stays
    byte-reproducible.
    """
    return build_scene(scene_params(None, wall_dist))
