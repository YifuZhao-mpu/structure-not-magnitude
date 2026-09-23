"""
Sensor rig for GATE A. Camera and LiDAR sit at DIFFERENT origins, separated by a
baseline we control. Both trace the same analytic scene, so for every LiDAR return
we can decide co-visibility EXACTLY:

    a return X is camera-co-visible  <=>  the camera's first hit along the pixel ray
                                          through X is X itself (within eps)

That is the ground truth that real data cannot give us. Everything downstream
(contaminated labels, coherence ablations) is derived from it without estimation.
"""
import numpy as np
from scene import default_scene, _unit

INF = 1e9


def look_at(eye, target, up=(0, 0, 1)):
    """
    Returns camera-to-world rotation with columns [right, DOWN, forward].

    The 'down' (not 'up') second column is deliberate: it matches the OpenCV /
    gsplat convention, so this matrix can be handed straight to the rasteriser as
    c2w. With an 'up' second column the maths still closes on itself (our renderer
    and our projection would agree), but every rendered image comes out vertically
    flipped, which would silently corrupt figures and any manual occlusion check.
    """
    f = _unit(np.asarray(target, float) - np.asarray(eye, float))
    r = _unit(np.cross(f, np.asarray(up, float)))
    d = np.cross(f, r)                     # down = forward x right
    return np.stack([r, d, f], 1)          # columns: right, down, forward


class Rig:
    """camera at `eye`; LiDAR at eye + baseline (expressed in world axes)"""
    def __init__(self, eye, target, baseline, W=320, H=200, fov_deg=70.0):
        self.eye = np.asarray(eye, float)
        self.R = look_at(eye, target)
        self.lidar_o = self.eye + np.asarray(baseline, float)
        self.W, self.H = W, H
        self.f = (W / 2) / np.tan(np.radians(fov_deg) / 2)
        self.K = np.array([[self.f, 0, W / 2], [0, self.f, H / 2], [0, 0, 1]])

    # ---- camera ----
    def pixel_dirs(self):
        ys, xs = np.mgrid[0:self.H, 0:self.W]
        x = (xs.ravel() + 0.5 - self.W / 2) / self.f
        y = (ys.ravel() + 0.5 - self.H / 2) / self.f
        d_cam = np.stack([x, y, np.ones_like(x)], 1)
        return _unit(d_cam @ self.R.T)      # world dirs

    def render(self, scene):
        d = self.pixel_dirs()
        col, t, idx = scene.shade(self.eye, d)
        return (col.reshape(self.H, self.W, 3),
                t.reshape(self.H, self.W),        # distance along camera ray
                idx.reshape(self.H, self.W), d)

    # ---- lidar ----
    def lidar_rays(self, n_az=900, n_el=48, el_lo=-22.0, el_hi=8.0, yaw_span=80.0):
        az = np.radians(np.linspace(-yaw_span / 2, yaw_span / 2, n_az))
        el = np.radians(np.linspace(el_lo, el_hi, n_el))
        A, E = np.meshgrid(az, el)
        A, E = A.ravel(), E.ravel()
        # forward axis of the rig, rotated by (az, el)
        d_local = np.stack([np.sin(A) * np.cos(E), -np.sin(E), np.cos(A) * np.cos(E)], 1)
        return _unit(d_local @ self.R.T)

    def lidar_returns(self, scene, max_range=80.0, **kw):
        """
        max_range mirrors a real sensor's range limit (nuScenes HDL-32E is ~70-100 m).
        Without it an infinite ground plane returns hits at hundreds of metres, and
        those few far points would dominate an L1 depth loss and swamp the effect
        this experiment is trying to measure.
        """
        d = self.lidar_rays(**kw)
        t, idx = scene.trace(self.lidar_o, d)
        m = (t < INF) & (t <= max_range)
        return self.lidar_o + t[m, None] * d[m], idx[m], t[m]


def covisibility(rig, scene, pts, eps=0.05, pixel_centre=True):
    """
    EXACT per-return labels.
      visible : camera's first hit along that pixel ray IS this point
      occluded: camera hits something strictly nearer  -> the contaminated case
      oob     : falls outside the image
    Also returns the naive projected depth (what a projection pipeline would
    hand the optimiser) and the TRUE camera-visible depth at the same pixel.
    """
    v = pts - rig.eye
    cam = v @ rig.R                                   # into camera axes
    z = cam[:, 2]
    ok = z > 1e-3
    uv = np.full((len(pts), 2), -1.0)
    uv[ok] = (cam[ok, :2] / z[ok, None]) * rig.f + np.array([rig.W / 2, rig.H / 2])
    u, vv = uv[:, 0], uv[:, 1]
    inim = ok & (u >= 0) & (u < rig.W) & (vv >= 0) & (vv < rig.H)

    dist_to_cam = np.linalg.norm(v, axis=1)           # along-camera-ray distance to X

    # Two DIFFERENT rays are needed, and conflating them is a real defect in both
    # directions:
    #   * VISIBILITY is a geometric fact about the return itself -- can the camera see
    #     X? -- so it must be decided along the continuous ray through X. Deciding it
    #     on the pixel-centre ray reclassifies ordinary pixel quantisation on slanted
    #     surfaces as "occlusion" (measured: occluded fraction 0.5-3% -> 15-19%, with a
    #     median residual of 0.13 m, i.e. quantisation, not occlusion).
    #   * The CORRECT LABEL is what the renderer produces at the supervised pixel, and
    #     the renderer integrates the pixel-centre ray. Labelling with the continuous
    #     ray leaves labels and samples on different rays, which breaks magnitude
    #     matching between the ablations.
    dirs = v / np.maximum(dist_to_cam, 1e-9)[:, None]
    t_cont = np.full(len(pts), INF)
    if inim.any():
        t_cont[inim], _ = scene.trace(rig.eye, dirs[inim])

    t_pix = np.full(len(pts), INF)
    if pixel_centre:
        iu = np.clip(np.floor(u).astype(np.int64), 0, rig.W - 1)
        iv = np.clip(np.floor(vv).astype(np.int64), 0, rig.H - 1)
        xc = (iu + 0.5 - rig.W / 2) / rig.f
        yc = (iv + 0.5 - rig.H / 2) / rig.f
        d_pix = _unit(np.stack([xc, yc, np.ones_like(xc)], 1) @ rig.R.T)
        if inim.any():
            t_pix[inim], _ = scene.trace(rig.eye, d_pix[inim])
    else:
        t_pix = t_cont

    # A supervised term is only well defined where the sampled pixel-centre ray
    # actually hits something. Where it misses, t_pix stays at the INF sentinel, and a
    # finite 1e9 sentinel passes every isfinite() check while being a catastrophic
    # depth target -- and, once its "residual" enters the permutation pool, it poisons
    # the magnitude-matched controls with billion-metre values. Eligibility is computed
    # once and applied identically to every condition, which is what keeps the
    # supervised pixel set common across conditions.
    eligible = inim & np.isfinite(t_pix) & (t_pix < 1e8)

    state = np.full(len(pts), "oob", dtype=object)
    vis = eligible & (np.abs(t_cont - dist_to_cam) < eps)
    # Contamination is the studied phenomenon: a return the camera cannot see, whose
    # asserted depth is TOO FAR at the pixel that gets supervised. Continuous-ray
    # occlusion alone does not imply that -- 6.8% of occluded returns have a pixel
    # whose own ray terminates FURTHER than the return, which is boundary quantisation
    # rather than background-onto-foreground misprojection. Those are excluded, which
    # also keeps every residual positive and therefore every permuted target physical.
    occ = eligible & (t_cont < dist_to_cam - eps) & (dist_to_cam - t_pix > eps)
    state[vis], state[occ] = "visible", "occluded"
    return dict(uv=uv, inim=eligible, state=state,
                naive_depth=dist_to_cam,   # what naive projection asserts at the pixel
                true_depth=t_pix,          # what the renderer yields at that pixel
                true_depth_ray=t_cont,     # visibility geometry along the ray to X
                eligible=eligible,
                error=np.where(occ, dist_to_cam - t_pix, 0.0))


def rig_params(seed=None, baseline=(0.0, 0.0, 0.75)):
    """
    Parameters for ONE sensor rig + trajectory. `seed=None` reproduces the original
    hardcoded rig EXACTLY (regression anchor); a seed draws an independent one.

    The VERTICAL baseline stays exactly as the caller set it -- it is the controlled
    variable the whole occlusion argument turns on. Only the lateral mounting offset
    is jittered, which real rigs genuinely have and which varies the occlusion
    geometry without silently changing the controlled quantity.
    """
    b = [float(v) for v in baseline]
    if seed is None:
        return dict(rig_seed=None, baseline=b, x0=-6.0, dx=0.45, amp=0.35, freq=0.22,
                    eye_z=1.55, tgt_dx=10.0, tgt_ky=0.4, tgt_z=1.1, fov_deg=70.0)
    rng = np.random.default_rng(70000 + int(seed)); U = rng.uniform
    b[0] += float(U(-0.12, 0.12)); b[1] += float(U(-0.12, 0.12))
    return dict(rig_seed=int(seed), baseline=b,
                x0=float(U(-7.0, -5.0)), dx=float(U(0.40, 0.52)),
                amp=float(U(0.20, 0.50)), freq=float(U(0.18, 0.28)),
                eye_z=float(U(1.45, 1.68)), tgt_dx=float(U(8.5, 11.5)),
                tgt_ky=float(U(0.30, 0.50)), tgt_z=float(U(1.00, 1.25)),
                fov_deg=float(U(65.0, 75.0)))


def trajectory(n=24, baseline=(0.0, 0.0, 0.75), rig=None, **kw):
    """forward-driving poses; LiDAR mounted `baseline` above/beside the camera.
    `rig` is a `rig_params` dict; None keeps the original hardcoded trajectory."""
    if rig is None:
        rig = rig_params(None, baseline)
    kw.setdefault("fov_deg", rig["fov_deg"])
    rigs = []
    for i in range(n):
        x = rig["x0"] + rig["dx"] * i
        y = rig["amp"] * np.sin(i * rig["freq"])
        rigs.append(Rig([x, y, rig["eye_z"]],
                        [x + rig["tgt_dx"], y * rig["tgt_ky"], rig["tgt_z"]],
                        rig["baseline"], **kw))
    return rigs


if __name__ == "__main__":
    sc = default_scene()
    rigs = trajectory(n=8)
    tot = {"visible": 0, "occluded": 0, "oob": 0}
    errs = []
    for r in rigs:
        pts, pid, _ = r.lidar_returns(sc)
        cv = covisibility(r, sc, pts)
        for k in tot:
            tot[k] += int((cv["state"] == k).sum())
        e = cv["error"][cv["state"] == "occluded"]
        errs.append(e)
    errs = np.concatenate(errs)
    n_in = tot["visible"] + tot["occluded"]
    print("GATE A synthetic rig — EXACT co-visibility (no estimation)\n")
    print(f"  LiDAR returns in image : {n_in:,}")
    print(f"    camera-co-visible    : {tot['visible']:,}  ({100*tot['visible']/n_in:.2f}%)")
    print(f"    camera-OCCLUDED      : {tot['occluded']:,}  ({100*tot['occluded']/n_in:.2f}%)")
    print(f"  out of image           : {tot['oob']:,}")
    print(f"\n  induced depth error on occluded returns:")
    print(f"    mean {errs.mean():.2f} m | median {np.median(errs):.2f} m | max {errs.max():.2f} m")
