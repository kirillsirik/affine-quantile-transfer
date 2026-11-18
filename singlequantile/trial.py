"""One trial of the single-quantile design at the 0.90 level with known scores, from the grid's functions.

trial(rs, replicate, noise, coefficient_box, ball) follows fit_trial of singlequantile/grid/run_simulations.py
draw for draw. coefficient_box=10.0 and ball=True reproduce fit_trial; coefficient_box=None and ball=False
remove every bound.
"""
import numpy as np

TAU, N_SOURCE, N_TARGET, N_EVALUATION = 0.9, 1000, 100, 5000
DEPARTURES = (0.0, 0.5, 1.0)
SEED = 2026


def trial(rs, replicate, noise, coefficient_box, ball):
    """Rows (one per departure) of squared prediction errors and bound activity for one replication.

    rs is the module singlequantile/grid/run_simulations.py.
    """
    seed = np.random.SeedSequence([SEED, replicate, rs.NOISES.index(noise), rs.TAUS.index(TAU), N_TARGET, 0, 0])
    rng = np.random.default_rng(seed)
    xs, zs = rs.draw_x(rng, N_SOURCE)
    xt, zt = rs.draw_x(rng, N_TARGET)
    xe, ze = rs.draw_x(rng, N_EVALUATION)
    ws, wt, we = xs[:, 0], xt[:, 0], xe[:, 0]
    source_truth = rs.truth(zs, ws)
    source_noise = rs.noise_sample(rng, N_SOURCE, noise, TAU)
    target_noise = rs.noise_sample(rng, N_TARGET, noise, TAU)
    ys = source_truth + source_noise
    fs, ft, fe = rs.features(zs, ws), rs.features(zt, wt), rs.features(ze, we)
    bs, _, _ = rs.qr(fs, ys, TAU, coef_box=coefficient_box)
    source_fit_target, source_fit_evaluation = ft @ bs, fe @ bs

    def active(beta):
        return 0 if coefficient_box is None else int(np.sum(np.isclose(np.abs(beta), coefficient_box, atol=1e-5)))

    rows = []
    for departure in DEPARTURES:
        truth = rs.target_q(ze, we, departure)
        yt = rs.target_q(zt, wt, departure) + target_noise
        bt, _, _ = rs.qr(ft, yt, TAU, coef_box=coefficient_box)
        bp, _, _ = rs.qr(np.vstack([fs, ft]), np.r_[ys, yt], TAU, coef_box=coefficient_box)
        inputs = np.column_stack([np.ones(N_TARGET), source_fit_target, zt, wt])
        inputs_evaluation = np.column_stack([np.ones(N_EVALUATION), source_fit_evaluation, ze, we])
        if ball:
            ba, _, _, diagnostics = rs.affine_qr(inputs, yt, TAU)
            fallback = diagnostics["fallback"]
        else:
            ba, _, _ = rs.qr(inputs, yt, TAU)
            fallback = 0
        rows.append({"replicate": replicate, "noise": noise, "departure": departure,
                     "mse_transfer": float(np.mean((inputs_evaluation @ ba - truth) ** 2)),
                     "mse_target_only": float(np.mean((fe @ bt - truth) ** 2)),
                     "mse_pooled": float(np.mean((fe @ bp - truth) ** 2)),
                     "transfer_norm": float(np.linalg.norm(ba)),
                     "source_bound_active": active(bs), "target_bound_active": active(bt),
                     "pooled_bound_active": active(bp), "transfer_fallback": int(fallback)})
    return rows
