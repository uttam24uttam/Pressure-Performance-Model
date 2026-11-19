"""Evidence card (one PNG per top-10 player). All numbers come from the pipeline tables passed in `p`."""
import textwrap

import numpy as np

from pipeline.common import plotstyle as ps


def _bars(ax, labels, normal, pressure, unit, title):
    y = np.arange(len(labels))
    h = 0.36
    ax.barh(y + h / 2, normal, height=h, color=ps.BLUE, label="normal balls")
    ax.barh(y - h / 2, pressure, height=h, color=ps.PINK, label="pressure balls")
    ax.set_yticks(y, labels, fontsize=8)
    ax.axvline(0, color=ps.GREY, lw=0.8)
    for yi, (n_, p_) in enumerate(zip(normal, pressure)):
        for val, dy in ((n_, h / 2), (p_, -h / 2)):
            if np.isfinite(val):
                ax.text(val, yi + dy, f" {val:+.1f}" if unit == "runs" else f" {val:.1f}", va="center",
                        ha="left" if val >= 0 else "right", fontsize=7, color=ps.INK)
    ax.set_title(title, fontsize=9, loc="left")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()


def card(p: dict, path):
    ps.apply()
    fig = ps.plt.figure(figsize=(12, 13.5))
    gs = fig.add_gridspec(5, 3, height_ratios=[0.9, 1.25, 1.25, 1.25, 1.1], hspace=0.7, wspace=0.55)
    role_word = "Batter" if p["role"] == "bat" else "Bowler"
    # ---- header
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.text(0, 0.95, f"#{p['rank']}  {p['name']}", fontsize=20, fontweight="bold", va="top", color=ps.INK)
    ax.text(0, 0.55, f"{role_word} · {p['teams']}", fontsize=10, color=ps.MUTED, va="top")
    kv = [("final score (RUP)", f"{p['RUP']:+.2f}"),
          ("rank range (80% of 1,000 re-runs)", f"{p['rank_p10']:.0f}–{p['rank_p90']:.0f}"),
          ("pressure balls", f"{p['pressure_balls']:.0f} in {p['pressure_innings']:.0f} {'innings' if p['role'] == 'bat' else 'spells'}"),
          ("wins added on pressure balls", f"{p['wpa_pressure']:+.2f} (win probability added)"),
          ("strongest pressure situation", p["strongest_type"])]
    for i, (k_, v_) in enumerate(kv):
        x = (i % 3) * 0.34
        yy = 0.28 - (i // 3) * 0.3
        ax.text(x, yy, k_, fontsize=8, color=ps.MUTED, va="top")
        ax.text(x, yy - 0.11, v_, fontsize=10.5, fontweight="bold", color=ps.INK, va="top")
    # ---- pressure vs normal (outcome and process)
    ax = fig.add_subplot(gs[1, 0])
    _bars(ax, ["outcome\n(runs above expected)", "process\n(" + ("shot decision" if p["role"] == "bat" else "delivery execution") + ")"],
          [p["B_raw"], p["proc_B_raw"]], [p["P_raw"], p["proc_P_raw"]], "runs",
          "Pressure vs normal balls: runs above expected per 100 balls")
    lo_, hi_ = ax.get_xlim()
    ax.set_xlim(lo_ - 0.08 * (hi_ - lo_), hi_ + 0.08 * (hi_ - lo_))
    ax.legend(fontsize=7, loc="lower right")
    # ---- R_P / R_B / D / C with 80% intervals
    ax = fig.add_subplot(gs[1, 1])
    labs = ["pressure rating", "usual rating", "change under pressure"]
    vals = [p["R_P"], p["R_B"], p["D_R"]]
    lo = [p["R_P_p10"], p["B_p10"], p["D_p10"]]
    hi = [p["R_P_p90"], p["B_p90"], p["D_p90"]]
    y = np.arange(3)
    ax.errorbar(vals, y, xerr=[np.subtract(vals, lo), np.subtract(hi, vals)], fmt="o", color=ps.PINK,
                ecolor=ps.INK, capsize=3, ms=7)
    for yi, (v_, h_) in enumerate(zip(vals, hi)):
        ax.text(h_, yi, f"  {v_:+.1f}", ha="left", va="center", fontsize=7.5)
    ax.set_xlim(min(0, min(lo)) - 2, max(hi) + 0.25 * (max(hi) - min(0, min(lo))) + 2)
    ax.set_yticks(y, labs, fontsize=8)
    ax.axvline(0, color=ps.GREY, lw=0.8)
    ax.invert_yaxis()
    ax.set_title("Ratings per 100 balls (bar = 80% of 1,000 re-runs)", fontsize=9, loc="left")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", visible=False)
    ax = fig.add_subplot(gs[1, 2])
    ax.errorbar([p["C"]], [0], xerr=[[p["C"] - p["C_p10"]], [p["C_p90"] - p["C"]]], fmt="o", color=ps.PINK,
                ecolor=ps.INK, capsize=3, ms=7)
    ax.axvline(p["C_pool_median"], color=ps.GREY, ls="--", lw=1)
    ax.text(p["C_pool_median"], 0.35, " pool median", fontsize=7, color=ps.MUTED)
    ax.set_yticks([0], ["C"], fontsize=8)
    ax.set_ylim(-0.6, 0.6)
    ax.set_title(f"Consistency {p['C']:.1%}: share of his {p['units']:.0f}\npressure matches above expected (adjusted)",
                 fontsize=9, loc="left")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", visible=False)
    # ---- raw vs expected
    ax = fig.add_subplot(gs[2, 0])
    if p["role"] == "bat":
        lab, unit = "strike rate", "SR"
    else:
        lab, unit = "economy", "econ"
    xs = np.arange(2)
    ax.bar(xs - 0.18, [p["raw_normal"], p["raw_pressure"]], 0.34, color=[ps.BLUE, ps.PINK], label="actual")
    ax.bar(xs + 0.18, [p["exp_normal"], p["exp_pressure"]], 0.34, color="white", edgecolor=[ps.BLUE, ps.PINK],
           hatch="//", label="expected (context)")
    for x_, a_, e_ in zip(xs, [p["raw_normal"], p["raw_pressure"]], [p["exp_normal"], p["exp_pressure"]]):
        ax.text(x_ - 0.18, a_, f"{a_:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(x_ + 0.18, e_, f"{e_:.1f}", ha="center", va="bottom", fontsize=7, color=ps.MUTED)
    ax.set_xticks(xs, ["normal", "pressure"], fontsize=8)
    ax.set_title(f"Raw {lab} vs expected", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.tick_params(axis="y", labelsize=7)
    # ---- process stats table
    ax = fig.add_subplot(gs[2, 1:])
    ax.axis("off")
    ax.set_title("Process on pressure vs normal balls (player | pool median)", fontsize=9, loc="left")
    rows = p["process_rows"]
    tb = ax.table(cellText=[[r[0], r[1], r[2], r[3], r[4]] for r in rows],
                  colLabels=["", "normal", "pressure", "pool normal", "pool pressure"], loc="center",
                  cellLoc="center", colLoc="center", colWidths=[0.36, 0.16, 0.16, 0.16, 0.16])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 1.35)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_edgecolor("#E3E5E8")
        if r_ == 0:
            cell.set_text_props(fontweight="bold", color=ps.MUTED)
        if c_ == 2 and r_ > 0:
            cell.set_text_props(color=ps.PINK, fontweight="bold")
    # ---- splits
    ax = fig.add_subplot(gs[3, 0:2])
    sl = p["splits"]
    y = np.arange(len(sl))
    ax.barh(y, [s[1] for s in sl], color=[ps.PINK if s[2] >= 30 else "#F3C3DB" for s in sl], height=0.6)
    ax.set_yticks(y, [f"{s[0]} ({s[2]} balls)" for s in sl], fontsize=8)
    for yi, s in enumerate(sl):
        ax.text(s[1], yi, f" {s[1]:+.1f}", va="center", ha="left" if s[1] >= 0 else "right", fontsize=7)
    ax.axvline(0, color=ps.GREY, lw=0.8)
    ax.invert_yaxis()
    xs_ = [s_[1] for s_ in sl]
    span = max(max(xs_), 0) - min(min(xs_), 0)
    ax.set_xlim(min(min(xs_), 0) - 0.12 * span, max(max(xs_), 0) + 0.15 * span)
    ax.set_title("Pressure balls by situation: runs above expected per 100 balls (pale = fewer than 30 balls)", fontsize=9, loc="left")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", visible=False)
    # ---- pressure types
    ax = fig.add_subplot(gs[3, 2])
    pt = p["ptypes"]
    y = np.arange(len(pt))
    cols = [ps.PINK if (t[0] == p["strongest_key"]) else (ps.BLUE if t[3] >= 30 else "#C9D2EA") for t in pt]
    ax.barh(y, [t[2] for t in pt], color=cols, height=0.6)
    ax.set_yticks(y, [f"{t[1]} ({t[3]})" for t in pt], fontsize=7.5)
    for yi, t in enumerate(pt):
        ax.text(t[2], yi, f" {t[2]:+.1f}", va="center", ha="left" if t[2] >= 0 else "right", fontsize=7)
    ax.axvline(0, color=ps.GREY, lw=0.8)
    ax.invert_yaxis()
    xt_ = [t[2] for t in pt]
    span = max(max(xt_), 0) - min(min(xt_), 0)
    ax.set_xlim(min(min(xt_), 0) - 0.15 * span, max(max(xt_), 0) + 0.25 * span)
    ax.set_title("Pressure situations: rating per 100 balls (balls)", fontsize=9, loc="left")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", visible=False)
    # ---- top 3 WPA balls
    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    ax.set_title("Three highest-WPA pressure balls", fontsize=9, loc="left")
    for i, line in enumerate(p["top_balls"][:3]):
        ax.text(0, 0.86 - i * 0.33, "\n".join(textwrap.wrap(line, 150)), fontsize=8.5, va="top", color=ps.INK)
    fig.text(0.5, 0.005, "Runs above expected = actual minus what an average player would produce in the same match "
             "situation (context models, out-of-fold). Pressure ball = one of the 20% of deliveries with the highest Leverage Index in its innings.",
             ha="center", fontsize=7, color=ps.MUTED)
    fig.savefig(path)
    ps.plt.close(fig)
