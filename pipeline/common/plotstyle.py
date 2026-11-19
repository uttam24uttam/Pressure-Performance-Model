"""Shared matplotlib style: RR-friendly pink/blue palette (CVD-checked: min OKLab dE 9.6 protan)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#254AA5"      # series 1 / "normal" / batting
PINK = "#D6247E"      # series 2 / "pressure" / bowling
GREY = "#8A8F98"      # reference lines only, never a data series
INK = "#1F2328"
MUTED = "#5B6170"
GRID = "#E3E5E8"
GOLD = "#D9A21B"      # categorical 3
SKY = "#3FA7D6"       # categorical 4  (4-colour set min OKLab dE: normal 26, protan 9.6, deutan 13.5, tritan 20.3)
CAT = [BLUE, PINK, GOLD, SKY]
MARKERS = ["o", "s", "^", "D"]
SEQ_CMAP = "RdPu"     # single-hue-family sequential (light -> dark pink)
DPI = 200


def apply():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
    })
