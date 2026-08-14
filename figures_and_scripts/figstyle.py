"""Shared figure style for the 2D SLAM paper -- one refined, colorblind-safe standard.

Okabe-Ito palette; consistent sans-serif type; soft slate spines/ticks; a light
HORIZONTAL-only grid sitting behind the data; clean framed legends; bold panel
labels; vector (PDF) + 300-dpi PNG output to both the local figureN dir and the
2D manuscript figures dir.

Round-4 refinement (visual only -- no data, chart types, or annotations change):
  * type: smaller, quieter titles; consistent label/tick sizing; Source Sans 3
    first in the stack (falls back to Helvetica/Arial/DejaVu) so the plots match
    the hand-authored Fig 1 schematic when that font is installed.
  * grid: faint solid horizontal rules only (axes.grid.axis = 'y'), drawn behind
    the data; no heavy dotted cross-hatch.
  * spines/ticks: softened to slate, top/right removed, ticks pointing out.
  * legend: square, hairline slate edge on white -- readable over data, not boxy.
  * helpers: panel() for A/B/C labels, no_grid() for spatial-map panels, and
    fmt_si() for compact tick/annotation numbers (14.2M, 320k, ...).

Usage from a figures_and_scripts/figureN/ script:
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import figstyle; figstyle.apply()
    ...
    figstyle.save(fig, __file__, 'snn_slam_xxx')
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Okabe-Ito colorblind-safe palette ───────────────────────────────────────
OKABE = {
    'black': '#000000', 'orange': '#E69F00', 'skyblue': '#56B4E9', 'green': '#009E73',
    'yellow': '#F0E442', 'blue': '#0072B2', 'vermillion': '#D55E00', 'purple': '#CC79A7',
    'grey': '#8f99a2',
}
# Semantic roles (consistent across every figure)
C_TRUE   = '#26313d'             # ground truth (near-black slate, softer than pure black)
C_IMU    = OKABE['vermillion']   # pure IMU / dead-reckoning / unanchored
C_OL     = OKABE['orange']       # SNN open-loop
C_CL     = OKABE['green']        # SNN closed-loop / "ours"
C_ANCH   = OKABE['blue']         # gravity-anchored
C_OFF    = OKABE['orange']       # loop closure OFF
C_ON     = OKABE['green']        # loop closure ON

# Neutral supporting tones (use these instead of ad-hoc greys/black)
C_INK    = '#26313d'             # primary ink for titles/labels
C_MUTED  = '#5c6b7a'             # secondary text / annotations
C_SPINE  = '#3a4652'             # axis spines + ticks
C_GRID   = '#e2e6ea'             # faint grid rules


def apply():
    # Applied key-by-key so that a param unsupported by an older matplotlib is
    # skipped (with a note) rather than aborting the whole figure run.
    params = {
        # canvas / export
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42,   # editable (TrueType) text in vector output
        # type
        'font.family': 'sans-serif',
        'font.sans-serif': ['Source Sans 3', 'Source Sans Pro', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9.5, 'axes.labelsize': 10.5, 'axes.labelweight': 'normal',
        'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlepad': 7.0,
        'axes.titlecolor': C_INK, 'axes.labelcolor': C_INK,
        'text.color': C_INK,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
        # spines + ticks (soft slate, only left/bottom, pointing out)
        'axes.edgecolor': C_SPINE, 'axes.linewidth': 0.9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.color': C_SPINE, 'ytick.color': C_SPINE,
        'xtick.labelcolor': C_INK, 'ytick.labelcolor': C_INK,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
        'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
        # grid: faint, solid, HORIZONTAL only, behind the data
        'axes.grid': True, 'axes.grid.axis': 'y', 'axes.axisbelow': True,
        'grid.color': C_GRID, 'grid.linestyle': '-', 'grid.linewidth': 0.8, 'grid.alpha': 1.0,
        # lines / legend
        'lines.linewidth': 1.9, 'lines.solid_capstyle': 'round',
        'legend.frameon': True, 'legend.framealpha': 0.95, 'legend.facecolor': 'white',
        'legend.edgecolor': '#d3d8dd', 'legend.fancybox': False, 'legend.borderpad': 0.5,
        'legend.handlelength': 1.9, 'legend.columnspacing': 1.3, 'legend.labelspacing': 0.4,
        'legend.title_fontsize': 8.5,
    }
    skipped = []
    for k, v in params.items():
        try:
            plt.rcParams[k] = v
        except (KeyError, ValueError):
            skipped.append(k)
    if skipped:
        print(f"  [figstyle] matplotlib {matplotlib.__version__} ignored "
              f"{len(skipped)} newer rcParams: {', '.join(skipped)}")


def panel(ax, label, x=-0.115, y=1.02):
    """Bold A/B/C panel label at the top-left of an axes (in axes fraction)."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11.5, fontweight='bold',
            color=C_INK, va='bottom', ha='right')


def no_grid(ax):
    """Turn the grid off -- for spatial-map panels where a y-only grid is misleading."""
    ax.grid(False)


def style_legend(leg):
    """Nudge a legend frame to the house hairline style (safe no-op if leg is None)."""
    if leg is None:
        return leg
    fr = leg.get_frame()
    fr.set_linewidth(0.8)
    fr.set_edgecolor('#d3d8dd')
    return leg


def fmt_si(v, digits=1):
    """Compact SI-ish number for tick/annotation labels: 14.2M, 1.4M, 320k, 27."""
    a = abs(v)
    if a >= 1e9:
        return f'{v/1e9:.{digits}f}G'
    if a >= 1e6:
        return f'{v/1e6:.{digits}f}M'
    if a >= 1e3:
        return f'{v/1e3:.0f}k'
    return f'{v:.0f}'


def save(fig, script_file, basename):
    """Write <basename>.pdf (vector) + <basename>.png (300 dpi) to the local figureN dir AND
    the 2D manuscript figures/ dir. Returns the manuscript PDF path."""
    here = os.path.dirname(os.path.abspath(script_file))
    targets = [here]
    man = os.path.normpath(os.path.join(here, '..', '..', '..',
                                        'manuscript', 'figures'))
    if os.path.isdir(man):
        targets.append(man)
    out = None
    for d in targets:
        for ext in ('pdf', 'png'):
            p = os.path.join(d, f'{basename}.{ext}')
            fig.savefig(p)
            if d == man and ext == 'pdf':
                out = p
        print(f"  saved {basename}.pdf/.png -> {d}")
    return out
