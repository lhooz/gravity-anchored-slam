#!/usr/bin/env python3
"""
plot_topomap.py — topological-map figure (Fig 5): the place/pose graph the system builds on a
revisit-rich circuit, loop closure OFF vs ON.

Revisit DETECTION is identical in both runs (the same matched keyframe pairs are recognised as
the same place). Each purple chord connects the two estimates of one such revisited place, so
its length is the residual inter-lap drift between them:

  A  Loop closure OFF : raw VO+anchor odometry -- matched revisits sit ~15 cm apart (drift).
  B  Loop closure ON  : pose-graph relaxation on those matches pulls the revisits together
                        to ~2 cm.

The chord lengths are read from the actual pose-graph keyframe estimates dumped by
scripts/slam_variance.py --compare-lc: `node_orig` (poses BEFORE relaxation) and
`node_corrected` (poses AFTER relaxation), both in the estimate frame. Chord length is
frame-invariant, so the reported mean gap is exact; for DISPLAY we align each node set to the
ground-truth keyframes `node_gt` with the same rigid SE(2) (rotation+translation, no scale)
used to score ATE, so the map overlays the GT loop. This makes the map numbers consistent with
the Fig 4 ATE (16.0 -> 2.7 cm) and the manuscript (~83% reduction; Fig 4 quantifies it).

Inputs (this dir, or $SLAM_FIG_DATA): loop_closure_results_graph.npz with
  node_orig, node_corrected (n_nodes x 3), node_gt (n_nodes x 2), lc_edges (node index pairs),
  rep_gt (dense GT track), rep_seed.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.environ.get('SLAM_FIG_DATA', _HERE)
_GRAPH = os.path.join(_DATA, 'loop_closure_results_graph.npz')
import sys; sys.path.insert(0, os.path.join(_HERE, '..'))
import figstyle

C_OFF = figstyle.C_OFF   # orange (loop closure OFF)
C_ON  = figstyle.C_ON    # green  (loop closure ON)
C_GT  = figstyle.C_TRUE  # near-black ground truth
C_LC  = figstyle.OKABE['purple']


def _align_se2(src, dst):
    """Rigid SE(2) Umeyama (rotation+translation, NO scale): best R,t mapping src->dst.
    Mirrors snn_slam_system.get_optimal_alignment_2d so the map uses the ATE convention."""
    m = ~(np.isnan(src[:, 0]) | np.isnan(dst[:, 0]))
    s, d = src[m], dst[m]
    if len(s) < 5:
        return np.eye(2), np.zeros(2)
    mu_s, mu_d = s.mean(0), d.mean(0)
    H = (s - mu_s).T @ (d - mu_d)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    return R, mu_d - R @ mu_s


def _resid(nodes, edges):
    """Mean matched-revisit chord length (cm). Frame-invariant."""
    d = [np.hypot(nodes[a, 0] - nodes[b, 0], nodes[a, 1] - nodes[b, 1]) for a, b in edges
         if not (np.isnan(nodes[a, 0]) or np.isnan(nodes[b, 0]))]
    return float(np.mean(d)) * 100 if d else float('nan')


def main():
    g = np.load(_GRAPH)
    node_gt = np.asarray(g['node_gt'], dtype=float)                     # GT frame (x,y)
    orig = np.asarray(g['node_orig'], dtype=float)[:, :2]               # estimate frame, before relax
    corr = np.asarray(g['node_corrected'], dtype=float)[:, :2]          # estimate frame, after relax
    lc = np.asarray(g['lc_edges'], dtype=int)
    gt = np.asarray(g['rep_gt'], dtype=float)
    seed = int(g['rep_seed'])

    # Align each pose-graph node set to the GT keyframes (rigid SE(2), no scale) for display.
    R0, t0 = _align_se2(orig, node_gt); node_off = (R0 @ orig.T).T + t0
    R1, t1 = _align_se2(corr, node_gt); node_on = (R1 @ corr.T).T + t1

    ne = len(lc)
    show = lc[np.linspace(0, ne - 1, min(ne, 28)).astype(int)] if ne else lc

    # Shared SQUARE window over both aligned node sets + GT, so both panels render at ONE metric
    # scale (1 m is the same length in A and B). With independent auto-scales the tight relaxed loop
    # was magnified ~1.6x relative to the drift spiral, understating the very shrinkage this figure
    # shows -- and making the on-page purple-chord lengths not comparable between panels.
    _pts = np.vstack([node_off[~np.isnan(node_off[:, 0])],
                      node_on[~np.isnan(node_on[:, 0])], gt])
    _xmid = 0.5 * (_pts[:, 0].min() + _pts[:, 0].max())
    _ymid = 0.5 * (_pts[:, 1].min() + _pts[:, 1].max())
    _half = 0.5 * max(np.ptp(_pts[:, 0]), np.ptp(_pts[:, 1])) * 1.12
    XLIM = (_xmid - _half, _xmid + _half); YLIM = (_ymid - _half, _ymid + _half)

    figstyle.apply()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.8, 4.6))

    def _panel(ax, node, col, title, letter):
        ax.plot(gt[:, 0], gt[:, 1], color=C_GT, lw=1.2, alpha=0.7, zorder=2)
        v = ~np.isnan(node[:, 0])
        # faint pose-graph polyline (keyframe order) as context
        ax.plot(node[v, 0], node[v, 1], color=col, lw=0.6, alpha=0.25, zorder=1)
        ax.scatter(node[v, 0], node[v, 1], s=6, c=col, alpha=0.55, edgecolors='none', zorder=4)
        for a, b in show:                                                                       # chords ON TOP
            if not (np.isnan(node[a, 0]) or np.isnan(node[b, 0])):
                ax.plot([node[a, 0], node[b, 0]], [node[a, 1], node[b, 1]], color=C_LC, lw=1.5,
                        alpha=0.9, solid_capstyle='round', zorder=7)
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect('equal', adjustable='box')
        figstyle.no_grid(ax); ax.set_xlabel('x (m)'); ax.set_title(title, loc='left')
        figstyle.panel(ax, letter)
        ax.annotate(f'mean revisit gap {_resid(node, lc):.0f} cm', xy=(0.04, 0.04),
                    xycoords='axes fraction', ha='left', va='bottom', fontsize=8, color=figstyle.C_INK,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d3d8dd', lw=0.7), zorder=9)

    _panel(axL, node_off, C_OFF, 'Loop closure OFF', 'A')
    _panel(axR, node_on,  C_ON,  'Loop closure ON',  'B')
    axL.set_ylabel('y (m)')

    # ONE shared legend below the panels (off the data): the per-panel upper-right legend sat on top
    # of panel A's drift spiral (its whole visual message). Place-node marker is neutral grey -- the
    # panel titles + colored dots already read as OFF/ON. The chord count is honest about decimation.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=C_GT, lw=1.4, alpha=0.8, label='ground truth'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=figstyle.C_MUTED,
               markeredgecolor='none', markersize=5, label='place nodes'),
        Line2D([0], [0], color=C_LC, lw=1.8, solid_capstyle='round',
               label=f'matched revisits (n={ne}; {len(show)} shown)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8.5, frameon=True,
               framealpha=1.0, edgecolor='#d3d8dd', bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(top=0.92, bottom=0.20, wspace=0.14)
    figstyle.save(fig, __file__, 'snn_slam_topomap')
    plt.close(fig)


if __name__ == '__main__':
    main()
