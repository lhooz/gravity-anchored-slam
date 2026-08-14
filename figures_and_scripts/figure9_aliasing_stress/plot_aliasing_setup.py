#!/usr/bin/env python3
"""SI schematic of the perceptual-aliasing stress test: standard circuit (unique landmark
ring) vs. the k=2 aliasing circuit (rotationally symmetric ring), drawn from the ACTUAL
obstacle generators. Two loop positions half a lap apart face identical landmark arcs on the
aliased course, so spatially-distant places look alike."""
import os, sys
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, '..', '..', 'neuro-symbolic-slam')))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, '..', '..', 'neuro-symbolic-slam', 'src')))
import figstyle
import jax
from sparse_forest import (generate_circuit_obstacles, generate_circuit_obstacles_aliased,
                           N_CIRCUIT_OBS, CIRCUIT_RADIUS, ROOM_W, ROOM_H)

std = np.array(generate_circuit_obstacles(jax.random.PRNGKey(7)))
ali = np.array(generate_circuit_obstacles_aliased(jax.random.PRNGKey(7), kfold=2))
base = N_CIRCUIT_OBS // 2
cx0, cy0 = ROOM_W / 2, ROOM_H / 2

figstyle.apply()
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.3))


def draw(ax, obs, title, letter, symmetric):
    ax.add_patch(Rectangle((0, 0), ROOM_W, ROOM_H, fill=False, ec='k', lw=1.5))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cx0 + CIRCUIT_RADIUS * np.cos(th), cy0 + CIRCUIT_RADIUS * np.sin(th),
            '--', color=figstyle.C_ANCH, lw=1.3, label='0.4 m loop')
    for i, o in enumerate(obs):
        # base sector uses the Okabe 'orange' key so the swatch matches the word "orange" in the key
        # and caption (it was vermillion, a different hue that figstyle reserves for the IMU role).
        col = (figstyle.OKABE['orange'] if (symmetric and i < base)
               else (figstyle.OKABE['green'] if symmetric else figstyle.OKABE['grey']))
        ax.add_patch(Rectangle((o[0], o[1]), o[2] - o[0], o[3] - o[1], color=col, alpha=0.9))
    if symmetric:
        # size-matched frames around the aliased pair (a fixed-size marker did not enclose the
        # varying-size landmark rectangles); a small inflation seats the frame just outside each.
        pad = 0.012
        for idx in (2, base + 2):
            o = obs[idx]
            ax.add_patch(Rectangle((o[0] - pad, o[1] - pad), (o[2] - o[0]) + 2 * pad,
                                   (o[3] - o[1]) + 2 * pad, fill=False, ec='k', lw=1.8, zorder=5))
        # two agent positions half a lap apart facing identical arcs (math prime for A' throughout)
        for a, lab in ((np.pi / 2, 'A'), (-np.pi / 2, "A$'$")):
            px, py = cx0 + CIRCUIT_RADIUS * np.cos(a), cy0 + CIRCUIT_RADIUS * np.sin(a)
            ax.plot(px, py, '^', color='k', ms=10)
            ax.annotate(lab, (px, py), textcoords='offset points', xytext=(6, 6), fontweight='bold')
    ax.set_xlim(-0.05, ROOM_W + 0.05); ax.set_ylim(-0.05, ROOM_H + 0.05)
    ax.set_aspect('equal'); ax.set_title(title, loc='left'); ax.set_xticks([]); ax.set_yticks([])
    figstyle.panel(ax, letter, x=-0.02)
    for s in ax.spines.values():   # the arena Rectangle is the visual border; drop axes spines
        s.set_visible(False)


draw(axes[0], std, 'Standard circuit: unique ring', 'A', symmetric=False)
draw(axes[1], ali, 'Aliasing circuit ($k{=}2$ symmetric)', 'B', symmetric=True)
# Color key only; the caption already explains the A / A' geometry (0.8 m apart, identical arcs).
axes[1].text(0.5, -0.04, "orange = base sector      green = its 180° replica",
             transform=axes[1].transAxes, ha='center', va='top', fontsize=8.2)
axes[0].legend(loc='upper right', bbox_to_anchor=(0.965, 0.965), fontsize=8, framealpha=1.0)
fig.tight_layout()
figstyle.save(fig, __file__, 'snn_slam_aliasing_setup')
plt.close(fig)
print('wrote snn_slam_aliasing_setup.{pdf,png}')
