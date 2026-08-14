#!/usr/bin/env python3
"""
make_architecture.py -- Figure 1: neuromorphic spiking SLAM architecture.

Round-4 redesign: Figure 1 is now authored as a hand-crafted VECTOR MASTER,
``snn_slam_architecture.svg`` (committed next to this script). This script's job
is to *integrate* that master into the build -- i.e. render it to the publication
assets the manuscript consumes (a vector PDF, preferred, plus a 300+ dpi PNG) and
copy them into the 2D manuscript's ``figures/`` directory.

Why a vector master instead of drawing with matplotlib?  The schematic is
illustrative (arena + sensing, MAV sensor suite, and the spiking-SLAM pipeline
block diagram), so it is authored directly in SVG for full typographic and
layout control, then rendered here. Every block still corresponds to something
real in the code (verified against snn_slam_system.py, snn_place_cells.py,
sparse_forest.py, scripts/slam_variance.py):

  (a) 2x2 m walled arena, sparse obstacles, revisit-rich circuit, 90-deg DVS FoV
      wedge, and the three ToF beams cast to the nearest wall/obstacle.
  (b) MAV sensor suite on a schematic insect-scale body: event camera (DVS,
      256 px / 90 deg), 3-beam ToF rangefinder (0.1-2.83 m), 6-axis IMU.
  (c) Parallel spiking pathways -> 2D place-cell network + topological map;
      spiking complementary filter extracts gravity pitch and injects a Gaussian
      current INTO the ring attractor (CANN); appearance-keyed sparse-hash loop closure
      with a geometric pose gate and pose-graph relaxation.

Rendering backends are tried in order of preference; the first available wins:
  1. cairosvg           (pip install cairosvg)        -> PDF + PNG, best fidelity
  2. rsvg-convert       (librsvg CLI)                  -> PDF + PNG
  3. inkscape           (Inkscape >= 1.0 CLI)          -> PDF + PNG
  4. svglib + reportlab (pip install svglib reportlab) -> PDF (PNG stays committed)
If none is available the script says how to install one and falls back to the
committed high-resolution PNG so the build still succeeds.
"""
import os
import shutil
import subprocess
import sys
from shutil import which

HERE = os.path.dirname(os.path.abspath(__file__))
BASENAME = 'snn_slam_architecture'
SVG = os.path.join(HERE, f'{BASENAME}.svg')
PNG = os.path.join(HERE, f'{BASENAME}.png')
PDF = os.path.join(HERE, f'{BASENAME}.pdf')
MAN = os.path.normpath(os.path.join(HERE, '..', '..', '..',
                                    'manuscript', 'figures'))
PNG_WIDTH = 2660   # px, ~2x the 1330-unit viewBox  (~410 dpi at 6.5 in wide)


def _try_cairosvg():
    try:
        import cairosvg
    except Exception:
        return False
    cairosvg.svg2pdf(url=SVG, write_to=PDF)
    cairosvg.svg2png(url=SVG, write_to=PNG, output_width=PNG_WIDTH)
    print('  rendered with cairosvg  -> PDF + PNG')
    return True


def _try_rsvg():
    exe = which('rsvg-convert')
    if not exe:
        return False
    subprocess.run([exe, '-f', 'pdf', '-o', PDF, SVG], check=True)
    subprocess.run([exe, '-w', str(PNG_WIDTH), '-o', PNG, SVG], check=True)
    print('  rendered with rsvg-convert  -> PDF + PNG')
    return True


def _try_inkscape():
    exe = which('inkscape')
    if not exe:
        return False
    subprocess.run([exe, SVG, '--export-type=pdf',
                    f'--export-filename={PDF}'], check=True)
    subprocess.run([exe, SVG, '--export-type=png', f'--export-filename={PNG}',
                    f'--export-width={PNG_WIDTH}'], check=True)
    print('  rendered with inkscape  -> PDF + PNG')
    return True


def _try_svglib():
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
    except Exception:
        return False
    renderPDF.drawToFile(svg2rlg(SVG), PDF)
    print('  rendered with svglib+reportlab  -> PDF (PNG left as the committed raster)')
    return True


def _copy_to_manuscript():
    if not os.path.isdir(MAN):
        print(f'  (manuscript figures dir not found: {MAN} -- skipping copy)')
        return
    for ext in ('pdf', 'png'):
        src = os.path.join(HERE, f'{BASENAME}.{ext}')
        if os.path.exists(src):
            dst = os.path.join(MAN, f'{BASENAME}.{ext}')
            shutil.copyfile(src, dst)
            print(f'  copied -> {dst}')


def main():
    if not os.path.exists(SVG):
        sys.exit(f'ERROR: vector master not found: {SVG}')
    ok = _try_cairosvg() or _try_rsvg() or _try_inkscape() or _try_svglib()
    if not ok:
        print('  NOTE: no SVG->PDF backend found '
              '(tried cairosvg, rsvg-convert, inkscape, svglib).')
        print('        For a vector PDF install one, e.g.:  pip install cairosvg')
        print('        Falling back to the committed PNG raster.')
    _copy_to_manuscript()
    print('Fig 1 architecture integrated from snn_slam_architecture.svg.')


if __name__ == '__main__':
    main()
