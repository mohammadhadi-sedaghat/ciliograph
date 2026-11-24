# Ciliograph

Ciliograph turns a high-speed video of beating cilia into a digital beat
pattern: a smooth, periodic description of one cilium's shape over its beat
cycle, fitted directly to the frames. It also measures the ciliary beat
frequency (CBF) and the cilium length, exports the cilium's node coordinates
at every time step, and tracks particles to measure transport.

Everything is in one Python file, ciliograph.py, with an interactive window
and a command line. It runs on Windows, Linux and macOS, and can be built into
a Windows program that needs no Python.

## Features

- Import a folder of TIFF, PNG or JPEG frames; set the scale (pixels per
  µm) and frame rate; level and crop the frames.
- Enhance the frames for the measurements: background removal, lighting
  flattening, denoising, deflickering, deblurring, local contrast, filters for
  thin structures, or a neural network of your own (ONNX).
- Find the cilium base, one beat cycle with its CBF, and the cilium length
  automatically.
- Auto fit the beat pattern to the frames: a Fourier series, with the
  cilium's length held constant at every phase.
- Export the x–y coordinates of the cilium's nodes over the beat.
- Track particles (MCC tab) and measure their path, speed and velocity.
- Validate on rendered carpets with a known beat, and run a built-in
  self-test.

## Installation

Python 3.9 or newer:

```bash
pip install numpy matplotlib pillow
```

Optional: onnxruntime (AI models on the enhance tab), cupy-cuda12x (an
NVIDIA GPU for the tracer), scipy. The window uses Tk, which comes with
Python on Windows and macOS; on Linux install python3-tk.

### Windows program without Python

Run build_standalone.bat once on a PC that has Python and an internet
connection. It builds dist\Ciliograph\Ciliograph.exe and
dist\Ciliograph_windows.zip in a private build environment (your own Python
is not touched) and runs the self-test with the new program. Unzip it anywhere
and double-click Ciliograph.exe. build_standalone onefile makes a single
.exe; build_standalone noai leaves out the AI-model support.

## Quick start

```bash
python ciliograph.py                        # open the window, then Import images...
python ciliograph.py --images path/to/frames
```

Work left to right along the tabs:

| Tab | What to do |
|---|---|
| video | Import images…, then type ppum (pixels per µm) and fps. Level rotates the frames until the line of cilia bases is horizontal; Crop limits the analysis to a region. Bytescaling, gamma, sharpness and brightness change only the view, not the measurements. |
| enhance | Optional clean-up of the frames for the measurements. Auto tries the settings and keeps the one that gives the most consistent pattern. |
| position | Find base places the model cilium on the one to measure; the sliders fine-tune the base and its orientation. frame draws the pattern upright (a symmetric fan) or relative to the cell surface. |
| cycle | Find cycle finds one beat around the base and its CBF. effective stroke sets which way the effective stroke sweeps. |
| length | Find length estimates the cilium length, or Click tip sets it by hand. Also the number of nodes and time steps to export. |
| params | Auto fit fits the beat to the frames (about a minute); pressing it again polishes the current beat, Fresh fit starts over. Export nodes writes the coordinates. |
| MCC | Pick particle, click on it, then Track and Export: path length, speed and velocity of a particle carried by the cilia. |

Save writes everything to ciliograph.json beside the images; nothing is
written to disk until you press it. Help opens a guide to every part of
the window and the keyboard shortcuts. Auto fit's detailed readout is printed
in the terminal window.

## Outputs

| File | Written by | Contents |
|---|---|---|
| ciliograph.json | Save | all settings and the fitted beat's Fourier coefficients |
| output/&lt;name&gt;_z01a.data | Export nodes | x and y of every node at every time step, one column pair per step; the first pair is the cycle start |
| output/&lt;name&gt;_cilium_shapes.csv | Export nodes | step, phase, time, arc length, x, y |
| output/&lt;name&gt;_cilium_shapes.npz | Export nodes | the same as NumPy arrays |
| output/particle_trajectory.csv | MCC Export | time, position (px and µm), step length and instantaneous speed in every frame |
| output/particle_velocity.png | MCC Export | the trajectory coloured by time, and the speed over time |

output/ is created in the image folder; &lt;name&gt; is the recording's name.

## The beat model

```
R(s, t) = L · Σn Σm ( Anm cos(2πnt/T) + Bnm sin(2πnt/T) ) · s^m
```

with 8 Fourier modes in time (n = 0…7) and a cubic in the position s along
the cilium (m = 1…3), for the coordinate x along the cell surface and z
along its normal. The fit starts from a published beat pattern and

- holds the cilium at the same length L at every phase,
- allows one stroke each way (no back-and-forth within a stroke),
- starts the cycle at the start of the recovery stroke (the cilium's sideways
  extreme).

Auto fit builds three candidates — a beat through frame-by-frame traces of the
cilium, one through phase-averaged frames, and one fitted to all frames
jointly — and keeps the one that follows the cilium best in every frame.

## Command line

```bash
python ciliograph.py <command> --help
```

| Command | What it does |
|---|---|
| fit | open the window (the default) |
| shapes | export the node coordinates and figures from a ciliograph.json |
| cycle, locate, length, level | find the beat cycle, the base, the length, the rotation |
| track | follow one cilium through every frame |
| beatfit | fit a beat pattern to a traced centreline file |
| beat | report how periodic the beat is, and where |
| synth | render a carpet of cilia with a known beat |
| validate | run the steps on such a carpet and score them against the truth |
| info | summarise a ciliograph.json |
| check | read every frame and report any that fail |
| selftest | check the model, the file format and the fitters (no data needed) |

## Checking the results

synth renders a carpet of cilia with a known beat, and validate runs Find
base, Find cycle, Find length and Auto fit on it and scores each against the
truth. Use it to judge an enhancement setting or an AI model before trusting
it on your own recordings:

```bash
python ciliograph.py synth --out synthetic/lawn --ppum 12 --cbf 15 --fps 300 --frames 120 \
    --cilia 40 --spacing 0.5 --wavelength 100000 --blur 1.8 --noise 0.03 \
    --jitter 0.03 --scatter 1 --bright 0.5 --body 0.35
python ciliograph.py validate --images synthetic/lawn --assume-period
python ciliograph.py validate --images synthetic/lawn --assume-period --model path/to/model.onnx
```

## AI models (optional)

Any image-to-image network in ONNX format can be loaded on the enhance tab
(needs onnxruntime). Every later step then reads the network's output, so
judge a network with validate as above, not by how the frames look.

On Windows, if importing onnxruntime makes Python quit silently (common with an
Anaconda installed for all users), make an environment of your own and start
Ciliograph from it:

```bat
conda create -n cilio --override-channels -c conda-forge python=3.12 numpy scipy matplotlib-base pillow onnxruntime
conda env config vars set -n cilio PYTHONNOUSERSITE=1
conda activate cilio
```

### Training a cilia network

train_cilia_net.py trains a small network that draws the cilia in a frame
and nothing else. It learns from carpets rendered with a known answer, and
frames you trace by hand can be added. It needs PyTorch, kept in an
environment of its own:

```bat
conda create -n train --override-channels -c conda-forge python=3.12 numpy pillow pytorch-cpu onnx onnxruntime
conda activate train
python train_cilia_net.py render --out training
python train_cilia_net.py check --data training
python train_cilia_net.py train --data training --out models\cilia_net.onnx
```

Then load models\cilia_net.onnx on the enhance tab, after setting the base
and the length on the raw frames.

## Speed

Find cycle, Auto fit, Auto enhance and the tracer use every CPU core, and an
NVIDIA GPU as well when CuPy is installed; the results are the same as on one
core. --workers N or --device cpu changes this, and
python ciliograph.py info shows what is in use.

## Limitations

- Where neighbouring cilia overlap, no single cilium can be separated: the
  fitted pattern then follows the texture of the bundle at that spot, and Auto
  fit's readout says so. The readout also reports how far the runner-up beats
  lie from the winner, a measure of how well the frames determine the beat.
- The quality of the frames sets the quality of the pattern: a high frame
  rate, high magnification and uncompressed frames help most. Enhancement and
  AI models cannot add detail the frames do not contain.

## Files

| File | Purpose |
|---|---|
| ciliograph.py | the program |
| train_cilia_net.py | optional: train a network that draws the cilia |
| build_standalone.bat | build the Windows program |
| aa.bat, aa-shapes.bat (Windows), aa, aa-shapes (Linux/macOS) | shortcuts for the window and for shapes |
