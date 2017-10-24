# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import convolve
from scipy.spatial.distance import euclidean, pdist
from matplotlib import animation
from glob import glob
from time import strftime, time
from matplotlib import cm
from operator import itemgetter
import matplotlib.pyplot as plt

import scipy
#from JSAnimation import IPython_display
#from IPython.display import display
from pickle import load, dump

import matplotlib
matplotlib.rc('animation', ffmpeg_path='./ffmpeg/ffmpeg', writer='ffmpeg')

class Struct(object):
    pass

def save(radars):
    with open('temp.pickle', 'w') as temp:
        dump(radars, temp)

def load():
    with open('temp.pickle', 'r') as temp:
        radars = load(temp)
    return radars

def log_time(msg):
    print msg + ': ' + strftime('%M:%S') + '{0:.2f}'.format(time() % 1).lstrip('0')

def window_avg(data, window):
    return convolve(data, np.ones(window), 'same')

def to_m(bins):
    return bins * 0.00914366997

def to_bins(m):
    return m / to_m(1)

def waterfall(frames):
    plt.figure()
    plt.imshow(frames.T, origin='lower', interpolation='none', aspect='auto', vmin=0,
               cmap=cm.jet, extent=[0, len(frames), 0, len(frames[0])])
    plt.show()

def waterfall_all(radars):
    for i in range(len(radars.frames)):
        print radars.names[i]
        waterfall(radars.frames[i])

def one_dimension(test_radar=0, test_frame=50, frame=None):
    if frame is None:
        frame = radars.frames[test_radar, test_frame]
    plt.figure()
    plt.bar(np.arange(radars.max_dist), frame, linewidth=0, width=1, color='black')
    plt.xlim(0, radars.max_dist)
    plt.show()

def from_files(radars, directory):
    radars.names = sorted(glob(directory + '/*.csv'))
    radars.timestamps = [0] * len(radars.names)
    radars.raw = [0] * len(radars.names)
    for i, file_name in enumerate(radars.names):
        radars.raw[i] = []
        radars.timestamps[i] = []
        with open(file_name, 'r') as file_obj:
            for line in file_obj:
                row = line.split(', ')
                if len(row) > 16 and row[12] == '4' and row[1] == 'MrmFullScanInfo':
                    #radars.raw[i].append(map(np.float64, row[16:])) # slow
                    radars.raw[i].append(np.array(row[16:], dtype=np.float32))
                    radars.timestamps[i].append(float(row[0]))
    #mint = min(map(len, radars.raw))
    #for i in range(len(radars.names)):
    #    radars.raw[i] = radars.raw[i][:mint]
    #    radars.timestamps[i] = radars.timestamps[i][:mint]
    radars.raw = [x[:min(map(len, radars.raw))] for x in radars.raw]
    assert all([len(radars.raw[0]) == x for x in map(len, radars.raw)])
    return radars

def gen_info(radars,offset=0, pow1=0.5, pow2=0.1, distance_pow=0, window=30, threshold=80): #offset=-120
    radars.window = window
    radars.pow1 = pow1
    radars.pow2 = pow2
    radars.distance_pow = distance_pow
    radars.threshold = threshold
    radars.num_radars = len(radars.raw)
    radars.time_extent = len(radars.raw[0])
    radars.max_dist = len(radars.raw[0][0]) - offset
    radars.side = to_bins(10.0)  # hardcoded input
    r = radars.side
    radars.positions = np.array([(0, 0), (r, 0), (r, r), (0, r)]) # more hardcoded input
    radars.field = np.array([0, radars.side] * 2)
    if offset > 0:
        radars.nraw = np.array([[b[offset:] for b in a] for a in radars.raw])
    elif offset < 0:
        radars.nraw = np.dstack((np.zeros((radars.num_radars, radars.time_extent, -offset)), np.array(radars.raw)))
    else:
        radars.nraw = np.array(radars.raw)
    assert len(radars.nraw[0, 0]) == radars.max_dist

def test_limit(p, test_radar, test_frames):
    radars.threshold = p
    res = 10
    frames = np.apply_along_axis(window_avg, 2, np.fabs(radars.nraw), radars.window)
    limit = np.percentile(frames, radars.threshold, axis=2)[test_radar, test_frames]
    plt.bar(np.arange(0, radars.max_dist, res), frames[test_radar, test_frames, ::res] - limit, color='black', width=res)
    plt.xlim(0, radars.max_dist)
    plt.show()

def frame_combine(radars, idx, res):
    Z = np.empty((radars.num_radars, radars.side / res, radars.side / res), dtype=np.float32)
    for radar in range(radars.num_radars):
        data = radars.frames[radar, idx]
        x0, y0 = radars.positions[radar]
        for i, j in np.ndindex(Z.shape[1:]):
            Z[radar, i, j] = data[min(int(np.sqrt((i * res - x0)**2 + (j * res - y0)**2)), len(data) - 1)]
    P = np.cumprod(Z, axis=0)[-1] * np.amin(Z, axis=0)**radars.pow2
    P /= P.max() # normalize
    P[np.isnan(P)] = 0
    return P

def frame_overlay(radars, idx, res):
    Z = np.empty((radars.num_radars, radars.side / res, radars.side / res), dtype=np.float32)
    for radar in range(len(Z)):
        data = radars.frames[radar, idx]
        x0, y0 = radars.positions[radar]
        for i, j in np.ndindex(Z.shape[1:]):
            Z[radar, i, j] = data[min(int(np.sqrt((i * res - x0)**2 + (j * res - y0)**2)), len(data) - 1)]
    P = np.cumsum(Z, axis=0)[-1]
    return P

def show_frame_combine(radars, idx, dres):
    return plt.imshow(frame_combine(radars, idx, dres) * 2, origin='lower', interpolation='none',
                      extent=to_m(radars.field), vmin=0, vmax=1)

def show_frame_overlay(radars, idx, dres):
    return plt.imshow(frame_overlay(radars, idx, dres), origin='lower', interpolation='none',
                      extent=to_m(radars.field), vmax=4)

def show_frames_combine(radars, dres, tres, fps=24):
    fig = plt.figure()
    heatmap = show_frame_combine(radars, 0, dres)
    
    def update(i):
        heatmap.set_array(frame_combine(radars, i * tres, dres))
        return heatmap,
    
    return animation.FuncAnimation(fig, update, frames=radars.time_extent / tres, interval=int(1000 / fps), blit=True)

def show_frames_overlay(radars, dres, tres, fps=24):
    fig = plt.figure()
    heatmap = show_frame_overlay(radars, 0, dres)
    
    def update(i):
        heatmap.set_array(frame_overlay(radars, i * tres, dres))
        return heatmap,
    
    return animation.FuncAnimation(fig, update, frames=radars.time_extent / tres, interval=int(1000 / fps), blit=True)

def gen_frames(radars):
    # raw data
    radars.frames = radars.nraw.copy()
    
    # absolute value
    radars.frames = np.fabs(radars.frames) / 10000
    
    # subtract noise and make positive
    #radars.frames[:, :, :] = radars.frames[:, :, :] - radars.frames.std(axis=2)[:, :, np.newaxis] * 10
    #radars.frames[:, :, :] -= radars.frames[:, radars.silence, :].max(axis=2).max(axis=1)[:, np.newaxis, np.newaxis]
    radars.frames[:, :, :] -= np.percentile(radars.frames, radars.threshold, axis=2)[:, :, np.newaxis]
    radars.frames = np.maximum(radars.frames, 0)

    # distance correction
    radars.frames *= (np.arange(radars.max_dist))**radars.distance_pow
    
    # windowed average
    radars.frames = np.apply_along_axis(window_avg, 2, radars.frames, radars.window)
    
    # dull the points
    radars.frames **= radars.pow1
    
    # enfore the lowest maximum
    #radars.frames[:, radars.frames.max(axis=2) == 0, 0] = 0.01
    
    # normalize each frame to highest point
    radars.frames /= radars.frames.max(axis=2)[:, :, np.newaxis]
    assert len(radars.frames[0, 0]) == radars.max_dist


def main(data, output_file, offset=0, pow1=0.5, pow2=0.1, distance_pow=0.0, window=30, threshold=80, fps=5, dres=5, tres=1):
    log_time('Start')
    radars = Struct()
    from_files(radars, data)
    gen_info(radars, offset=offset, pow1=pow1, pow2=pow2, distance_pow=distance_pow, window=window, threshold=threshold)
    gen_frames(radars)
    anim = show_frames_combine(radars, dres, tres, fps=fps)
    log_time('Video in RAM')
    anim.save(output_file)
    #anim.save(output_file, writer=animation.FFMpegWriter())
    log_time('Done')

main('New Data 1', 'New Data 1.avi')
