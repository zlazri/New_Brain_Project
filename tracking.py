#!/usr/bin/env python3

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np

import scipy.interpolate
import scipy.optimize

import skimage.feature
import skimage.filters
import skimage.measure
import skimage.restoration
import skimage.transform

def adjust_t(t, d):
    return t - 1 - (d if t > np.fix(d/2) else 0)
    
def phase_corr_reg(F0, F1):
    X = np.fft.ifft2(np.multiply(F0, np.conj(F1)))
    max1 = np.amax(X, axis=0)
    argmax1 = np.argmax(X, axis=0)
    max2 = np. amax(max1, axis=0)
    argmax2 = np.argmax(max1, axis=0)
    tx = argmax2
    ty = argmax1[argmax2]
    m, n = F0.shape
    return adjust_t(ty, m), adjust_t(tx, n)

def get_mmap_stack(path, nframes=15000):
    shape = (15000, 4, 512, 512)
    return np.memmap(path, dtype=np.uint16, mode='r', shape=shape, order='C')

def get_image(m, index, channel=1):
    return np.array(m[index, channel, :, :], np.double)

def normalize(img):
    return img/np.max(img)

def get_average_image(m, i0, i1, channel=1):
    return np.average(m[i0:i1, channel, :, :], 0)

def denoise(img):
    return skimage.restoration.denoise_tv_chambolle(img, weight=0.2)

def blob_detect(img):
    blobs = skimage.feature.blob_log(
        normalize(denoise(img)),
        min_sigma=5,
        max_sigma=12,
        num_sigma=71,
        threshold=0.1)
    assert(len(blobs) > 0)
    return blobs

def make_blob_plot(img, t, blobs, pairs):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    num_blobs = blobs[t - 1].shape[0]
    for i in range(num_blobs):
        y0, x0, sigma = blobs[t - 1][i, :]
        r = np.sqrt(2)*sigma
        c = plt.Circle((x0, y0), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)
        if i in pairs[t]:
            j = pairs[t][i]
            y1, x1, _ = blobs[t][j, :]
            dx = x1 - x0
            dy = y1 - y0
            a = plt.Arrow(x0, y0, dx, dy)
            ax.add_patch(a)
    return fig

def get_xy(blobs):
    return blobs[:, 0:2]

def get_dist_matrix(xy1, xy2):

    '''Return a len(xy1) x len(xy2) distance matrix where the (i, j)th
entry contains the distance between xy1[i, :] and xy2[j, :].'''
    m1, m2 = (xy1.shape[0], xy2.shape[0])
    dist = np.zeros((m1, m2))
    for i, j in itertools.product(range(m1), range(m2)):
        dist[i, j] = np.linalg.norm(xy1[i, :] - xy2[j, :])
    return dist

def get_pairs(dist, max_radius=15):
    m1, m2 = dist.shape

    js = set()

    # Construct initial index mapping
    pairs = []
    if m1 < m2:
        for i in range(m1):
            j = np.argmin(dist[i, :])
            js.add(j)
            d = dist[i, j]
            pairs.append((i, j, d))
    else:
        for j in range(m2):
            i = np.argmin(dist[:, j])
            js.add(j)
            d = dist[i, j]
            pairs.append((i, j, d))

    # Remove pairs which map different i onto the same j. Keep the
    # pair which has the lowest distance (this is a heuristic and
    # isn't necessarily correct)
    tmp = []
    for j in js:
        tmp_ = []
        for k, pair in enumerate(pairs):
            if j == pair[1]:
                tmp_.append(pair)
        tmp_ = sorted(tmp_, key=lambda pair: pair[2])
        tmp.append(tmp_[0])

    # Compute a threshold value to prune index pairs which are
    # probably incorrect
    ds = sorted([d for (_, _, d) in tmp])
#    dmax = 3*ds[int(len(ds)/2)] # this is totally arbitrary...
    dmax = 3*ds[3*int(len(ds)/4)]

    return {i: j for (i, j, d) in tmp if d <= dmax}

def parse_args():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('path', type=str)
    parser.add_argument('t0', type=int)
    parser.add_argument('nframes', type=int)
    parser.add_argument('--plot', nargs='*')
    parser.add_argument('--plot_path', type=str, default='plots')
    return parser.parse_args()

class ParticleSequence():
    def __init__(self, blobs, pairs):
        self.seqs = []
        self.add_seqs(blobs, pairs)
        self.add_offsets()
    
    def add_seqs(self, blobs, pairs):
        assert(len(self.seqs) == 0)
        assert(len(blobs) == len(pairs) + 1)

        # Get the minimum and maximum time points
        t0 = min(blobs.keys())
        t1 = max(blobs.keys())

        # The first frame is just the first set of detected blobs
        xy = get_xy(blobs[t0])
        for i in range(xy.shape[0]):
            self.seqs.append([(xy[i, 0], xy[i, 1])])

        # We need to build a permutation which takes us all the way
        # back from the current pair to the original sequence index
        seq_perm = {j: i for i, j in pairs[t0 + 1].items()}
        
        B = dict()
        
        for t in range(t0 + 1, t1 + 1):
            nseqs = len(self.seqs)

            # Grab the (x, y)-coordinates of the blobs, add them to
            # the list of blob coordinates using the inverse
            # permutation
            xy = get_xy(blobs[t])
            idx_xy = set(range(len(xy[:,0])))
            p = pairs[t]
            not_updated = set(range(nseqs))
            
            for i in p.keys():
                j = p[i]
                if j in seq_perm:
                    i0 = seq_perm[j]
                    self.seqs[i0].append((xy[j, 0], xy[j, 1]))
                    not_updated.remove(i0)
                    idx_xy.remove(j)
            tmp = B

            for i in idx_xy:
                # If new index couldn't isn't in pairs,
                # try linking it to broken sequence
                
                blob_maybe =  (xy[i, 0], xy[i, 1])      # blob_maybe = blob that possibly fits a broken sequence
                blob_maybe_ = np.asarray(blob_maybe)    # blob_check = blob to which we compare blob_maybe to see
                added_check = False                     # if they are part of the same sequence.
                
                for i0 in B.keys():
                    blob_check = np.asarray(self.seqs[i0][B[i0]])         
                    dist = np.linalg.norm(blob_maybe_[:]-blob_check[:])
                    dmax = 10
                                    
                    if dist <= dmax:                         # If the blobs are close enough, add blob_maybe to
                        self.seqs[i0].append(blob_maybe)     # blob_check's sequence and remove it from the set
                        tmp.pop(i0)                          # of broken sequences/
                        not_updated.remove(i0)
                        seq_perm[i] = i0
                        added_check = True
                        break

                if added_check == False:                              # If blob_maybe isn't added to any broken sequences,
                    tmp[nseqs]= t                                     # it's a new blob. Thus, create a new sequence.
                    for n in range(1, t+1):
                        if n == 1:
                            self.seqs.append([(np.nan, np.nan)])
                        else:
                            self.seqs[nseqs].append((np.nan, np.nan))
                    self.seqs[nseqs].append(blob_maybe)
                    nseqs = nseqs + 1
            B = tmp        
            for i0 in not_updated:                       # Add NaNs in for broken sequences and
                self.seqs[i0].append((np.nan, np.nan))   # update the broken set.
                if i0 not in B.keys():
                    B[i0] = t-1
                
                 
            # Update the inverse permutation (the permutation which
            # goes from the current blob indexing to the indexing of
            # the sequence of blobs that we're updating)
            if t < t1:
                tmp = dict()
                p = pairs[t + 1]
                not_updated = set(seq_perm.keys())
                for j in seq_perm:
                    if j in p:
                        tmp[p[j]] = seq_perm[j]
                        not_updated.remove(j)
                seq_perm = tmp

            # TODO: we still need to account for indexes that we miss (done)
            # TODO: also need to insert NaNs and connect blobs back to
            # earlier ones if we start detecting them again (done)

    def add_offsets(self):
        self.offsets = []
        for seq in self.seqs:
            x0, y0 = seq[0]
            self.offsets.append([(x - x0, y - y0) for x, y in seq[1:]])

class TpsXyInterp():
    def __init__(self, x, y, dx, dy):
        tps = lambda d: scipy.interpolate.Rbf(x, y, d, function='thin_plate')
        self._tps_dx = tps(dx)
        self._tps_dy = tps(dy)

    def dx(self, x, y):
        return self._tps_dx(x, y)

    def dy(self, x, y):
        return self._tps_dy(x, y)

if __name__ == '__main__':
    args = parse_args()
    plots = args.plot if args.plot else []

    mmap = get_mmap_stack(args.path)

    if 'denoised' in plots:
        plt.figure()
        plt.imshow(denoise(normalize(get_image(mmap, args.t0))))
        plt.savefig('denoised.png')
        plt.close()

    print('Running blob detection')
    blobs = dict()
    for t in range(args.t0, args.t0 + args.nframes):
        print('- t = %d' % t)
        img = normalize(get_image(mmap, t))
        blobs[t] = blob_detect(img)

    print('Connecting blobs')
    pairs = dict()
    for t in range(args.t0 + 1, args.t0 + args.nframes):
        print('- t = %d' % t)
        xy1 = get_xy(blobs[t - 1])
        xy2 = get_xy(blobs[t])
        D = get_dist_matrix(xy1, xy2)
        pairs[t] = get_pairs(D)

    # Compute base translation offsets using phase correlation
    # registration
    print('Computing offsets using phase correlation')
    phcorr_offsets = dict()
    F0 = np.fft.fft2(get_image(mmap, args.t0))
    for t in range(args.t0 + 1, args.t0 + args.nframes):
        print('- t = %d' % t)
        F1 = np.fft.fft2(get_image(mmap, t))
        phcorr_offsets[t] = phase_corr_reg(F0, F1)

    # Create tracked sequences of blobs using the `blobs' and `pairs'
    # lists that were just created
    seq = ParticleSequence(blobs, pairs)

    # print(phcorr_offsets)
    # t0 = args.t0 + 1
    # for i, offsets in enumerate(seq.offsets):
    #     print('offsets %d' % i)
    #     for j, (x, y) in enumerate(offsets):
    #         dx, dy = phcorr_offsets[t0 + j]
    #         print(((x, y), (x + dx, y + dy)))

    # Get indices of full sequences (i.e. sequences where no blobs
    # were lost and which have the maximum number of frames)
    full_seq_is = [
        i for (i, seq) in enumerate(seq.seqs) if len(seq) == args.nframes]

    for i in full_seq_is:
        print(seq.seqs[i])

    # Construct thin plate spline interpolants for the (dx, dy) values
    # computed between each frame
    tps = dict()
    for t in range(args.t0 + 1, args.t0 + args.nframes):
        xy1 = get_xy(blobs[t - 1])
        xy2 = get_xy(blobs[t])
        p = pairs[t]
        rows = []
        for i, j in p.items():
            x, y = (xy1[i, 0], xy1[i, 1])
            dxy = xy2[j, :] - xy1[i, :]
            rows.append((x, y, dxy[0], dxy[1]))
        arr = np.array(rows)
        tps[t] = TpsXyInterp(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])

    # Optionally plot vector fields of (dx, dy) TPS estimates
    if 'tps' in plots:
        print('Saving thin plate spline plots')
        L = np.linspace(0, 512, 40);
        I, J = np.meshgrid(L, L)
        for t in range(args.t0 + 1, args.t0 + args.nframes):
            fig, ax = plt.subplots(1)
            U, V = (tps[t].dx(I, J), tps[t].dy(I, J))
            ax.quiver(U, V)
            fig.savefig('%s/tps_%d.png' % (args.plot_path, t))
            plt.close()

    # Optionally make plots of detected blobs and their offset vectors
    # (depicted using arrows)
    if 'blobs' in plots:
        print('Saving blob plots')
        for t in range(args.t0 + 1, args.t0 + args.nframes):
            print('- t = %d' % t)
            img = get_image(mmap, t)
            p = make_blob_plot(img, t, blobs, pairs)
            p.savefig('%s/blobs_%d.png' % (args.plot_path, t))

    # Optionally make plots of (x, y) & (dx, dy) trajectories of blobs
    # (for each blob we were able to track for all frames)
    if 'xy' in plots:
        print('Saving xy plots')
        T = np.arange(args.t0, args.t0 + args.nframes)
        for i, lst in enumerate(seq.seqs):
            fig, ax = plt.subplots(3)
            arr = np.array(lst)
            m = arr.shape[0]
            if m == args.nframes:
                print('- blob %d' % i)
                x = arr[:, 0]
                y = arr[:, 1]
                ax[0].plot(T, x)
                ax[0].set_title('x')
                ax[1].plot(T, y)
                ax[1].set_title('y')
                dx = x[1:] - x[:m-1]
                dy = y[1:] - y[:m-1]
                ax[2].plot(T[:m-1], dx)
                ax[2].plot(T[:m-1], dy)
                ax[2].set_title('dx, dy')
                fig.savefig('%s/xy_%d.png' % (args.plot_path, i))
                plt.close()
