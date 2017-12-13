def regseq(**kwargs):

    from skimage import img_as_float
    import os
    import cv2
    import numpy as np
    import matplotlib
    from matplotlib import pyplot as plt

    def adjust_t(t0, dim):
        if t0> np.fix(dim/2):
            t = t0-dim
        else:
            t = t0
        return t

    def phase_corr_reg(F0, F):
        X = np.fft.ifft2(np.multiply(F0, np.conj(F)))
        max1 = np.amax(X, axis=0)
        argmax1 = np.argmax(X, axis=0)
        max2 = np. amax(max1, axis=0)
        argmax2 = np.argmax(max1, axis=0)
        tx = argmax2
        ty = argmax1[argmax2]
        m,n = F0.shape
        tx = adjust_t(tx, m)
        ty = adjust_t(ty, n)
        tval={'tx':tx, 'ty':ty, 'm':m, 'n':n}
        return tval


    if len(kwargs)<2:
        print('first two arguments must be input and output path')

    if len(kwargs)<3:
        kwargs['opts'] = {'dimX':512, 'dimY':512, 'numChannels':4, 'channel':1, 'templateFrame':0, 'timePts':500, 'dtype':'uint16', 'shape':(15000, 4, 512, 512), 'startFrame':0, 'stopFrame':500, 'debug':True, 'mode':'r', 'order':'C'} 
    
    
    infile = np.memmap(kwargs['inpath'], dtype = kwargs['opts']['dtype'], mode = kwargs['opts']['mode'],  shape = kwargs['opts']['shape'], order = kwargs['opts']['order'])
#    img=preproc(infile[kwargs['opts']['templateFrame'],kwargs['opts']['channel'],:,:])
    tempframe = infile[kwargs['opts']['templateFrame'],kwargs['opts']['channel'],:,:]
    tempframe = img_as_float(tempframe)
    templatefft = np.fft.fft2(tempframe)

    outfile = open(kwargs['outpath'], 'w')
    for frame in range(kwargs['opts']['startFrame'],kwargs['opts']['stopFrame']):
#        img = preproc(infile[frame,kwargs['opts']['channel'],:,:])
        img = infile[frame,kwargs['opts']['channel'],:,:]
        img = img_as_float(img)
        tval = phase_corr_reg(templatefft, np.fft.fft2(img))
        print('frame = ' + str(frame) + ', tx = ' + str(tval['tx']) + ', ty = ' + str(tval['ty']))
        M =np.float64([[1, 0, tval['tx']],[0, 1, tval['ty']]])
        new_img = cv2.warpAffine(img,M,(tval['m'],tval['n']))
#        plt.imshow(new_img)
#        plt.show()
#        outfile.write(postproc(new_img))
        outfile.write(new_img)
    outfile.close()

kwargs = {'inpath':'/home/zlazri/Documents/spontdata/Image_0001_0001.raw', 'outpath':'/home/zlazri/Documents/spontdata/Aligned_Image_0001_0001.raw'}
regseq(**kwargs)
