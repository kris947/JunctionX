import numpy as np
import pydicom
import glob

filenames = glob.glob("../Dataset/386662/MR*.dcm")
filenames.sort(key=lambda x: x[x.find(" ")+1:-4].zfill(4))


res = []
for i in range(len(filenames)):
    ds = pydicom.read_file(filenames[i])
    x = ds.pixel_array #.astype('uint8')
    res.append(x)
    #breakpoint()

res = np.array(res)
res = res.reshape((-1,1,1))

res = (res - np.min(res)) / (np.max(res) - np.min(res) ) 
res = (res * 255).astype('uint8')

print(res.shape)

res.tofile('uint8_file.bin')
