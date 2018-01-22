import os
import numpy as np
from PIL import Image

img_path = '../../datasets/salicon/Trn'
lbl_path = '../../datasets/salicon/Trn_Map'
if __name__ == '__main__':
	flist = []
	cnt = 0
	for _,_,f in os.walk(img_path):
		flist.extend(f)
	for f in flist:
		img = np.array(Image.open(os.path.join(img_path,f)))
		if len(img.shape) == 2:
			print 'deleting %sth error img %s'%(cnt+1,f)
			cnt += 1
			os.remove(os.path.join(img_path, f))
			os.remove(os.path.join(lbl_path, f.split('.')[0]+'.jpeg'))
