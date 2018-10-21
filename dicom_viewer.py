import pydicom
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import pandas as pd

PACIENS = '385185'

def nothing(x):
    pass

def get_pcoord_from_rcoord(ds, coord):
	x,y,z = coord
	rcoord = [np.int32(x/ds.PixelSpacing[0])+ds.ImagePositionPatient[0]+ds.Rows/2, ds.Columns/2+np.int32(y/ds.PixelSpacing[1])+ds.ImagePositionPatient[1]]
	print(rcoord)
	return rcoord

def get_config():
	df=pd.read_csv("Measurement.csv",sep=";")
	return df

def anisodiff(img,niter=5,kappa=50,gamma=0.1,step=(1.,1.),option=2,ploton=False):
        """
        Anisotropic diffusion.
 
        Usage:
        imgout = anisodiff(im, niter, kappa, gamma, option)
 
        Arguments:
                img    - input image
                niter  - number of iterations
                kappa  - conduction coefficient 20-100 ?
                gamma  - max value of .25 for stability
                step   - tuple, the distance between adjacent pixels in (y,x)
                option - 1 Perona Malik diffusion equation No 1
                         2 Perona Malik diffusion equation No 2
                ploton - if True, the image will be plotted on every iteration
 
        Returns:
                imgout   - diffused image.
 
        kappa controls conduction as a function of gradient.  If kappa is low
        small intensity gradients are able to block conduction and hence diffusion
        across step edges.  A large value reduces the influence of intensity
        gradients on conduction.
 
        gamma controls speed of diffusion (you usually want it at a maximum of
        0.25)
 
        step is used to scale the gradients in case the spacing between adjacent
        pixels differs in the x and y axes
 
        Diffusion equation 1 favours high contrast edges over low contrast ones.
        Diffusion equation 2 favours wide regions over smaller ones.
		"""
        
        # ...you could always diffuse each color channel independently if you
        # really want
        if img.ndim == 3:
                warnings.warn("Only grayscale images allowed, converting to 2D matrix")
                img = img.mean(2)
 
        # initialize output array
        img = img.astype('float32')
        imgout = img.copy()
 
        # initialize some internal variables
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(imgout)
        gE = gS.copy()
 
        # create the plot figure, if requested
        if ploton:
                import pylab as pl
                from time import sleep
 
                fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
                ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
 
                ax1.imshow(img,interpolation='nearest')
                ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
                ax1.set_title("Original image")
                ax2.set_title("Iteration 0")
 
                fig.canvas.draw()
 
        for ii in range(niter):
 
                # calculate the diffs
                deltaS[:-1,: ] = np.diff(imgout,axis=0)
                deltaE[: ,:-1] = np.diff(imgout,axis=1)
 
                # conduction gradients (only need to compute one per dim!)
                if option == 1:
                        gS = np.exp(-(deltaS/kappa)**2.)/step[0]
                        gE = np.exp(-(deltaE/kappa)**2.)/step[1]
                elif option == 2:
                        gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
                        gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
 
                # update matrices
                E = gE*deltaE
                S = gS*deltaS
 
                # subtract a copy that has been shifted 'North/West' by one
                # pixel. don't as questions. just do it. trust me.
                NS[:] = S
                EW[:] = E
                NS[1:,:] -= S[:-1,:]
                EW[:,1:] -= E[:,:-1]
 
                # update the image
                imgout += gamma*(NS+EW)
 
                if ploton:
                        iterstring = "Iteration %i" %(ii+1)
                        ih.set_data(imgout)
                        ax2.set_title(iterstring)
                        fig.canvas.draw()
                        # sleep(0.01)
 
        return imgout
def ShowImage(title,img,ctype):
  plt.figure(figsize=(10, 10))
  if ctype=='bgr':
    b,g,r = cv2.split(img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(rgb_img)
  elif ctype=='hsv':
    rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
  elif ctype=='gray':
    plt.imshow(img,cmap='gray')
  elif ctype=='rgb':
    plt.imshow(img)
  else:
    raise Exception("Unknown colour type")
  plt.axis('off')
  plt.title(title)
  plt.show()

def remove_skull(image):
	print(image)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	ShowImage('asd',image,'gray')
	ret, thresh = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
	colormask = np.zeros(iage.shape, dtype=np.uint8)
	colormask[thresh!=0] = np.array((0,0,255))
	blended = cv2.addWeighted(image,0.7,colormask,0.1,0)

	ret, markers = cv2.connectedComponents(thresh)

	#Get the area taken by each component. Ignore label 0 since this is the background.
	marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
	#Get label of largest component by area
	largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
	#Get pixels which correspond to the brain
	brain_mask = markers==largest_component

	brain_out = image.copy()
	#In a copy of the original image, clear those pixels that don't correspond to the brain
	brain_out[brain_mask==False] = (0,0,0)
	ShowImage('Connected Components',brain_out,'rgb')

rs_contours = {}
for rs_file in glob.glob("./NewDS/Data/"+PACIENS+"/RS*"):
	rs = pydicom.read_file(rs_file)
	for contour_sequence in rs.ROIContourSequence[0].ContourSequence:
		sopiuid = contour_sequence.ContourImageSequence[0].ReferencedSOPInstanceUID
		if sopiuid not in rs_contours:
			rs_contours[sopiuid] = []
		rs_contours[sopiuid].append(np.array(contour_sequence.ContourData, dtype=np.float).reshape([-1,3]))

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# !!!

# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 380
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87
 
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# !!!

detector = cv2.SimpleBlobDetector_create(params)

print("RS_CONTOURS")
print(rs_contours)
filenames = glob.glob("./NewDS/Data/"+PACIENS+"/MR*.dcm")
filenames.sort(key=lambda x: x[x.find(" ")+1:-4].zfill(4))
i = 0
last_i = -1

kernel = np.ones((5,5),np.uint8)
print(filenames[:5])

cv2.namedWindow('options')
print(len(filenames))
first = True
detected_keypoints=[]
conf=get_config()

selected_images=[]
while True:
	#print(i, filenames[i])
	
	ds = pydicom.read_file(filenames[i])

	img = ds.pixel_array
	currentfile=int(filenames[i].split('/')[-2])
	print(currentfile)
	#print(ds.pixel_array.shape)
	mimax=conf.loc[conf['ID']==currentfile]
	
	img_min = np.min(img)
	img_max = np.max(img)
	print(i, filenames[i])
	curr_min=mimax['MIN']
	curr_max=mimax['MAX']
	
	if first:
		my_min_global = 326+img_min
		my_max_global = 580+img_min
		first = False
	if last_i != i:
		print(i, filenames[i])
		cv2.destroyWindow('options')
		cv2.namedWindow('options')
		cv2.createTrackbar('min', 'options', 0, img_max-img_min, nothing)
		cv2.createTrackbar('max', 'options', img_max-img_min, img_max-img_min, nothing)
		cv2.setTrackbarPos('min', 'options', my_min_global-img_min)
		cv2.setTrackbarPos('max', 'options', my_max_global-img_min)
	my_min = cv2.getTrackbarPos('min','options')
	my_max = cv2.getTrackbarPos('max','options')
	my_min_global = curr_min
	my_max_global = curr_max
	
	

	# for i in range(img_min,img_max,0.01*(img-max-img_min)):
	# 	my_min=i
	# 	my_max=i
	print(my_min_global, my_max_global)
	
	
	
	img = np.maximum(img, my_min+img_min)
	img = np.minimum(img, my_max+img_min)
	img = (img-np.min(img))/(np.max(img)-np.min(img))




	# img=anisodiff(img)

	img = (img*255).astype(np.uint8)
	new_img = img.copy()
# remove_skull(new_img)


	#!!!

	new_img = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, kernel)
	new_img = cv2.medianBlur(new_img, 5)
	new_img = cv2.bitwise_not(new_img)


	ret,new_img = cv2.threshold(new_img,127,255,0)
	cv2.imshow("thres_img",new_img)

	#!!!
	inverze_img= cv2.bitwise_not(new_img)

	keypoints = detector.detect(new_img)
	if len(keypoints) > 0 :
		for elem in keypoints:
			detected_keypoints.append(elem)

	# keypoints2=detector.detect(inverze_img)
	# if (len(keypoints2) > 0)  and (len(detected_keypoints)!=0):
	# 	if (cv2.KeyPoint_overlap(detected_keypoints[-1]	, keypoints2[0]	)):
	# 		keypoints=keypoints2
	# 		detected_keypoints.append(keypoints)


	#print("MINMAX", np.min(np.min(img)), np.max(np.max(img)))

	'''
	'''

	img = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	

	if ds.SOPInstanceUID in rs_contours:
			#print("CONTOUR NUM: ", len(rs_contours[ds.SOPInstanceUID]))
			for contour in rs_contours[ds.SOPInstanceUID]:
				rcontour = [get_pcoord_from_rcoord(ds,x) for x in contour]
				cv2.polylines(img, [np.int32(rcontour)], 1, (0,255,0))

	#cv2.
				print(contour)

	cv2.imshow("dicom image", img)
	last_i = i
	key = cv2.waitKey(10)
	if key == ord('a'):
		i = np.max([0, i-1])
	elif key == ord("d"):
		i = np.min([i+1, len(filenames)-1])
	elif key == ord("q"):
		cv2.destroyAllWindows()
		break
