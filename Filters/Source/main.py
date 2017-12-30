# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):

   # Write histogram equalization here
   #img = cv2.imread('input1.jpg',1)
   b,g,r=cv2.split(img_in)

   bhist,bbins=np.histogram(b.flatten(),256,[0,256])
   cdf = bhist.cumsum()
   cdf_normalized1 = cdf * bhist.max()/ cdf.max()
   cdf_m = np.ma.masked_equal(cdf,0)
   cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
   cdf1 = np.ma.filled(cdf_m,0).astype('uint8')

   ghist,gbins=np.histogram(g.flatten(),256,[0,256])
   cdf = ghist.cumsum()
   cdf_normalized = cdf * ghist.max()/ cdf.max()
   cdf_m = np.ma.masked_equal(cdf,0)
   cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
   cdf2 = np.ma.filled(cdf_m,0).astype('uint8')

   rhist,rbins=np.histogram(r.flatten(),256,[0,256])
   cdf = rhist.cumsum()
   cdf_normalized = cdf * rhist.max()/ cdf.max()
   cdf_m = np.ma.masked_equal(cdf,0)
   cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
   cdf3 = np.ma.filled(cdf_m,0).astype('uint8')

   imgfinalb=cdf1[b];
   imgfinalg=cdf2[g];
   imgfinalr=cdf3[r];

   merged_img = np.dstack((imgfinalb,imgfinalg,imgfinalr))
   img_out = merged_img
   return True, img_out

def Question1():
   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);

   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
                     
   # Write out the result
   output_name = sys.argv[3] + "1.jpg"
   cv2.imwrite(output_name, output_image)

   return True
                                    
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):

   # Write low pass filter here
   img_in=cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
   imgfft = np.fft.fft2(img_in)
   fshifthp = np.fft.fftshift(imgfft) #shifting the dc point to the centre
   imgrows,imgcols=img_in.shape
   crow,ccol=int(imgrows/2) , int(imgcols/2)
   mask = np.zeros((imgrows,imgcols),np.uint8) # in mask, center square is 1, remaining all zeros
   mask[crow-10:crow+10, ccol-10:ccol+10] = 1
# apply mask and inverse FFT
   fshiftlp = np.fft.fftshift(imgfft)
   fshiftlp = fshiftlp*mask
   f_ishift = np.fft.ifftshift(fshiftlp)
   img_lowpass = np.fft.ifft2(f_ishift)
   img_lowpass=np.abs(img_lowpass)
   img_out=img_lowpass # Low pass filter result
   return True, img_out

def high_pass_filter(img_in):

   # Write high pass filter here
   img_in=cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
   imgfft = np.fft.fft2(img_in)
   fshifthp = np.fft.fftshift(imgfft) #shifting the dc point to the centre
   imgrows,imgcols=img_in.shape
   crow,ccol=int(imgrows/2) , int(imgcols/2)
   fshifthp[crow-10:crow+10, ccol-10:ccol+10]=0 # masking the centre square(20x20) to be 0 and the rest is 1
   f_ishift=np.fft.ifftshift(fshifthp) #shift the dc point back by inverse shift
   img_highpass=np.fft.ifft2(f_ishift) #inverse FFT to get the image back to spacial domain
   img_highpass=np.abs(img_highpass) #getting rid of negatives if any
   img_out = img_highpass # High pass filter result
                                                           
   return True, img_out

def deconvolution(img_in):
   
   # Write deconvolution codes here
   img_in=cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
   gk = cv2.getGaussianKernel(21,5)
   gk = gk * gk.T
   def ft(im, newsize=None):
       dft = np.fft.fft2(np.float32(im),newsize)
       return np.fft.fftshift(dft)
   def ift(shift):
       f_ishift = np.fft.ifftshift(shift)
       img_back = np.fft.ifft2(f_ishift)
       return np.abs(img_back)
   imf = ft(img_in, (img_in.shape[0],img_in.shape[1])) # make sure sizes match
   gkf = ft(gk, (img_in.shape[0],img_in.shape[1])) # so we can multiple easily
   imconvf = imf / gkf
# now for example we can reconstruct the blurred image from its FT
   unblurred = ift(imconvf)
   unblurred = 255*unblurred
   unblurred = unblurred.astype('uint8')
   img_out = unblurred # Deconvolution result
   return True, img_out

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "2.jpg"
   output_name2 = sys.argv[4] + "3.jpg"
   output_name3 = sys.argv[4] + "4.jpg"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here
   A = img_in1[:,:img_in1.shape[0]]
   B = img_in2[:img_in1.shape[0],:img_in2.shape[0]]
# generate Gaussian pyramid for A
   G = A.copy()
   gpA = [G]
   for i in xrange(6):
      G = cv2.pyrDown(G)
      gpA.append(G)
# generate Gaussian pyramid for B
   G = B.copy()
   gpB = [G]
   for i in xrange(6):
      G = cv2.pyrDown(G)
      gpB.append(G)
# generate Laplacian Pyramid for A
   lpA = [gpA[5]]
   for i in xrange(5,0,-1):
       GE = cv2.pyrUp(gpA[i])
       L = cv2.subtract(gpA[i-1],GE)
       lpA.append(L)
# generate Laplacian Pyramid for B
   lpB = [gpB[5]]
   for i in xrange(5,0,-1):
       GE = cv2.pyrUp(gpB[i])
       L = cv2.subtract(gpB[i-1],GE)
       lpB.append(L)
# Now add left and right halves of images in each level
   LS = []
   for la,lb in zip(lpA,lpB):
       rows,cols,dpt = la.shape
       ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
       LS.append(ls)
# now reconstruct
   ls_ = LS[0]
   for i in xrange(1,6):
       ls_ = cv2.pyrUp(ls_) 
       ls_ = cv2.add(ls_, LS[i])
# image with direct connecting each half
   real = np.hstack((A[:,:cols/2],B[:,cols/2:]))             
   img_out = ls_ # Blending result
   
   return True, img_out

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "5.jpg"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
	  
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()
