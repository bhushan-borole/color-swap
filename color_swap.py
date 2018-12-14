import numpy as np
import cv2
import argparse

def image_stats(image):
	# compute the mean and standard deviation of each channel
	(l, a, b) = cv2.split(image)
	(l_mean, l_std) = (l.mean(), l.std())
	(a_mean, a_std) = (a.mean(), a.std())
	(b_mean, b_std) = (b.mean(), b.std())
 
	# return the color statistics
	return (l_mean, l_std, a_mean, a_std, b_mean, b_std)

def color_transfer(source, target):
	'''
	convert the images from the RGB to LAB color space(3 axes model)
	L : white to mid-gray
	a : cyan to magenta
	b : blue to yellow
	'''
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype('float32')
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype('float32')

	# compute color statistics
	(l_mean_src, l_std_src, a_mean_src, a_std_src, b_mean_src, b_std_src) = image_stats(source)
	(l_mean_tar, l_std_tar, a_mean_tar, a_std_tar, b_mean_tar, b_std_tar) = image_stats(target)

	# subtrtacting means from the target image
	(l, a, b) = cv2.split(target)
	l -= l_mean_tar
	a -= a_mean_tar
	b -= b_mean_tar

	# scaling
	l = (l_std_tar / l_std_src) * l
	a = (a_std_tar / a_std_src) * a
	b = (b_std_tar / b_std_src) * b

	# add in the source mean
	l += l_mean_src
	a += a_mean_src
	b += b_mean_src

	# clip the pixel intensities if they fall out of range
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)

	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype('uint8'), cv2.COLOR_LAB2BGR)

	return transfer

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('-s', '--source', help='path of the source image')
	ap.add_argument('-t', '--target', help='path of the target image')
	args = vars(ap.parse_args())

	# load the images
	source = cv2.imread(args['source'])
	target = cv2.imread(args['target'])

	transfer = color_transfer(source, target)
	cv2.imshow('Source', source)
	cv2.imshow('Target', target)
	cv2.imshow('Transfer', transfer)
	cv2.waitKey(0)

if __name__ == '__main__':
	main()
