import numpy as np 

def main():
	'''
	A/a class is cadi, B/b class is civic
	'''

	# Prob of class
	Pa = 4 / 8.0
	Pb = 4 / 8.0

	# Average of both features of each class
	ma1 = (50 + 55 + 40 + 60) / 4.0
	ma2 = (88 + 85 + 125 + 92) * 1000 / 4.0
	mb1 = (22 + 35 + 45 + 32) / 4.0
	mb2 = (21 + 30 + 31 + 79) * 1000 / 4.0

	# 2x1 stacks of of feature means
	ma = np.vstack([ma1, ma2])
	mb = np.vstack([mb1, mb2])
	print(f'ma.shape = {ma.shape}')

	# Cadi
	A = np.asarray([
		[50, 55, 40, 60],
		[88000, 85000, 125000, 92000]
	])
	print(f'A.shape = {A.shape}')

	# Civic
	B = np.asarray([
		[22, 35, 45, 32],
		[21000, 30000, 31000, 79000]
	])

	# Cadi covariance between features and cov inverse
	A_cov = np.cov(A)
	A_cov_inv = np.linalg.inv(A_cov)
	print(f'A_cov.shape = {A_cov.shape}')

	# Civic covariance between features and cov inverse
	B_cov = np.cov(B)
	B_cov_inv = np.linalg.inv(B_cov)

	# Weighted inverse covariance
	Wa = -0.5 * A_cov_inv
	Wb = -0.5 * B_cov_inv
	print(f'Wa.shape = {Wa.shape}')

	# Not sure what this is exactly called
	wa = np.matmul(A_cov_inv, ma)
	wb = np.matmul(B_cov_inv, mb)
	print(f'wa.shape = {Wa.shape}')

	# Not sure what this is exactly either
	wa0_p1 = -0.5 * np.matmul(np.matmul(ma.T, A_cov_inv), ma)
	wa0_p2 = 0.5 * np.log(np.linalg.det(A_cov))
	wa0 = wa0_p1 - wa0_p2 + np.log(Pa)
	print(f'wa0_p1 = {wa0_p1}, {wa0_p1.shape}')
	print(f'wa0_p2 = {wa0_p2}, {wa0_p2.shape}')
	print(f'wa0.shape = {wa0.shape}')

	# Not sure what this is exactly either
	wb0_p1 = -0.5 * np.matmul(np.matmul(mb.T, B_cov_inv), mb)
	wb0_p2 = 0.5 * np.log(np.linalg.det(B_cov))
	wb0 = wb0_p1 - wb0_p2  + np.log(Pb)

	# Pred for 33, 31000
	x = np.vstack([33, 31000])
	A_pred = np.matmul(np.matmul(x.T, Wa), x) + np.matmul(wa.T, x) + wa0
	B_pred = np.matmul(np.matmul(x.T, Wb), x) + np.matmul(wb.T, x) + wb0
	print(f'{x.flatten()}: A_pred = {A_pred[0, 0]}, B_pred = {B_pred[0, 0]}')
	
	# Pred for 62, 101000
	x = np.vstack([62, 101000])
	A_pred = np.matmul(np.matmul(x.T, Wa), x) + np.matmul(wa.T, x) + wa0
	B_pred = np.matmul(np.matmul(x.T, Wb), x) + np.matmul(wb.T, x) + wb0
	print(f'{x.flatten()}: A_pred = {A_pred[0, 0]}, B_pred = {B_pred[0, 0]}')

if __name__ == '__main__':
	main()