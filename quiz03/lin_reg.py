import numpy as np 

def main():
	x = np.asarray([12, 14, 17, 19])
	r = np.asarray([13, 14, 18, 17])
	N = x.size
	A = np.asarray([[N, x.sum()], [x.sum(), np.sum(x * x)]])
	A_1 = np.linalg.inv(A)
	print(f'A_1:\n{A_1}')
	
	y = np.asarray([[r.sum()], [np.sum(x * r)]])
	w = np.matmul(A_1, y)
	print(f'w:\n{w}')

if __name__ == '__main__':
	main()