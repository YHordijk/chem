import numpy as np
import math
from scipy.special import factorial, factorial2, comb

def gint(m, center1, exponent1, n, center2, exponent2): #calculate one electron integral of two gaussian primitives
	newcenter = (exponent1 * center1 + exponent2 * center2) / (exponent1 + exponent2)
	newexponent = exponent1 + exponent2
	tempexponent = exponent1 * exponent2 / (exponent1 + exponent2)
	e12 = np.exp(-tempexponent*(center1 - center2)**2)
	res = 0
	for i in range(m + 1):
		for j in range(n + 1):
			if (i + j) % 2 == 0:
				res += (np.sqrt(np.pi/newexponent)*comb(m, i)*comb(n, j)*
						factorial2(i + j - 1)/(2*newexponent)**((i + j)/2)*
						(newcenter - center1)**(m - i)*(newcenter - center2)**(n - j)
					   )
	res = e12 * res
	return res
						

def E(l1, l2, t, center1, center2, exponent1, exponent2):#calculate the gaussian-hermite expansion coefficient using recurence
	newcenter = (exponent1 * center1 + exponent2 * center2) / (exponent1 + exponent2)
	sumexponent = exponent1 + exponent2
	diffcenter = center1 - center2
	redexponent = exponent1 * exponent2 / (exponent1 + exponent2)
	if t > l1 + l2:
		return 0
	if l1 < 0 or l2 < 0 or t < 0:
		return 0
	elif l1 == 0 and l2 == 0 and t == 0:
		return np.exp(-redexponent*diffcenter**2)
	elif l1 > 0:
		return (1/(2*sumexponent)*E(l1-1, l2, t - 1,center1,center2,exponent1,exponent2)
				+(newcenter - center1)*E(l1-1,l2,t,center1,center2,exponent1,exponent2)
				+(t + 1)*E(l1-1,l2, t+1, center1,center2,exponent1,exponent2))
	elif l1 == 0:
		return (1/(2*sumexponent)*E(l1, l2-1, t - 1,center1,center2,exponent1,exponent2)
				+(newcenter - center2)*E(l1,l2-1,t,center1,center2,exponent1,exponent2)
				+(t + 1)*E(l1,l2-1, t+1, center1,center2,exponent1,exponent2))
	return 0


def S(m1, m2, center1, center2, exponent1, exponent2): #calculate overlap type integral
	return np.sqrt(np.pi/(exponent1 + exponent2))*E(m1, m2, 0, center1, center2, exponent1, exponent2)


def T(m1, m2, center1, center2, exponent1, exponent2): #calculate kinetic type integral
	res = 0
	res += -2*exponent2*S(m1, m2 + 2, center1, center2, exponent1, exponent2)
	res += exponent2*(2*m2+1)*S(m1, m2, center1, center2, exponent1, exponent2)
	res += -1/2*m2*(m2-1)*S(m1, m2 - 2, center1, center2, exponent1, exponent2)
	return res
	

def F(n, x): #calculate Boys function value by numerical integration
	if x < 1e-7:
		return 1/(2*n + 1)
	if n == 20:
		res1 = 1/(2*n + 1)
		#if x < 1e-7:
		#    return res1
		for k in range(1,11):
			res1 += (-x)**k/factorial(k)/(2*n+2*k+1)
		res2 = factorial2(2*n-1)/2**(n+1)*np.sqrt(np.pi/x**(2*n+1))
		res = min(res1, res2)
		return res
	return (2*x*F(n+1,x)+np.exp(-x))/(2*n+1)


def R(t, u, v, n, p, x, y, z):
	if t < 0 or u < 0 or v < 0:
		return 0
	if t == 0 and u == 0 and v == 0:
		return (-2*p)**n*F(n,p*(x**2+y**2+z**2))
	if t > 0:
		return (t-1)*R(t-2,u,v,n+1,p,x,y,z)+x*R(t-1,u,v,n+1,p,x,y,z)
	if u > 0:
		return (u-1)*R(t,u-2,v,n+1,p,x,y,z)+y*R(t,u-1,v,n+1,p,x,y,z)
	if v > 0:
		return (v-1)*R(t,u,v-2,n+1,p,x,y,z)+z*R(t,u,v-1,n+1,p,x,y,z)
	
	
def overlap(ao1, ao2): #calculate overlap matrix <psi|phi>
	l1, m1, n1 = ao1.angular
	l2, m2, n2 = ao2.angular
	x1, y1, z1 = ao1.center
	x2, y2, z2 = ao2.center
	res = 0
	for i in range(len(ao1.pre_exponents)):
		for j in range(len(ao2.pre_exponents)):
			exponent1 = ao1.exponents[i]
			exponent2 = ao2.exponents[j]

			res += (ao1.pre_exponents[i]*ao2.pre_exponents[j]*
					S(l1, l2, x1, x2, exponent1, exponent2)*
					S(m1, m2, y1, y2, exponent1, exponent2)*
					S(n1, n2, z1, z2, exponent1, exponent2))
	return ao1.pre_factor * ao2.pre_factor * res



def kinetic(ao1, ao2): #calculate kinetic integral <psi|-1/2*del^2|phi>
	l1, m1, n1 = ao1.angular
	l2, m2, n2 = ao2.angular
	x1, y1, z1 = ao1.center
	x2, y2, z2 = ao2.center
	res = 0
	for i in range(len(ao1.pre_exponents)):
		for j in range(len(ao2.pre_exponents)):
			exponent1 = ao1.exponents[i]
			exponent2 = ao2.exponents[j]
			res += (ao1.pre_exponents[i]*ao2.pre_exponents[j]*
					(T(l1,l2,x1,x2,exponent1,exponent2)*S(m1,m2,y1,y2,exponent1,exponent2)*S(n1,n2,z1,z2,exponent1,exponent2) +
					 S(l1,l2,x1,x2,exponent1,exponent2)*T(m1,m2,y1,y2,exponent1,exponent2)*S(n1,n2,z1,z2,exponent1,exponent2) +
					 S(l1,l2,x1,x2,exponent1,exponent2)*S(m1,m2,y1,y2,exponent1,exponent2)*T(n1,n2,z1,z2,exponent1,exponent2))
				   )
	return res


def oneelectron(ao1, centerC, ao2):
	l1, m1, n1 = ao1.angular
	l2, m2, n2 = ao2.angular
	a = l1 + m1 + n1
	b = l2 + m2 + n2
	c = a + b
	x1, y1, z1 = ao1.center
	x2, y2, z2 = ao2.center
	xc, yc, zc = centerC # coordinate of atom with charge Z
	res = 0
	for i in range(len(ao1.pre_exponents)):
		for j in range(len(ao2.pre_exponents)):
			exponent1 = ao1.exponents[i]
			exponent2 = ao2.exponents[j]
			p = exponent1 + exponent2
			xp = (exponent1*x1+exponent2*x2)/p
			yp = (exponent1*y1+exponent2*y2)/p
			zp = (exponent1*z1+exponent2*z2)/p
			xpc = xp - xc
			ypc = yp - yc
			zpc = zp - zc
			for t in range(c+1):
				for u in range(c+1):
					for v in range(c+1):
						res += (ao1.pre_exponents[i]*ao2.pre_exponents[j]*
								2*np.pi/p*E(l1,l2,t,x1,x2,exponent1,exponent2)*
								E(m1,m2,u,y1,y2,exponent1,exponent2)*
								E(n1,n2,v,z1,z2,exponent1,exponent2)*
								R(t,u,v,0,p,xpc,ypc,zpc))
	return res
				  
	
def twoelectron(ao1, ao2, ao3, ao4):
	res = 0
	l1, m1, n1 = ao1.angular
	l2, m2, n2 = ao2.angular
	l3, m3, n3 = ao3.angular
	l4, m4, n4 = ao4.angular
	x1, y1, z1 = ao1.center
	x2, y2, z2 = ao2.center
	x3, y3, z3 = ao3.center
	x4, y4, z4 = ao4.center
	a = l1 + m1 + n1
	b = l2 + m2 + n2
	c = l3 + m3 + n3
	d = l4 + m4 + n4
	for i in range(len(ao1.pre_exponents)):
		for j in range(len(ao2.pre_exponents)):
			for k in range(len(ao3.pre_exponents)):
				for l in range(len(ao4.pre_exponents)):
					exponent1 = ao1.exponents[i]
					exponent2 = ao2.exponents[j]
					exponent3 = ao3.exponents[k]
					exponent4 = ao4.exponents[l]
					p = (exponent1 + exponent2)
					q = (exponent3 + exponent4)
					alpha = p*q/(p + q)
					xp = (x1*exponent1+x2*exponent2)/p
					yp = (y1*exponent1+y2*exponent2)/p
					zp = (z1*exponent1+z2*exponent2)/p
					xq = (x3*exponent3+x4*exponent4)/q
					yq = (y3*exponent3+y4*exponent4)/q
					zq = (z3*exponent3+z4*exponent4)/q
					xpq = xp - xq
					ypq = yp - yq
					zpq = zp - zq
					for t in range(a + b + 1):
						for u in range(a + b + 1):
							for v in range(a + b + 1):
								for tau in range(c + d + 1):
									for miu in range(c + d + 1):
										for phi in range(c + d + 1):
											res += (ao1.pre_exponents[i]*ao2.pre_exponents[j]*ao3.pre_exponents[k]*ao4.pre_exponents[l]*
													2*np.pi**(5/2)/p/q/np.sqrt(p+q)*
													E(l1, l2, t, x1, x2, exponent1, exponent2)*
													E(m1, m2, u, y1, y2, exponent1, exponent2)*
													E(n1, n2, v, z1, z2, exponent1, exponent2)*
													E(l3, l4, tau, x3, x4, exponent3, exponent4)*
													E(m3, m4, miu, y3, y4, exponent3, exponent4)*
													E(n3, n4, phi, z3, z4, exponent3, exponent4)*
													(-1)**(tau + miu + phi)*
													R(t+tau, u+miu, v+phi, 0, alpha, xpq, ypq, zpq)
													)
	return res
							

def coulombicAttraction(ao1, atomlist, ao2):
	res = 0
	for a in atomlist:
		Z = a.atom_number
		centerC = a.position
		res += -Z*oneelectron(ao1, centerC, ao2)
	return res







def overlap_matrix(aos):
	M = np.zeros((len(aos), len(aos)))
	for i in range(len(aos)):
		M[i,i] = 1.
		for j in range(i, len(aos)):
			M[i,j] = overlap(aos[i], aos[j])
			M[j,i] = M[i,j]
	return M

def fock_matrix(aos, atomlist, coeffs):
	n = len(aos)
	F = np.zeros((n,n))
	#first build one-electron hamiltonian
	H = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			H[i,j] = kinetic(aos[i], aos[j]) + coulombicAttraction(aos[i], atomlist, aos[j])

	#build coulomb repulsion tensor
	CRT = np.zeros((n,n,n,n))
	for i in range(n):
		for j in range(n):
			for k in range(n):
				for l in range(n):
					CRT[i][j][k][l] = twoelectron(aos[i],aos[j],aos[k],aos[l])

	#build J - .5K matrix
	JK = np.zeros((n,n))
	for j in range(n):
		for k in range(n):
			for l in range(n):
				for m in range(n):
					for o in range(n):
						JK[j][k] += (coeffs[l][o]*coeffs[m][o] * (2*CRT[j][k][l][m]-CRT[j][m][k][l]))

	F = H + JK

	return F, JK

def rothaan_hall(S, F):
	E, C = np.linalg.eigh((F, S))
	return E, C

def hartree_fock(aos, atomlist, epsilon=1e-7, C=None, max_iter=100):
	n = len(aos)
	if C is None:
		C = np.eye(n)

	S = overlap_matrix(aos)

	energy_list = []
	delta = 0
	for i in range(max_iter):
		F, JK = fock_matrix(aos, atomlist, C)
		E, C = rothaan_hall(S, F)
		EHF = 0
		C = np.zeros((n,n))
		for i in range(n):
			EHF += 2*E[i][i] - 1/2*JK[i][i]
			C = C + 2*C[i][i] - 1/2*JK[i][i]

		energy_list.append(EHF)

		print(f'Iteration: {i}, Energy: {EHF}, Delta: {delta}')
		if i > 1:
			delta = abs(energy_list[-1] - energy_list[-2])
			if abs(energy_list[-1] - energy_list[-2]) < epsilon:
				print(f'SCF converged')
				break

	return energy_list, C