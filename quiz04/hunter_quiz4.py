import numpy as np
import math


p_cad = 4/7.0
p_civ = 4/7.0

cad = np.asarray([[50, 55, 40,60], [88000, 85000, 125000, 92000]])
civ = np.asarray([[22, 35, 45,32], [21000, 30000, 31000, 79000]])

cad_conv = np.cov(cad)
civ_conv = np.cov(civ)

cad_conv_inv = np.linalg.inv(cad_conv)
civ_conv_inv = np.linalg.inv(civ_conv)

m_cad = np.vstack([[51.25],[97500]])
m_civ = np.vstack([[33.5],[40250]])

Wi_cad = -0.5*cad_conv_inv
Wi_civ = -0.5*civ_conv_inv

wi_cad = np.matmul(cad_conv_inv, m_cad)
wi_civ = np.matmul(civ_conv_inv, m_civ)

wi0_cad = 0.5*np.matmul(m_cad.T,np.matmul(cad_conv_inv, m_cad)) - 0.5*np.log(np.absolute(cad_conv)) + np.log(p_cad)
wi0_civ = 0.5*np.matmul(m_civ.T,np.matmul(civ_conv_inv, m_civ)) - 0.5*np.log(np.absolute(civ_conv)) + np.log(p_civ)

#sample = np.asarray([33,31000])
sample = np.asarray([62,101000])

cad_pred = np.matmul(sample.T,np.matmul(Wi_cad,sample)) + np.matmul(wi_cad.T,sample) + wi0_cad
civ_pred = np.matmul(sample.T,np.matmul(Wi_civ,sample)) + np.matmul(wi_civ.T,sample) + wi0_civ

print(cad_pred)

print(np.linalg.det(cad_pred))
print(np.linalg.det(civ_pred))

