from __future__ import division
import numpy as np 
import pywt
from matplotlib import pyplot as plt
import cv2
from Tkinter import *
#1)Reading files and converting to rgb 
to_hide_og= cv2.imread("to_hide.jpg")
to_hide_og = cv2.cvtColor(to_hide_og, cv2.COLOR_BGR2RGB)
to_send_og = cv2.imread("to_send.jpg")
to_send_og = cv2.cvtColor(to_send_og, cv2.COLOR_BGR2RGB)
###
dimh,dimw,dimch= to_send_og.shape

#2) seperating chaanels for cover and hidden images 
to_send_r = to_send_og[:,:,0]
to_send_g = to_send_og[:,:,1]
to_send_b = to_send_og[:,:,2]

to_hide_r = to_hide_og[:,:,0]
to_hide_g = to_hide_og[:,:,1]
to_hide_b = to_hide_og[:,:,2]

#3) taking wavelet transform for cover and hidden  image 
send_r = pywt.dwt2(to_send_r, "haar")
cAr, (cHr, cVr, cDr) = send_r
send_g = pywt.dwt2(to_send_g, "haar")
cAg, (cHg, cVg, cDg) = send_g
send_b = pywt.dwt2(to_send_b, "haar")
cAb, (cHb, cVb, cDb) = send_b


hide_r = pywt.dwt2(to_hide_r, "haar")
cAr1, (cHr1, cVr1, cDr1) = hide_r
hide_g = pywt.dwt2(to_hide_g, "haar")
cAg1, (cHg1, cVg1, cDg1) = hide_g
hide_b = pywt.dwt2(to_hide_b, "haar")
cAb1, (cHb1, cVb1, cDb1) = hide_b

print(cAr.shape)

#4)compute svd for cover, hiding  image 

Pr, Dr, Qr = np.linalg.svd(cAr, full_matrices=False)
Pg, Dg, Qg = np.linalg.svd(cAg, full_matrices=False)
Pb, Db, Qb = np.linalg.svd(cAb, full_matrices=False)

print(Pr.shape,Dr.shape,Qr.shape)#just for debugging


P1r, D1r, Q1r = np.linalg.svd(cAr1, full_matrices=False)
P1g, D1g, Q1g = np.linalg.svd(cAg1, full_matrices=False)
P1b, D1b, Q1b = np.linalg.svd(cAb1, full_matrices=False)


#5)watermarking R,G,B channels using approximate values IE cA, also cAr == Dr logically //add hidden and cover 

S_wimgr=Dr+(0.10*D1r);
S_wimgg=Dg+(0.10*D1g);
S_wimgb=Db+(0.10*D1b);

#6)merging -get new values (R,G,B) using Pr S_wimgr and Qr --> basically reconstruct the SVD 

wimgr = np.dot(Pr*S_wimgr,Qr);

wimgg = np.dot(Pg*S_wimgg,Qg);

wimgb = np.dot(Pb*S_wimgb,Qb);

a=wimgr.astype(int)
b=wimgg.astype(int)
c=wimgb.astype(int)

#7) merge reconstructed svd , this is using approximate values hence dimension/2
wimg =cv2.merge((a,b,c))
h,w,ch= wimg.shape
print("stego no of rows and cols are " + str(h) +","+str(w) )

#8)rgb coeffs for idwt, so that you can recreate a original img but with cA now having hidden info  
proc_r=wimg[:,:,0],(cHr, cVr, cDr)
proc_g=wimg[:,:,1],(cHg, cVg, cDg)
proc_b=wimg[:,:,2],(cHb, cVb, cDb)

#  3 stego images 
processed_rgbr= pywt.idwt2(proc_r,'haar');
processed_rgbg= pywt.idwt2(proc_g,'haar');
processed_rgbb= pywt.idwt2(proc_b,'haar');

wimghd=cv2.merge((processed_rgbr.astype(int),processed_rgbg.astype(int),processed_rgbb.astype(int)))
h,w,ch= wimghd.shape
print("stego no of rows and cols are " + str(h) +","+str(w) )
fig1 = plt.figure(frameon=False)
fig1.set_size_inches(float(w)/100,float(h)/100)
ax1 = plt.Axes(fig1, [0., 0., 1., 1.])
ax1.set_axis_off()
fig1.add_axes(ax1)
ax1.imshow(wimghd,aspect='auto')
fig1.savefig('stego.png')

#9)applying dwt to 3 stego channel images to get coeffs of stego image in R,G,B

Psend_r = pywt.dwt2(processed_rgbr, "haar")
PcAr, (PcHr, PcVr, PcDr) = Psend_r

Psend_g = pywt.dwt2(processed_rgbg, "haar")
PcAg, (PcHg, PcVg, PcDg) = Psend_g

Psend_b = pywt.dwt2(processed_rgbb, "haar")
PcAb, (PcHb, PcVb, PcDb) = Psend_b

#10)again do svd to decompose the approximate value PcAr
PPr, PDr, PQr = np.linalg.svd(PcAr, full_matrices=False)
PPg, PDg, PQg = np.linalg.svd(PcAg, full_matrices=False)
PPb, PDb, PQb = np.linalg.svd(PcAb, full_matrices=False)

#11) subtract from R,G,B channels values of cover image 
S_ewatr=(PDr-Dr)/0.10;
S_ewatg=(PDg-Dg)/0.10;
S_ewatb=(PDb-Db)/0.10;

#12)merging-> merge  new approximations with hidden SVD found earlier 
ewatr = np.dot(P1r*S_ewatr,Q1r);
ewatg = np.dot(P1g*S_ewatg,Q1g);
ewatb = np.dot(P1b*S_ewatb,Q1b);

#13)merge recreate hidden image -still  based on approximations ,hence dim /2
d=ewatr.astype(int)
e=ewatg.astype(int)
f=ewatb.astype(int)
eimg =cv2.merge((d,e,f))

#14)coeffs of original hidden image except the new derived appproximation 
eproc_r=eimg[:,:,0],(cHr1, cVr1, cDr1)
eproc_g=eimg[:,:,1],(cHg1, cVg1, cDg1)
eproc_b=eimg[:,:,2],(cHb1, cVb1, cDb1)

#15) hidden stego images get  high res r,g,b seperate images/channels usign idwt
eprocessed_rgbr= pywt.idwt2(eproc_r,'haar');
eprocessed_rgbg= pywt.idwt2(eproc_g,'haar');
eprocessed_rgbb= pywt.idwt2(eproc_b,'haar');
#plt.figure(6)

# just converting float to int prior to cv2.merge 
x1= eprocessed_rgbr.astype(int)

y1= eprocessed_rgbg.astype(int)

z1= eprocessed_rgbb.astype(int)

#16) combine different high res r,g,b to get hidden image //figure 9 is final output.
hidden_rgb = cv2.merge((x1,y1,z1))

h1,w1,ch1= hidden_rgb.shape
print("hidden no of rows and cols are "+ str(h1) +","+ str(w1))

fig = plt.figure(frameon=False)
fig.set_size_inches(7.20,4.80)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(hidden_rgb,aspect='auto')
fig.savefig('hidden_rgb.png')

plt.show()

