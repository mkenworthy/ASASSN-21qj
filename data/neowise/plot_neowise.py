import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import unique,vstack,Table


t = ascii.read('ASASSN-21qj_2013-2021.tbl')

fig, ax = plt.subplots(2,1,figsize=(8,5),sharex=True)
ax[0].errorbar(t['mjd'],t['w1mpro'],yerr=t['w1sigmpro'],fmt='.')
ax[0].invert_yaxis()
ax[1].errorbar(t['mjd'],t['w2mpro'],yerr=t['w2sigmpro'],fmt='.')
ax[1].invert_yaxis()
ax[1].set_xlabel('Epoch [MJD]')
ax[0].set_ylabel('w1mpro')
ax[1].set_ylabel('w2mpro')

# calculate weighted mean at each epoch

fig, ax = plt.subplots(3,1,figsize=(8,7),sharex=True)
ax[0].errorbar(t['mjd'],t['w1mpro'],yerr=t['w1sigmpro'],fmt='.')
ax[0].invert_yaxis()
ax[1].errorbar(t['mjd'],t['w2mpro'],yerr=t['w2sigmpro'],fmt='.')
ax[1].invert_yaxis()

ax[2].set_xlabel('Epoch [MJD]')
ax[0].set_ylabel('w1mpro')
ax[1].set_ylabel('w2mpro')
ax[2].set_ylabel('w1-w2')
ax[0].set_ylim(12,10.5)
ax[1].set_ylim(12,10.5)
import uncertainties
from uncertainties.umath import *
from uncertainties import unumpy
from uncertainties import umath

t_s = 56787
t_e = 59532
nmag = 16

wt = np.zeros(nmag)
w1 = unumpy.uarray(wt,wt)
w2 = unumpy.uarray(wt,wt)
wcol = unumpy.uarray(wt,wt)

for (j,i) in enumerate(np.linspace(t_s, t_e, 16)):
#    ax[0].scatter(i-60,11.4)
#    ax[0].scatter(i+60,11.4)
#    ax[1].scatter(i-60,11.4)
#    ax[1].scatter(i+60,11.4)
    m = (t['mjd']>(i-60))*(t['mjd']<(i+60))

    w_time = t['mjd'][m]

    w1_mag = t['w1mpro'][m]
    w1_sig = t['w1sigmpro'][m]
    w2_mag = t['w2mpro'][m]
    w2_sig = t['w2sigmpro'][m]

    w1all = unumpy.uarray(w1_mag, w1_sig)
    w2all = unumpy.uarray(w2_mag, w2_sig)

    w1mean = w1all.mean()
    w2mean = w2all.mean()

    w1w2col = w1mean-w2mean

    w1[j] = w1mean
    w2[j] = w2mean
    wcol[j] = w1w2col
    wt[j] = np.average(t['mjd'][m])

ax[0].errorbar(wt,unumpy.nominal_values(w1),yerr=unumpy.std_devs(w1),fmt='.')
ax[1].errorbar(wt,unumpy.nominal_values(w2),yerr=unumpy.std_devs(w2),fmt='.')
ax[2].errorbar(wt,unumpy.nominal_values(wcol),yerr=unumpy.std_devs(wcol),fmt='.')
fig.suptitle("NEOWISE photometry of ASASSSN-21qj")
print(wcol)
fig.savefig("NEOWISE_ASASSN-21dj.pdf")

tn = Table([wt, unumpy.nominal_values(w1),unumpy.std_devs(w1),
            unumpy.nominal_values(w2),unumpy.std_devs(w2),
            unumpy.nominal_values(wcol),unumpy.std_devs(wcol)],
            names=['MJD','w1','w1err','w2','w2err','w1w2','w1w2err'])

tn.format = '%.4f'
tn['MJD'].format = '%.5f'
tn['w1'].format = '%.3f'
tn['w1err'].format = '%.3f'
tn['w2'].format = '%.3f'
tn['w2err'].format = '%.3f'
tn['w1w2'].format = '%.3f'
tn['w1w2err'].format = '%.3f'

print(tn)
tn.write('obs_NEOWISE.ecsv',format='ascii.ecsv',overwrite=True)

plt.show()

