import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import unique

from datetime import datetime
from astropy.time import Time


fin = 'light_curve_5740d76a-4445-4b58-bc83-4f5b93cf6b31.csv'
t = ascii.read(fin)

print(t)

#      HJD           UT Date       Camera FWHM Limit   mag   mag_err flux(mJy) flux_err Filter
# ------------- ------------------ ------ ---- ------ ------ ------- --------- -------- ------
# 2457420.65322 2016-02-02.1500246     be 1.46 17.458  13.45   0.005    15.995     0.08      V

t['MJD'] = t['HJD']-2400000.5


# how soon is now?
now = Time(datetime.utcnow(),format='datetime')
print('current MJD is {}'.format(now.mjd))

(ax) = plt.subplots(1,1,figsize=(12,6))
plt.errorbar(t['MJD'],t['flux(mJy)'],yerr=t['flux_err'],fmt='.')
plt.ylabel('Flux [mJy]')
plt.xlabel('Epoch [MJD]')
plt.title('data from {}'.format(fin))


# get a list of the unique bandpasses
t_by_filter = t.group_by('Filter')
print('all observed photometric bands:')
print(t_by_filter.groups.keys)
print(t_by_filter.groups[0])

(ax) = plt.subplots(1,1,figsize=(12,6))
plt.ylabel('Flux [mJy]')
plt.xlabel('Epoch [MJD]')
plt.title('data from {}'.format(fin))


for key, group in zip(t_by_filter.groups.keys, t_by_filter.groups):
    print(f'****** {key["Filter"]} *******')
    print(group)
    plt.errorbar(group['MJD'],group['flux(mJy)'],yerr=group['flux_err'],label=key['Filter'],fmt='.')

    print('')

plt.legend()

(tV, tg) = t_by_filter.groups

print(tV)

V_flux_norm = 15.71
g_flux_norm = 11.13


tV['norm'] = tV['flux(mJy)']/V_flux_norm
tg['norm'] = tg['flux(mJy)']/g_flux_norm

tV['normerr'] = tV['flux_err']/V_flux_norm
tg['normerr'] = tg['flux_err']/g_flux_norm

(ax) = plt.subplots(1,1,figsize=(12,6))
plt.ylabel('Flux [mJy]')
plt.xlabel('Epoch [MJD]')
plt.title('data from {}'.format(fin))
plt.errorbar(tV['MJD'],tV['flux(mJy)'],yerr=tV['flux_err'],label='V',fmt='.')
plt.errorbar(tg['MJD'],tg['flux(mJy)'],yerr=tg['flux_err'],label='g',fmt='.')


(ax) = plt.subplots(1,1,figsize=(12,6))
plt.ylabel('Flux [normalised]')
plt.xlabel('Epoch [MJD]')
plt.title('data from {}'.format(fin))
plt.errorbar(tV['MJD'],tV['norm'],yerr=tV['normerr'],label='V',fmt='.',alpha=0.3)
plt.errorbar(tg['MJD'],tg['norm'],yerr=tg['normerr'],label='g',fmt='.',alpha=0.3)


plt.show()
