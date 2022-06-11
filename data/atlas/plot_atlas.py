import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import unique,vstack

from datetime import datetime
from astropy.time import Time

fin='job211831.txt'
t = ascii.read(fin)

print(t)

# MJD          m      dm   uJy   duJy F err chi/N     RA       Dec        x        y     maj  min   phi  apfit mag5sig Sky   Obs
# 58037.635279  13.616  0.003 12984   44 o  0 1718.56 123.84754 -38.98983  1666.50  4612.12 2.32 2.21 -22.2 -0.420 18.66 19.35 02a58037o0713o

# how soon is now?
now = Time(datetime.utcnow(),format='datetime')
print('current MJD is {}'.format(now.mjd))

# reject noisy points
t = t[(t['duJy']<100)]

# reject low flux points
t = t[(t['uJy']>500)]


(ax) = plt.subplots(1,1,figsize=(12,6))
plt.errorbar(t['MJD'],t['uJy'],yerr=t['duJy'],fmt='.')
plt.ylabel('Flux [uJy]')
plt.xlabel('Epoch [MJD]')
plt.title('data from {}'.format(fin))


# get a list of the unique bandpasses
t_by_filter = t.group_by('F')
print('all observed photometric bands:')
print(t_by_filter.groups.keys)
print(t_by_filter.groups[0])

(ax) = plt.subplots(1,1,figsize=(12,6))
plt.ylabel('Flux [uJy]')
plt.xlabel('Epoch [MJD]')
plt.title('data from {}'.format(fin))


for key, group in zip(t_by_filter.groups.keys, t_by_filter.groups):
    print(f'****** {key["F"]} *******')
    print(group)
    plt.errorbar(group['MJD'],group['uJy'],yerr=group['duJy'],label=key['F'],fmt='.')

    print('')

plt.legend()

(tc, to) = t_by_filter.groups

print(tc)

c_flux_norm = 12000
o_flux_norm = 15000


tc['fnorm'] = tc['uJy']/c_flux_norm
to['fnorm'] = to['uJy']/o_flux_norm

tc['fnormerr'] = tc['duJy']/c_flux_norm
to['fnormerr'] = to['duJy']/o_flux_norm

(ax) = plt.subplots(1,1,figsize=(12,6))
plt.ylabel('Flux [uJy]')
plt.xlabel('Epoch [MJD]')
plt.title('data from {}'.format(fin))
plt.errorbar(tc['MJD'],tc['uJy'],yerr=tc['duJy'],label='c',fmt='.')
plt.errorbar(to['MJD'],to['uJy'],yerr=to['duJy'],label='o',fmt='.')


(ax) = plt.subplots(1,1,figsize=(12,6))
plt.ylabel('Flux [normalised]')
plt.xlabel('Epoch [MJD]')
plt.title('Normalised ASASSN data {}'.format(fin))
plt.errorbar(tc['MJD'],tc['fnorm'],yerr=tc['fnormerr'],label='c',fmt='.',alpha=0.3)
plt.errorbar(to['MJD'],to['fnorm'],yerr=to['fnormerr'],label='o',fmt='.',alpha=0.3)

plt.legend()


tn = vstack([tc,to])

tn.write('obs_ATLAS.ecsv',format='ascii.ecsv',overwrite=True)

plt.show()
