import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import autograd.numpy as npa
from autograd import grad, value_and_grad

import legume
from legume.minimize import Minimize

# Number of PhC periods in x and y directions
Nx, Ny = 20, 14

#Target Q-factor
Q0 = 10000

# Regular PhC parameters
a = 430
ra = 0.26
dslab = 250/a                # thickness slab 250nm/a
n_slab = 3.1649              # refr. index slab

NA = 0.14                    # fiber NA

n_l = 1.4682   #refr. index lower cladding
#n_l= 1.
n_u = 1.    #refr. index upper cladding
#n_u = 1.4682
n_holes=1.  #refr.  index holes

# Initialize a lattice and PhC
lattice = legume.Lattice([Nx, 0], [0, Ny*np.sqrt(3)/2])

# Make x and y positions in one quadrant of the supercell
# We only initialize one quadrant because we want to shift the holes symmetrically
xp, yp = [], []
ny = Ny // 2 + 1
for iy in range(ny):
    nx = Nx // 2 + 1
    if iy%2 == 1:
        nx = Nx // 2
    for ix in range(nx):
        if iy == 0:
            if ix == 0 or ix == 1:                  # Skip first two holes to create an L3 cavity
                continue



        xp.append(ix + (iy % 2) * 0.5)
        yp.append(iy*np.sqrt(3)/2)


nx = Nx // 2 + 1
nc = len(xp)

# Initialize shift parameters to zeros
dx, dy = np.zeros((nc,)), np.zeros((nc,))

# Define L3 PhC cavity with shifted holes
def cavity(dx, dy):
    # Initialize PhC
    phc = legume.PhotCryst(lattice, eps_l=n_l**2, eps_u=n_u**2)

    # Add a layer to the PhC
    phc.add_layer(d=dslab, eps_b=n_slab**2)

    # Apply holes symmetrically in the four quadrants
    for ic, x in enumerate(xp):
        yc = yp[ic] if yp[ic] == 0 else yp[ic] + dy[ic]
        xc = x if x == 0 else xp[ic] + dx[ic]
        phc.add_shape(legume.Circle(x_cent=xc, y_cent=yc, r=ra))
        if nx-0.6 > xp[ic] > 0 and (ny-1.1)*np.sqrt(3)/2 > yp[ic] > 0:
            phc.add_shape(legume.Circle(x_cent=-xc, y_cent=-yc, r=ra))
        if nx-1.1 > xp[ic] > 0:
            phc.add_shape(legume.Circle(x_cent=-xc, y_cent=yc, r=ra))
        if (ny-1.1)*np.sqrt(3)/2 > yp[ic] > 0 and nx-1.1 > xp[ic]:
            phc.add_shape(legume.Circle(x_cent=xc, y_cent=-yc, r=ra))

    return phc

# Solve for a cavity defined by shifts dx, dy
def gme_cavity_k(dx, dy, gmax, options, kpoints, f_lb=0.27):
    phc = cavity(dx, dy)


    options['compute_im'] = False
    gme = legume.GuidedModeExp(phc, gmax=gmax)
    # Solve for the real part of the frequencies
    gme.run(kpoints=np.array([[0], [0]]), **options)
    indmode=Nx*Ny-3

    # Find the imaginary frequency of the fundamental cavity mode, as well as the radcoup and radgvec data
    (freq_im, radcoupmode, radgvecmode) = gme.compute_rad(0, [indmode])
    fims = [freq_im]

    return (gme, npa.array(fims), radcoupmode,radgvecmode, indmode)



# Set some GME options
options = {'gmode_inds': [0],
           'verbose': True,
           'gradients': 'approx',
           'numeig': Nx*Ny+1,       # get Nx*Ny+1 eigenvalues
           'eps_eff': 'average',

           'eig_solver': 'eigh'
          }
gmax = 2.

# Create a grid in k space (non-negative kx, ky only)
nkx = 1
nky = 1
kx = np.linspace(0, 2*np.pi/Nx, nkx)
ky = np.linspace(0, 2*np.pi/Ny/np.sqrt(3)*2, nky)
kxg, kyg = np.meshgrid(kx, ky)
kxg = kxg.ravel()
kyg = kyg.ravel()

# Run the simulation for the starting cavity (zero shifts as initialized above)
(gme, fims,radcoupmode,radgvecmode, indmode) = gme_cavity_k(dx, dy, gmax=gmax, options=options, kpoints=np.vstack((kxg, kyg)))

# We can also visualize the cavity and the mode profile of the fundamental mode
ax = legume.viz.field(gme, 'e', 0, indmode, z=dslab/2, component='y', val='re', N1=300, N2=300)
plt.show()


# To compute gradients, we need to set the `legume` backend to 'autograd'
legume.set_backend('autograd')

# Starting parameters
pstart = np.zeros((2*nc, ))

def EfficiencyCalc(radcoupmode,radgvecmode,NA=NA):           # Integrate the radcoup data over the NA
    dl_te = radcoupmode['l_te'][0]
    kxky_dl_te = radgvecmode['l'][0]
    kx = kxky_dl_te[0]
    ky = kxky_dl_te[1]
    kxky_dl_te = tuple(zip(kx, ky))
    NAkval = [index for index, kxky in enumerate(kxky_dl_te) if
              np.sqrt(np.square(kxky[0]) + np.square(kxky[1])) < NA/n_l * np.sqrt(
                  np.square(np.max(kx)) + np.square(np.max(ky)))]
    NAcoup_dl_te = dl_te[NAkval]
    dl_te_gamma = npa.sum(npa.square(np.abs(NAcoup_dl_te)))
    dl_te_tot = npa.sum(npa.square(np.abs(dl_te)))


    du_te = radcoupmode['u_te'][0]
    kxky_du_te = radgvecmode['u'][0]
    kx = kxky_du_te[0]
    ky = kxky_du_te[1]
    kxky_du_te = tuple(zip(kx, ky))
    NAkval = [index for index, kxky in enumerate(kxky_du_te) if
              np.sqrt(np.square(kxky[0]) + np.square(kxky[1])) < NA/n_u * np.sqrt(
                  np.square(np.max(kx)) + np.square(np.max(ky)))]
    NAcoup_du_te = du_te[NAkval]
    du_te_gamma = npa.sum(npa.square(np.abs(NAcoup_du_te)))
    du_te_tot = npa.sum(npa.square(np.abs(du_te)))

    dl_tm = radcoupmode['l_tm'][0]
    kxky_dl_tm = radgvecmode['l'][0]
    kx = kxky_dl_tm[0]
    ky = kxky_dl_tm[1]
    kxky_dl_tm = tuple(zip(kx, ky))
    NAkval = [index for index, kxky in enumerate(kxky_dl_tm) if
              np.sqrt(np.square(kxky[0]) + np.square(kxky[1])) < NA/n_l * np.sqrt(
                  np.square(np.max(kx)) + np.square(np.max(ky)))]
    NAcoup_dl_tm = dl_tm[NAkval]
    dl_tm_gamma = npa.sum(npa.square(np.abs(NAcoup_dl_tm)))
    dl_tm_tot = npa.sum(npa.square(np.abs(dl_tm)))

    du_tm = radcoupmode['u_tm'][0]
    kxky_du_tm = radgvecmode['u'][0]
    kx = kxky_du_tm[0]
    ky = kxky_du_tm[1]
    kxky_du_tm = tuple(zip(kx, ky))
    NAkval = [index for index, kxky in enumerate(kxky_du_tm) if
              np.sqrt(np.square(kxky[0]) + np.square(kxky[1])) < NA/n_u * np.sqrt(
                  np.square(np.max(kx)) + np.square(np.max(ky)))]
    NAcoup_du_tm = du_tm[NAkval]
    du_tm_gamma = npa.sum(npa.square(np.abs(NAcoup_du_tm)))
    du_tm_tot = npa.sum(npa.square(np.abs(du_tm)))
    return (dl_te_gamma, dl_te_tot, du_te_gamma, du_te_tot, dl_tm_gamma, dl_tm_tot, du_tm_gamma, du_tm_tot)

def of_total(params):
    dx = params[0:nc]
    dy = params[nc:]
    (gme, fims,radcoupmode,radgvecmode, indmode) = gme_cavity_k(dx, dy, gmax, options, np.vstack((kxg, kyg)))
    Nk = gme.kpoints[0, :].size

    Q = gme.freqs[0, indmode]/2/fims[0]

    (dl_te_gamma, dl_te_tot, du_te_gamma, du_te_tot, dl_tm_gamma, dl_tm_tot, du_tm_gamma, du_tm_tot)=EfficiencyCalc(radcoupmode,radgvecmode,NA)

    gamma_loss= dl_te_gamma+du_te_gamma+dl_tm_gamma+du_tm_gamma
    total_loss = dl_te_tot+du_te_tot+dl_tm_tot+du_tm_tot
    gamma_loss_fiber = dl_te_gamma+dl_tm_gamma

    total_loss_fiber = dl_te_tot+dl_tm_tot
    fiberair_loss_ratio = (dl_te_tot+dl_tm_tot)/(du_te_tot+du_tm_tot)

    Ey = gme.get_field_xy('e', 0, indmode, z=dslab/2, component='y', Nx=3, Ny=3)[0]['y'][1][1]
    modevol = npa.square(npa.abs(Ey))
    Q_ratio=npa.arctan(Q/Q0)
    #crit_coup_ratio=npa.arctan(gamma_loss/(total_loss-gamma_loss))
    crit_coup_ratio = gamma_loss_fiber / (total_loss-gamma_loss_fiber)
    #crit_coup_ratio = gamma_loss / (total_loss - gamma_loss)
    # We put a negative sign because we use in-built methods to *minimize* the objective function
    return -Q_ratio*crit_coup_ratio*modevol

# The autograd function `value_and_grad` returns simultaneously the objective value and the gradient
obj_grad = value_and_grad(of_total)



options['verbose'] = False


# Initialize an optimization object
opt = Minimize(of_total)

# Starting parameters are the un-modified cavity
pstart = np.zeros((2*nc, ))

# Run an 'adam' optimization
(p_opt, ofs) = opt.adam(pstart, step_size=0.005, Nepochs=25, bounds = [-0.25, 0.25])
# Optimized parameters
dx = p_opt[0:nc]
dy = p_opt[nc:]

# Run the simulation
(gme, fims,radcoupmode,radgvecmode, indmode) = gme_cavity_k(dx, dy, gmax, options, np.vstack((kxg, kyg)))
(dl_te_gamma, dl_te_tot, du_te_gamma, du_te_tot, dl_tm_gamma, dl_tm_tot, du_tm_gamma, du_tm_tot) = EfficiencyCalc(radcoupmode,radgvecmode, NA)
Ey = gme.get_field_xy('e', 0, indmode, z=dslab/2, component='y', Nx=3, Ny=3)[0]['y'][1][1]
modevol = npa.square(npa.abs(Ey))
print("Quality factor averaged over %d k-points:  %1.2f" %(nkx*nky, gme.freqs[0, indmode]/2/np.average(fims)))
print("Loss into the fiber NA divided by the total loss in fiber lightcone:  %1.6f" %((dl_te_gamma+dl_tm_gamma)/(dl_te_tot+dl_tm_tot)))
print("Loss into the fiber NA divided by the total loss on both sides:  %1.6f" %((dl_te_gamma+dl_tm_gamma)/(dl_te_tot+dl_tm_tot+du_tm_tot+du_te_tot)))
print("Critical coupling ratio K1/(K-K1):  %1.6f" %((dl_te_gamma+dl_tm_gamma)/((dl_te_tot+dl_tm_tot+du_te_tot+du_tm_tot)-(dl_te_gamma+dl_tm_gamma))))
print("Max of EY^2 at the center of the unit cell: %1.6f" % modevol )
ax = legume.viz.field(gme, 'e', 0, indmode, z=dslab/2, component='y', val='re', N1=500, N2=500)

j=1
with open("dxdata.txt", "w") as outfile:
    for dxiter in range(len(dx)):
        linex = 'dx' + str(j) + ' ' + str(dx[dxiter]) + ' ""\n'
        j = j+1

        outfile.write(linex)

k=1
with open("dydata.txt", "w") as outfile2:
    for dyiter in range(len(dy)):
        liney = 'dy' + str(k) + ' ' + str(dy[dyiter]) + ' ""\n'
        k = k+1

        outfile2.write(liney)

for l in range(Ny//2):
    #plot y lines
    plt.plot([-10, 10], [-ra + l*np.sqrt(3) / 2, -ra + l*np.sqrt(3) / 2], 'k-', linewidth=0.3)
    plt.plot([-10, 10], [ra + l*np.sqrt(3) / 2, ra + l*np.sqrt(3) / 2], 'k-', linewidth=0.3)
    plt.plot([-10, 10], [-ra - l*np.sqrt(3) / 2, -ra - l*np.sqrt(3) / 2], 'k-', linewidth=0.3)
    plt.plot([-10, 10], [ra - l*np.sqrt(3) / 2, ra - l*np.sqrt(3) / 2], 'k-', linewidth=0.3)


for l in range(Nx//2):
    #plot x lines
    plt.plot([-ra+l, -ra+l], [-6 , 6], 'k-', linewidth=0.3)
    plt.plot([ra + l, ra + l], [-6, 6], 'k-', linewidth=0.3)
    plt.plot([-ra - l, -ra - l], [-6, 6], 'k-', linewidth=0.3)
    plt.plot([ra - l, ra - l], [-6, 6], 'k-', linewidth=0.3)





ax = legume.viz.field(gme, 'e', 0, indmode, z=dslab/2, component='y', val='re', N1=500, N2=500)

plt.show()
