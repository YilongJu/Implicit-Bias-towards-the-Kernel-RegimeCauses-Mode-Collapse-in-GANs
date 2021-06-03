# Two dimensional shallow visualizaation
# Break into two panels - 3D surface plot, and 2D (top down) view of breaklines in input space

import torch.utils.data
import matplotlib as mpl
from matplotlib import cm
from numpy import *
from matplotlib.pyplot import *
import torch.nn.functional as F
from itertools import chain
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import ConvexHull, convex_hull_plot_2d

mpl.style.use('seaborn')
cmap = matplotlib.cm.get_cmap('winter')
device = 'cpu' #CHANGE THIS TO 'cpu' if running locally

H = 1000 # 30 for gap task, trying 50 for rand grid, 100 for closer random grid?
NumIters = 25000
PlotFreq = 50 # 500
PrintFreq = 1000

# Assuming a square input space
MinInput = -3.
MaxInput = +3.

learning_rate = 3.0e-5 # 3e-5

numplot = NumIters//PlotFreq + 1

makeVid = True          #Save the Animation File

torch.manual_seed(1802)
np.random.seed(1802)

# Generate Target Fn
# Cases discussed - 2D data gap, 1D function in 2D (e.g. one dimension useless), 20 random points
# 2D data gap
func1 = False
if func1: # 2D Datagap
    ndat = 100
    xin = np.zeros((ndat,2))
    yin = np.zeros((ndat,1))
    for i in range(ndat):
        distance = 0
        while distance < .5:
            x = np.random.rand()*4-2
            y = np.random.rand()*4-2
            distance = np.sqrt(x**2 + y**2)
        xin[i,0] = x
        xin[i,1] = y
        yin[i,0] = 1 - distance
    xin = torch.from_numpy(xin)
    xin = xin.to(torch.float)
    yin = torch.from_numpy(yin)
    yin = yin.to(torch.float)
func2 = False
if func2: # 1D function in 2D input
    ndat = 100
    xin = np.zeros((ndat, 2))
    yin = np.zeros((ndat, 1))
    for i in range(ndat):
        x = np.random.rand() * 4 - 2
        y = np.random.rand() * 4 - 2
        z = x**2 - 2 # function that is one dimensional in input space
        z = (x-y)**2 - 2 # different function that is 1d in input space.
        xin[i, 0] = x
        xin[i, 1] = y
        yin[i, 0] = z
    xin = torch.from_numpy(xin)
    xin = xin.to(torch.float)
    yin = torch.from_numpy(yin)
    yin = yin.to(torch.float)
func3 = False
if func3: # Random, sparse data
    ndat = 25
    xin = np.zeros((ndat, 2))
    yin = np.zeros((ndat, 1))
    for i in range(ndat):
        x = np.random.rand() * 4 - 2
        y = np.random.rand() * 4 - 2
        z = np.random.rand() * 4 - 2
        xin[i, 0] = x
        xin[i, 1] = y
        yin[i, 0] = z
    xin = torch.from_numpy(xin)
    xin = xin.to(torch.float)
    yin = torch.from_numpy(yin)
    yin = yin.to(torch.float)
func4 = True
if func4: # Random data on grid
    ndat = 25
    xin = np.zeros((ndat,2))
    yin = np.zeros((ndat,1))
    for i in range(ndat):
        x = np.mod(i,5)-2
        y = (i//5) - 2
        z = np.random.rand()*4 - 2
        xin[i, 0] = x
        xin[i, 1] = y
        yin[i, 0] = np.random.rand()*4 - 2
    xin = torch.from_numpy(xin)
    xin = xin.to(torch.float)
    yin = torch.from_numpy(yin)
    yin = yin.to(torch.float)

x = torch.as_tensor(np.arange(MinInput, MaxInput + .1, .1), dtype=torch.float)
y = torch.as_tensor(np.arange(MinInput, MaxInput + .1, .1), dtype=torch.float)
X, Y = np.meshgrid(x.data.numpy(), y.data.numpy())



gridin = torch.as_tensor(np.zeros((x.size()[0]*y.size()[0],2)))
for i in range(x.size()[0]):
    for j in range(y.size()[0]):
        gridin[i*x.size()[0] + j][0] = x[i]
        gridin[i*x.size()[0] + j][1] = y[j]
gridin = gridin.to(torch.float)

model = torch.nn.Sequential(
    torch.nn.Linear(2, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, 1)
)

# Modify Model Init Conditions
# Just like in 1D, do case where we fix to uniform BreakLine distribution in theta, r, and then fix breaklines and only allow delta slopes to change
FrozenForcedBP = True
if FrozenForcedBP:
    with torch.no_grad():
        kmax = 201  # of concentric rings # should be odd - from -rmax to rmax including 0
        rmax = 2.5  # farthest ring size
        jmax = 50
        i = 0
        H = kmax * jmax
        model = torch.nn.Sequential(
            torch.nn.Linear(2, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 1)
        )
        for k in range(-(kmax // 2), (kmax // 2) + 1):
            for j in range(jmax):
                model[0].bias[i] = rmax * (k / (kmax // 2))
                theta = -np.pi + 2 * j * (np.pi / jmax)
                model[0].weight[i, 0] = np.sin(theta)
                model[0].weight[i, 1] = np.cos(theta)
                i += 1
        #model[0].weight.requires_grad = False
        #model[0].bias.requires_grad = False
        model[2].weight[:] *= .1

# Are the initial thetas actually uniformly distributed? Shouldn't be - pytroch init is uniform, so biased away from cardinal directions
#torch.sort(torch.atan(model[0].weight[:,0]/model[0].weight[:,1]))[0]

Zstore = np.zeros((numplot,x.size()[0],y.size()[0]))
BPstore = np.zeros((numplot,H,4))
lstore = np.zeros(numplot)

# Modify alpha
alpha = .1
if alpha!=1:
    with torch.no_grad():
        model[0].weight[:]*=alpha
        model[0].bias[:]*=alpha
        model[2].weight[:]/=alpha

loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(NumIters + 1):
    out_pred = model(xin)
    loss = loss_fn(out_pred, yin)
    if t % PrintFreq == 0:
        print(t, loss.item())
    if np.mod(t, PlotFreq) == 0:
        with torch.no_grad():
            zgrid = model(gridin)
            Z = zgrid.data.numpy().reshape(X.shape)
            # Store Z - Zstd[int(t / plotfreq), :, :] = Z
            Zstore[t//PlotFreq,:,:] = Z
            # Store Breakplane information - e.g. normal vector and offset, = w/norm(w), b/norm(w)
            BPstore[t//PlotFreq,:,0:2] = torch.t(torch.t(model[0].weight)/torch.norm(model[0].weight,dim = 1))
            BPstore[t//PlotFreq,:,2] = -model[0].bias/torch.norm(model[0].weight,dim = 1)
            # Might as well store delta slope information as well!
            BPstore[t//PlotFreq,:,3] = torch.norm(model[0].weight,dim = 1)*model[2].weight.data
            lstore[t//PlotFreq] = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# 2D RBF 'smoothing spline' fit to compare against
from scipy.interpolate import Rbf
rbfi = Rbf(xin[:,1],xin[:,0], yin[:,0], function = 'thin_plate') # Reverse spline x/y here
# Generate x, y test points
di = rbfi(X,Y)

# Alex Williams help fn to plot colored lines
# from https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
from matplotlib.collections import LineCollection
def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings
    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection
    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)
    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

# Generate data for plot lines given normal, offset e.g. w*x - b = 0
# Which has point nearest to x-axis at x = w*b
# And then further out perpendicular to w = inv(w) * [-1,1] or [1,-1]
# Calculate edges w.r.t. MinInput,MaxInput, distances are abs(wb - (min or max, depending on sign of perp))
# Effective distances are distances / perp, take the smallest effective distance, use wb + (smallest dist) * perp
# Repeat for opposite perp, giving us two points
# ... or, can we just plot point OUTSIDE xmin/xmas, and the line will still show? Tested, yes it will
# e.g. just use w*b +\- k * perp, where k is moderately large e.g. maxinput-mininput
linepts = np.zeros((numplot,H,2,2)) # Time x Units x [start,end] x [x,y]
startpts = BPstore[:,:,0:2] * np.expand_dims(BPstore[:,:,2],2) # 501x30x2 of the starting location w*b
startptsnorm = np.copy(startpts)
startptsnorm[:,:,0] = startptsnorm[:,:,0]/(np.linalg.norm(startpts,axis = 2))*np.sign(BPstore[:,:,2])
startptsnorm[:,:,1] = startptsnorm[:,:,1]/(np.linalg.norm(startpts,axis = 2))*np.sign(BPstore[:,:,2])
perps = 1/BPstore[:,:,0:2] * np.array([-1,1]) # 501x30x2 of the perpendicular directions to go
endpts1 = startpts + (MaxInput - MinInput)*perps #501x30x2, where last is x/y
endpts2 = startpts - (MaxInput - MinInput)*perps # ^
linepts[:,:,0,:] = endpts1
linepts[:,:,1,:] = endpts2
linepts2 = np.copy(linepts)
linepts2[:,:,0,:] +=.05*startptsnorm#/startptsnorm[:,:,:]
linepts2[:,:,1,:] +=.05*startptsnorm#/startptsnorm[:,:,:]
# Helper for color
coloruse = np.copy(BPstore[:,:,3])
# Get convex hull
hull = ConvexHull(xin.numpy())

# Generate Output Animation a 2 panel version
mpl.rcParams['agg.path.chunksize'] = 10000*10 # Recommended to fix error, but just terminates drawing partway through...
fig = plt.figure()
fig.suptitle('H = '+str(H)+' hidden units, Initialization Scale $\\alpha$ = '+str(alpha)+', T: '+str(0)+', Loss = {0:4f}'.format(lstore[0]))

ax = fig.add_subplot(221, projection='3d')
plot = ax.plot_surface(X,Y,Zstore[0,:,:])
ax.scatter(xin[:,1], xin[:,0], yin[:,0], c='red', s=20)
ax.set_zlim(-2, 2)

ax2 = fig.add_subplot(222, projection='3d')
plot2 = ax2.plot_surface(X,Y,Zstore[0,:,:])
ax2.scatter(xin[:,1], xin[:,0], yin[:,0], c='red', s=20)
ax2.set_zlim(-2, 2)
ax2.view_init(azim=45)

ax3 = fig.add_subplot(223)
plot3 = ax3.plot(linepts[0,0,:,0],linepts[0,0,:,1])
for i in range(1,H):
    ax3.plot(linepts[0,i,:,0],linepts[0,i,:,1])
ax3.set_xlim([MinInput,MaxInput])
ax3.set_ylim([MinInput,MaxInput])

ax4 = fig.add_subplot(224)
c = ax4.pcolorfast(X, Y, Zstore[0, :-1, :-1] - di[:-1,:-1], cmap='RdBu', vmin = -2, vmax = 2)
ax4.plot(xin.numpy()[hull.vertices,0], xin.numpy()[hull.vertices,1], 'r--', lw=2)
#fig.colorbar(c, ax=ax4)



def update(frame,Zstd,plotin):
    fig.suptitle('H = ' + str(H) + ' hidden units, Initialization Scale $\\alpha$ = ' + str(alpha) + ', T: ' + str(
        frame*PlotFreq) + ', Loss = {0:4f}'.format(lstore[frame]))
    ax.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    plotin = ax.plot_surface(X,Y,Zstore[frame,:,:], cmap = cm.coolwarm)
    ax.scatter(xin[:, 1], xin[:, 0], yin[:, 0], c='red', s=20)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.plot_wireframe(X, Y, di, color='r', alpha=.1)

    plotin2 = ax2.plot_surface(X,Y,Zstore[frame,:,:], cmap = cm.coolwarm)
    ax2.scatter(xin[:, 1], xin[:, 0], yin[:, 0], c='red', s=20)
    ax2.set_zlim(-2, 2)
    ax2.view_init(azim=45)
    ax2.set_xlabel('X')
    ax2.plot_wireframe(X,Y,di,color = 'r', alpha = .1)

    plotin3 = ax3.scatter(MinInput-1,MinInput-1)
    #plotin3 = ax3.plot(linepts[frame, 0, :, 0], linepts[frame, 0, :, 1])
    #for i in range(1, H):
    #    ax3.plot(linepts[frame, i, :, 0], linepts[frame, i, :, 1])
    #lc = multiline(linepts[frame, :, :, 1], linepts[frame, :, :, 0], coloruse[frame,:], ax=ax3, cmap='bwr', lw = 1)
    #lc2 = multiline(linepts2[frame, :, :, 1], linepts[frame, :, :, 0], coloruse[frame, :], ax=ax3, cmap='bwr', lw=1, linestyle = ':')
    ax3.set_xlim([MinInput, MaxInput])
    ax3.set_ylim([MinInput, MaxInput])

    plotin4 = ax4.pcolorfast(X, Y, Zstore[frame, :-1, :-1] - di[:-1,:-1], cmap='RdBu', vmin = -2, vmax = 2)
    #fig.colorbar(plotin4, ax=ax4)
    ax4.plot(xin.numpy()[hull.vertices, 0], xin.numpy()[hull.vertices, 1], 'r--', lw=2)

    #plotin4 = ax4.contourf(X,Y,Zstore[frame,:,:], cmap = 'RdBu')
    #fig.colorbar(plotin4,ax = ax4)



    return plotin,
ani = animation.FuncAnimation(fig, update, numplot, fargs = (Zstore,plot))


ani.save('.\Pytorch\Viz2D\Viz2D.mp4',writer='ffmpeg',fps=20)

# Cant get the colorbar to correctly show on a subplot. Instead, try making a seperate plot for the top down view of fit -
# Even a single panel wouldnt work - animation method appears to be incompatible with colorbars...
# Instead, based on https://brushingupscience.com/2016/06/21/matplotlib-animations-the-easy-way/
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots(figsize=(5, 3))
ax.set(xlim=(MinInput, MaxInput), ylim=(MinInput, MaxInput))

cax = ax.pcolormesh(X, Y, Zstore[0, :-1, :-1]- di[:-1,:-1], cmap='RdBu', vmin = -2, vmax = 2)
fig.colorbar(cax)
def animate(i):
    cax.set_array((Zstore[i, :-1, :-1]- di[:-1,:-1]).flatten())
anim = FuncAnimation(fig,animate,interval = 50, frames = numplot-1)
plt.draw()
anim.save(('.\Pytorch\Viz2D\Viz2D_Delta.mp4'))

# The original (1st) plot output fails for the RndGrd case (e.g. 50 lines, 25000 timesteps with plotfreq of 50) due to on overflow error in the animation generator
# Try reproductig that plot using the method used for the 2nd gif
# Completely fails - generates one, non updating plot....
if (False):
    fig, axs = plt.subplots(2, 2, figsize = (6,4))
    axs[1,0].set(xlim=(MinInput, MaxInput), ylim=(MinInput, MaxInput))
    fig.suptitle('H = '+str(H)+' hidden units, Initialization Scale $\\alpha$ = '+str(alpha)+', T: '+str(0)+', Loss = {0:4f}'.format(lstore[0]))


    axs[0,0] = Axes3D(fig)
    plot = axs[0,0].plot_surface(X,Y,Zstore[0,:,:])
    plot_2 = axs[0,0].scatter(xin[:,1], xin[:,0], yin[:,0], c='red', s=20)
    axs[0,0].set_zlim(-2, 2)

    axs[0,1] = Axes3D(fig)
    plot2 = axs[0,1].plot_surface(X,Y,Zstore[0,:,:])
    plot2_2 = axs[0,1].scatter(xin[:,1], xin[:,0], yin[:,0], c='red', s=20)
    axs[0,1].set_zlim(-2, 2)
    axs[0,1].view_init(azim=45)

    plot3 = axs[1,0].plot(linepts[0,0,:,0],linepts[0,0,:,1])
    for i in range(1,H):
        axs[1,0].plot(linepts[0,i,:,0],linepts[0,i,:,1])
    axs[1,0].set_xlim([MinInput,MaxInput])
    axs[1,0].set_ylim([MinInput,MaxInput])

    c = axs[1,1].pcolorfast(X, Y, Zstore[0, :-1, :-1] - di[:-1,:-1], cmap='RdBu', vmin = -2, vmax = 2)
    fig.colorbar(c, ax = axs[1,1])

    def animate2(i):
        fig.suptitle('H = ' + str(H) + ' hidden units, Initialization Scale $\\alpha$ = ' + str(alpha) + ', T: ' + str(
            i*PlotFreq) + ', Loss = {0:4f}'.format(lstore[i]))
        plot.set_array((Zstore[i,:,:]).flatten())
        plot2.set_array((Zstore[i,:,:]).flatten())
        # How to do 3??
        c.set_array((Zstore[i, :-1, :-1]- di[:-1,:-1]).flatten())
    anim = FuncAnimation(fig,animate2,interval = 50, frames = numplot-1)
    plt.draw()
    anim.save(('.\Pytorch\Viz2D\Viz2D_Deltav2.mp4'))