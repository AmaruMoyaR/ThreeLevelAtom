import numpy as np
from qutip import * 
import qutip as q
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import matplotlib.cm as cm
import datetime as dt
from tqdm import tqdm

# xvec = np.linspace(-6, 6, 1500)
def wigner_cmap(W, levels=1024, shift=-0.01, max_color='#09224F',
                mid_color='#FFFFFF', min_color='#530017',
                neg_color='#FF97D4', invert=False):
    """A custom colormap that emphasizes negative values by creating a
    nonlinear colormap.

    Parameters
    ----------
    W : array
        Wigner function array, or any array.
    levels : int
        Number of color levels to create.
    shift : float
        Shifts the value at which Wigner elements are emphasized.
        This parameter should typically be negative and small (i.e -1e-5).
    max_color : str
        String for color corresponding to maximum value of data.  Accepts
        any string format compatible with the Matplotlib.colors.ColorConverter.
    mid_color : str
        Color corresponding to zero values.  Accepts any string format
        compatible with the Matplotlib.colors.ColorConverter.
    min_color : str
        Color corresponding to minimum data values.  Accepts any string format
        compatible with the Matplotlib.colors.ColorConverter.
    neg_color : str
        Color that starts highlighting negative values.  Accepts any string
        format compatible with the Matplotlib.colors.ColorConverter.
    invert : bool
        Invert the color scheme for negative values so that smaller negative
        values have darker color.

    Returns
    -------
    Returns a Matplotlib colormap instance for use in plotting.

    Notes
    -----
    The 'shift' parameter allows you to vary where the colormap begins
    to highlight negative colors. This is beneficial in cases where there
    are small negative Wigner elements due to numerical round-off and/or
    truncation.

    """
    cc = matplotlib.colors.ColorConverter()
    max_color = np.array(cc.to_rgba(max_color), dtype=float)
    mid_color = np.array(cc.to_rgba(mid_color), dtype=float)
    if invert:
        min_color = np.array(cc.to_rgba(neg_color), dtype=float)
        neg_color = np.array(cc.to_rgba(min_color), dtype=float)
    else:
        min_color = np.array(cc.to_rgba(min_color), dtype=float)
        neg_color = np.array(cc.to_rgba(neg_color), dtype=float)
    # get min and max values from Wigner function
    
    # nrm = matplotlib.colors.Normalize(-W.max(), W.max())
    bounds = [-W.max(), W.max()]
    # create empty array for RGBA colors
    adjust_RGBA = np.hstack((np.zeros((levels, 3)), np.ones((levels, 1))))
    zero_pos = int(np.round(levels * np.abs(shift - bounds[0])
                        / (bounds[1] - bounds[0])))
    num_pos = levels - zero_pos
    num_neg = zero_pos - 1
    # set zero values to mid_color
    adjust_RGBA[zero_pos] = mid_color
    # interpolate colors
    for k in range(0, levels):
        if k < zero_pos:
            interp = k / (num_neg + 1.0)
            adjust_RGBA[k][0:3] = (1.0 - interp) * \
                min_color[0:3] + interp * neg_color[0:3]
        elif k > zero_pos:
            interp = (k - zero_pos) / (num_pos + 1.0)
            adjust_RGBA[k][0:3] = (1.0 - interp) * \
                mid_color[0:3] + interp * max_color[0:3]
    # create colormap
    wig_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('wigner_cmap',
                                                            adjust_RGBA,
                                                            N=levels)
    return wig_cmap


def plot_wigner(rho, fig=None, ax=None, figsize=(6, 6),
                cmap=None, alpha_max=7.5, colorbar=False,
                method='clenshaw', projection='2d'):
    """
    Plot the the Wigner function for a density matrix (or ket) that describes
    an oscillator mode.

    Parameters
    ----------
    rho : :class:`qutip.Qobj`
        The density matrix (or ket) of the state to visualize.

    fig : a matplotlib Figure instance
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.

    figsize : (width, height)
        The size of the matplotlib figure (in inches) if it is to be created
        (that is, if no 'fig' and 'ax' arguments are passed).

    cmap : a matplotlib cmap instance
        The colormap.

    alpha_max : float
        The span of the x and y coordinates (both [-alpha_max, alpha_max]).

    colorbar : bool
        Whether (True) or not (False) a colorbar should be attached to the
        Wigner function graph.

    method : string {'clenshaw', 'iterative', 'laguerre', 'fft'}
        The method used for calculating the wigner function. See the
        documentation for qutip.wigner for details.

    projection: string {'2d', '3d'}
        Specify whether the Wigner function is to be plotted as a
        contour graph ('2d') or surface plot ('3d').

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.
    """

    if not fig and not ax:
        if projection == '2d':
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        elif projection == '3d':
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        else:
            raise ValueError('Unexpected value of projection keyword argument')

    if isket(rho):
        rho = ket2dm(rho)

    xvec = np.linspace(-alpha_max, alpha_max, 200)
    W0 = wigner(rho, xvec, xvec, method=method)

    W, yvec = W0 if isinstance(W0, tuple) else (W0, xvec)

    wlim = abs(W).max()

    if cmap is None:
        cmap = cm.get_cmap('RdBu')

    if projection == '2d':
        cf = ax.contourf(xvec, yvec, W, 100,
                         norm=matplotlib.colors.Normalize(-wlim, wlim), cmap=cmap)
    elif projection == '3d':
        X, Y = np.meshgrid(xvec, xvec)
        cf = ax.plot_surface(X, Y, W0, rstride=5, cstride=5, linewidth=0.5,
                             norm=matplotlib.colors.Normalize(-wlim, wlim), cmap=cmap)
    else:
        raise ValueError('Unexpected value of projection keyword argument.')

    if xvec is not yvec:
        ax.set_ylim(xvec.min(), xvec.max())

    ax.set_xlabel(r'$\rm{Re}(\alpha)$', fontsize=12)
    ax.set_ylabel(r'$\rm{Im}(\alpha)$', fontsize=12)

    if colorbar:
        fig.colorbar(cf, ax=ax)

    # ax.set_title("Wigner function", fontsize=12)

    return fig, ax


def plot_wigner3d(psi,xvec):
    W = wigner(psi, xvec, xvec)
    wmap = wigner_cmap(W)  # Generate Wigner colormap
    
    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(1, 2, 1)        
    plot_wigner(psi, fig=fig, ax=ax, alpha_max=5.5, cmap=wmap, colorbar = True);

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    plot_wigner(psi, fig=fig, ax=ax, projection='3d', alpha_max=5.5, cmap=wmap);
    # plt.close(fig)
    return fig


def figuremaker(psi_list, xvec, n=None, m =None):
    
    W_list = [wigner(psi, xvec, xvec) for psi in psi_list] #list of states
    wmap_list = [wigner_cmap(W) for W in W_list] #map for colors of 
    # n   # Number of columns ; m   # Number of rows
    fig, axes = plt.subplots(n, m, figsize=(6, 8))  # Create the figure and subplots
    fig.patch.set_visible(False)
    # 
    # Iterate over the rows and columns
    for i in range(n):
        for j in range(m):
            # Customize the plotting based on the position of the subplot
            if j == 0:
                # for w in range(len(psi_list)):
                ax = axes[i, j]
                ax.axis('off')
                ax = fig.add_subplot(n,m,i*m+j+1)   
                plot_wigner(psi_list[i], fig=fig, ax=ax, alpha_max=5.5, cmap=wmap_list[i], colorbar = True);
                # 
                # fig.tight_layout()
            
            elif j == 1:
                # for w in range(len(psi_list)):
                ax = axes[i, j]
                ax.axis('off')
                ax = fig.add_subplot(n, m, i*m+j+1, projection='3d')
                plot_wigner(psi_list[i], fig=fig, ax=ax, projection='3d', alpha_max=5.5, cmap=wmap_list[i]);
                # ax.axis('off')
                # fig.tight_layout()
                
    # plt.close(fig)
    # Adjust the spacing between subplots
    fig.tight_layout()
    return fig
    
    
    
def Hamiltonian_2levels(psi_inic, N , t = 300, w0 = 1. , w = 1., lamda = 1):
    #Constantes del sistema
    # w0 = 1
    # w = 1
    # lamda = 1
    #Creamos los operadores para el hamiltoniano
    sig_atomo = q.tensor(q.destroy(2),q.identity(N)) #producto tensorial entre sigma 2 y la identidad N
    a_campo = q.tensor(q.identity(2),q.destroy(N)) #producto tensorial entre la identidad 2 y operador destruccion N
    sig_z = q.tensor(q.sigmaz(),q.identity(N)) #producto tensorial entre sigma z y la identidad N
    #Creamos el hamiltoniano para la evolución temporal
    H_int = 0.5 * w0 * sig_z + w*a_campo.dag()*a_campo + lamda*(sig_atomo.dag()*a_campo + sig_atomo*a_campo.dag())
    
    #estado basal de 2 niveles
    g_atomo = q.basis(2,0)#ground state
    e_atomo = q.basis(2,1)#excited state
    estado_e = q.tensor(e_atomo,psi_inic)
    estado_g = q.tensor(g_atomo,psi_inic)
    
    tiempo = np.linspace(0,25*lamda/w,t)
    #Dado el estado inicial estado_g, la evolución se calcula usando mesolve
    estado_final = q.mesolve(H_int,estado_g,tiempo)
    evolucion_temporal_estado = estado_final.states
    tasa_inversion = q.expect(sig_z, evolucion_temporal_estado)
    
    return evolucion_temporal_estado, tiempo, tasa_inversion
    
# def PopulationInv(sig_z,estado): #calcula tasa de inversion
#     tasa_inversion = q.expect(sig_z,estado)
#     return tasa_inversion


def Plot_Population(tiempo, tasa_inversion, color):
    #Ploteamos la tasa de inversion en el tiempo
    fig, ax = plt.subplots()
    ax.plot(tiempo, tasa_inversion , c = str(color))
    ax.set_xlabel(r'$\lambda t / \omega $')
    ax.set_ylabel('W(t)')
    ax.set_title('Tasa de inversión estado')
    plt.show()
    
    
def WignerEvolution(State_in_time,xvec, listw): #esta se demora mas q yo en licenciarme
    
    for i in range(len(State_in_time)):
        listw.append(q.wigner(State_in_time[i],xvec,xvec))    
    return listw


def WignerEvolutionbar(State_in_time,xvec, listw): #esta se demora mas q yo en licenciarme
    for i in tqdm(range(len(State_in_time))):
        listw.append(q.wigner(State_in_time[i],xvec,xvec))    
    return listw


def VonEntropy(State,t):
    # rho = q.ket2dm(State)
    Entropy = [q.entropy_vn(State) for i in range(len(t))]
    return Entropy 


def AnimatedWigner(Wigners, xvec, savename):
        
        W = np.array(Wigners)
        wlim = abs(W).max()
        levels = np.linspace(-wlim, wlim, 30)
        cmap = cm.get_cmap('seismic_r')

        kw = dict(levels=levels, cmap=cmap, vmin=-wlim, vmax=wlim, origin='lower')
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax = fig.add_subplot(111, aspect='equal') #, autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10)
        cf = ax.contourf(xvec, xvec, W[0], **kw)
        cbar = fig.colorbar(cf, ax=ax)

        def update(frame):
            fig.clear()
            ax = fig.add_subplot(111, aspect='equal') #, autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10)
            cf = ax.contourf(xvec, xvec, W[frame], **kw)#100, norm=plt.Normalize(-wlim, wlim), cmap='RdPu')
            ax.set_xlabel(r'$\rm{Re}(\alpha)$', fontsize=12)
            ax.set_ylabel(r'$\rm{Im}(\alpha)$', fontsize=12)
            current_time = frame*0.1
            ax.set_title("Wigner function (Time: {:.1f} s)".format(current_time), fontsize=12)
            fig.colorbar(cf, ax=ax)

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=W.shape[0], repeat=False)
        writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Yo'), bitrate=-1)
        ani.save(savename + '.gif', writer=writer)
        plt.draw()
        plt.show()

# animation_name = input("Enter animation name: ")  # Prompt for animation name

def AnimatePopulation(tasa_inversion, tiempo, savename, color = 'dodgerblue'):
    # Create the figure and subplots
    fig = plt.figure(figsize=(6, 6))
    
    # Subplot 1: Population Inversion
    ax = fig.add_subplot(111)
    line, = ax.plot([], [], c=str(color))
    ax.set_xlabel(r'$\lambda t / \omega $')
    ax.set_ylabel('W(t)')
    ax.set_title('Tasa de inversión estado comprimido coherente')

    # Initialize the line data
    line.set_data([], [])

    # Update function
    def update(frame):
        ax.clear()
        # Update Wigner function subplot
        # Update Tasa de inversión subplot
        ax.scatter(x=tiempo[frame],y=tasa_inversion[frame], c='red', marker='o')
        current_time = frame * 0.1        
        line, = ax.plot(tiempo, tasa_inversion, c='dodgerblue')
        ax.set_ylim(min(tasa_inversion)-0.2, max(tasa_inversion)+0.2)
        ax.set_xlabel(r'$\lambda t / \omega $')
        ax.set_ylabel('W(t)')
        ax.set_title("Population Inversion (Time: {:.1f} s)".format(current_time), fontsize=12)
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=tasa_inversion.shape[0], repeat=False)
    # Save the animation
    writer = animation.FFMpegWriter(fps=25, metadata=dict(artist='Yo'), bitrate=-1)
    ani.save(savename + '.gif', writer=writer)
    plt.draw()
    plt.show()