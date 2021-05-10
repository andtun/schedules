import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation

Nx, Ny, Nt = 50, 50, 50
T_max = 1

x_grid = np.linspace(0, 1, Nx)
y_grid = np.linspace(0, 1, Ny)
t_grid = np.linspace(0, T_max, Nt)

def get_step(grid):
    return grid[1] - grid[0]

def get_gamma(h):
    return tau / h ** 2

h_x, h_y, tau = tuple(map(get_step, [x_grid, y_grid, t_grid]))
gamma_x, gamma_y = tuple(map(get_gamma, [h_x, h_y]))
solution = np.zeros(shape=(Nx, Ny, 2 * Nt + 1))

def F_1(ix, iy, jt):
    def tf_1(ix, iy, jt):
        return solution[ix, iy - 1, jt - 1] + solution[ix, iy + 1, jt - 1]

    def htg_1(ix, iy, jt):
        return np.exp(-tau * (jt + 1) / 2) * np.cos(np.pi * y_grid[iy])

    return 0.5 * gamma_y * tf_1(ix, iy, jt) +\
           (1 - gamma_y) * solution[ix, iy, jt - 1] +\
           0.5 * tau * htg_1(ix, iy, jt)


def F_2(ix, iy, jt):
    def tf_2(ix, iy, jt):
        return solution[ix - 1, iy, jt - 1] + solution[ix + 1, iy, jt - 1]

    def htg_2(ix, iy, jt):
        return np.exp(-tau * (jt - 1) / 2) * np.cos(np.pi * y_grid[iy])

    return 0.5 * gamma_x * tf_2(ix, iy, jt) +\
           (1 - gamma_x) * solution[ix, iy, jt - 1] +\
           0.5 * tau * htg_2(ix, iy, jt)


def iter_init(gamma, **kwargs):
    d = np.zeros(Nx)
    sigma = np.zeros(Nx)
    d[1] = kwargs.get('d1', 0)
    sigma[1] = 0
    A = 0.5 * gamma
    B = 1 + gamma
    C = 0.5 * gamma
    return d, sigma, A, B, C

def next_sigma(Fm, A, sigma_m, d_m, B):
    return (Fm - A * sigma_m) / (A * d_m - B)

def next_d(A, B, C, d_m):
    return C / (B - A * d_m)

def iter_x(iy, jt):
    d, sigma, A, B, C = iter_init(gamma=gamma_x, d1=0)

    m_range = range(1, Nx - 1)
    for m in m_range:
        Fm_1 = -F_1(m, iy, jt)
        d[m + 1] = next_d(A, B, C, d[m])
        sigma[m + 1] = next_sigma(Fm_1, A, sigma[m], d[m], B)
    solution[Nx - 1, iy, jt] = 0

    m_range = range(Nx - 1, 0, -1)
    for m in m_range:
        solution[m - 1, iy, jt] = d[m] * solution[m, iy, jt] + sigma[m]


def iter_y(ix, jt):
    d, sigma, A, B, C = iter_init(gamma=gamma_y, d1=1)

    m_range = range(1, Ny - 1)
    for m in m_range:
        Fm_1 = -F_2(ix, m, jt)
        d[m + 1] = next_d(A, B, C, d[m])
        sigma[m + 1] = next_sigma(Fm_1, A, sigma[m], d[m], B)
    solution[ix, Ny - 1, jt] = sigma[-1] / (1 - d[-1])

    m_range = range(Ny - 1, 0, -1)
    for m in m_range:
        solution[ix, m - 1, jt] = d[m] * solution[ix, m, jt] + sigma[m]

solution[:, :, 0] = 0

for j in range(1, 2*Nt, 2):
    for i2 in range(1, Ny-1):
        iter_x(i2, j)
    for i1 in range(1, Nx-1):
        iter_y(i1, j+1)

fig = plt.figure(figsize=(8,6))
plt.pcolormesh(y_grid, x_grid, solution[:, :, -1])
plt.colorbar()

%matplotlib inline

X, Y = np.meshgrid(x_grid, y_grid)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_wireframe(X[1:-1,1:-1], Y[1:-1,1:-1], solution[1:-1, 1:-1, 1], rstride=3, cstride=3)

% matplotlib inline

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
# ax.set_ylim([Y.min(), Y.max()])
# ax.set_xlim([X.min(), X.max()])
X, Y = np.meshgrid(x_grid, y_grid)


def animate(n):
    ax.clear()
    ax.set_zlim(bottom=solution.min(), top=solution.max())
    ax.plot_wireframe(X[1:-1, 1:-1], Y[1:-1, 1:-1], solution[1:-1, 1:-1, n], rstride=3, cstride=3, animated=True)
    return ax


anim = animation.FuncAnimation(fig, animate, frames=range(1, Nt))
anim.save('t.gif')