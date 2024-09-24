import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        x = np.linspace(0,1,N+1)
        self.xij, self.yij = np.meshgrid(x,x,indexing='ij', sparse=sparse)

    def D2(self, N):
        """Return second order differentiation matrix"""
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D2[0, :4] = 2, -5, 4, -1
        D2[-1, -4:] = -1, 4, -5, 2
        return D2

    @property
    def w(self):
        """Return the dispersion coefficient"""
        kx = self.mx*np.pi
        ky = self.my*np.pi
        return self.c*np.sqrt(kx**2+ky**2)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        U0 = sp.lambdify((x,y,t),self.ue(self.mx,self.my))(self.xij,self.yij,0)
        U1 = sp.lambdify((x,y,t),self.ue(self.mx,self.my))(self.xij,self.yij,self.dt)
        return U0,U1

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl/self.N/self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue_num = sp.lambdify((x,y,t),self.ue(self.mx,self.my))(self.xij,self.yij,t0)
        return np.sqrt(1/self.N**2*np.sum((u-ue_num)**2))
        

    def apply_bcs(self):
        """ All boundaries are zero """
        self.U[0] = 0
        self.U[-1] = 0
        self.U[:,0] = 0
        self.U[:,-1] = 0
        
    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.Nt = Nt
        self.cfl = cfl
        self.c =c
        self.N = N
        self.mx = mx
        self.my = my
        
        self.create_mesh(N)
        U0,U1 = self.initialize(N,mx,my)
        self.n = 1
        D2 = self.D2(N)
        
        if store_data>0:
            sol = {0:U0, 1:U1}
            while self.n <= Nt:
                self.U = 2*sol[self.n]-sol[self.n-1] + (cfl)**2*(D2@sol[self.n]+sol[self.n]@D2.T)
                self.apply_bcs()
                self.n+=1
                sol[self.n] = self.U
            return sol

        elif store_data==-1:
            sol = [U0,U1] # Use the modulus operator to cycle correctly through the temporarily stored solutions
            l2=[0,0]
            while self.n<=Nt:
                self.U = 2*sol[(self.n)%2]-sol[(self.n+1)%2] + (cfl)**2*(D2@sol[(self.n)%2]+sol[(self.n)%2]@D2.T)
                self.apply_bcs()
                sol[(self.n+1)%2] = self.U
                self.n+=1
                l2_n = self.l2_error(self.U,self.n*self.dt)
                l2.append(l2_n)
            return 1/self.N, l2
            
        else:
            raise ValueError('you chose...  poorly (value for store_data)')
        

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D2[0, :4] = 2, -5, 4, -1
        D2[-1, -4:] = -1, 4, -5, 2
        return D2

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        self.U[0] = sp.lambdify((x,y,t),self.ue(self.mx,self.my))(0,self.yij[0],(self.n+1)*self.dt) # x=0
        self.U[-1] = sp.lambdify((x,y,t),self.ue(self.mx,self.my))(1,self.yij[0],(self.n+1)*self.dt) # x=1
        self.U[:,0] = sp.lambdify((x,y,t),self.ue(self.mx,self.my))(self.xij[:,0],0,(self.n+1)*self.dt) # y=0
        self.U[:,-1] = sp.lambdify((x,y,t),self.ue(self.mx,self.my))(self.xij[:,0],1,(self.n+1)*self.dt) # y=1

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    sol = Wave2D()
    h, err = sol(10, 10, cfl=1/np.sqrt(2), mx=2, my=2, store_data=-1)
    assert err[-1] < 1e-12
    solN = Wave2D_Neumann()
    h, err = solN(10, 10, cfl=1/np.sqrt(2), mx=2, my=2, store_data=-1)
    assert err[-1] < 1e-12

if __name__=='__main__':
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()

