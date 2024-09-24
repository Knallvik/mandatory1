import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.x, self.h = np.linspace(0, self.L, self.N+1, retstep=True)
        
        return np.meshgrid(self.x, self.x, indexing='ij', sparse=True)

    def D2(self):
        """Return second order differentiation matrix"""
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D2[0, :4] = 2, -5, 4, -1
        D2[-1, -4:] = -1, 4, -5, 2
        D2 /= self.h**2
        return D2

    def laplace(self):
        """Return vectorized Laplace operator"""
        D2x = self.D2()
        D2y = D2x.copy()
        
        A = sparse.kron(D2x,sparse.eye(self.N+1)) + sparse.kron(sparse.eye(self.N+1),D2y) 
        A = A.tolil()
        
        return A

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        
        """
        bnds = []
        for i in range(self.px.N+1):
            for j in range(self.py.N+1):
                if i%self.px.N==0 or j%self.py.N==0:
                    bnds.append((self.py.N+1)*i+j)
        """
        B = np.ones((self.N+1,self.N+1), dtype=bool)
        B[1:-1,1:-1] = 0
        bnds = np.where(B.ravel()==1)[0]
        
        return bnds

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        # return A, b
        xi, yj = self.create_mesh()
        
        A = self.laplace()
        b = sp.lambdify((x,y), self.f)(xi,yj)
        bnds = self.get_boundary_indices()

        for i in bnds:
            A[i] = 0
            A[i,i] = 1
            #Set boundary conditions
            k = int(i/(self.N+1))
            j = i%(self.N+1)
            b[k,j] = sp.lambdify((x,y), self.ue)(xi[k,0],yj[0,j])

        return A.tocsr(), b

    def l2_error(self, u):
        """Return l2-error norm"""
        xi, yj = self.create_mesh()
        ue_vec = sp.lambdify((x,y), self.ue)(xi,yj)
        
        return np.sqrt(self.h**2*np.sum((ue_vec-u)**2))

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.N = N
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((self.N+1, self.N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

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
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def laplace_poly(self, x0, x_arr):
        """Return Laplace polynomial basis

        Parameters
        ----------
        x0: A number 
            The location of the point of interest
        x_arr: a list
            the x-values to consider near the point of interest, x

        Returns
        -------
        A list of the value of the basis polynomials
        """
        ls = np.empty_like(x_arr)
        for j in range(len(x_arr)):
            l=1
            for i, x_ in enumerate(x_arr):
                if j!=i:
                    l *= (x0-x_)/(x_arr[j]-x_)
            ls[j] = l
            
        return ls

    def laplace_func(self, x, y, order=2):
        """Return Laplace interpolation

        Parameters
        ----------
        x , y: A number 
            The locations of the points of interest
        order: int
            The order of accurazy desired

        Returns
        -------
        The value of the interpolation function evaluated at (x,y)
        """
        #Get the Mesh
        xi, yj = self.create_mesh()
        #number of points required for the desired accurazy
        k = order + 1

        #The indices of x/y rounded down
        ix = int((x/self.L)*self.N)
        jy = int((y/self.L)*self.N)
        
        #get the polynomials for x
        if ix+order > self.N:
            #loop to max N, with "k" steps.
            lsx = self.laplace_poly(x,xi[ix+(1-k):ix+1,0])
            ix_start = ix+(1-k)
        elif ix-order < 0:
            # loop from 0 to k
            lsx = self.laplace_poly(x,xi[:k,0])
            ix_start = 0
        else:
            # loop ix-1, ix, ix+1 if order=2, loop ix-1, ix, ix+1, ix+2 if order=3 and so on. 
            lowest_index = ix-int((k-1)/2)
            highest_index = ix+int(k/2)+1 #+1 because of how python gets the range i:N (goes to N-1)
            
            lsx = self.laplace_poly(x,xi[lowest_index:highest_index,0])
            ix_start = lowest_index
            
        #get the polynomials for y
        if jy+order > self.N:
            #loop to jy, with "k" steps.
            lsy = self.laplace_poly(y,yj[0,jy+(1-k):jy+1])
            jy_start = jy+(1-k)
        elif jy-order < 0:
            # loop from 0 to k
            lsy = self.laplace_poly(y,yj[0,:k])
            jy_start = 0
        else:
            # loop jy-1, ix, jy+1 if order=3, loop jy-1, jy, jy+1, jy+2 if order=4 and so on. 
            lowest_index = jy-int((k-1)/2)
            highest_index = jy+int(k/2)+1 #+1 because of how python gets the range i:N (goes to N-1)
            
            lsy = self.laplace_poly(y,yj[0,lowest_index:highest_index])
            jy_start = lowest_index
            
        print(lsx,lsy)
        u = 0
        #Keep consistent with where the function is evaluated
        i = ix_start
        #Sum over the basis plynomials at x and y (the weight of each mesh function value)
        for lx in lsx:
            #See i = ix_start
            j = jy_start
            for ly in lsy:
                u += self.U[i,j]*lx*ly
                j+=1
            i+=1
        return u
        
        
    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """        
        u = self.laplace_func(x,y,order=3)
        return u
        

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    #print(abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()))
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3

if __name__=='__main__':
    test_convergence_poisson2d()
    test_interpolation()

