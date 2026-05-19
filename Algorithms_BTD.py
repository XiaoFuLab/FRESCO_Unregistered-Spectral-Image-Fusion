import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy import sparse
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm


class BSCLL1_soft_constraint:
    '''
    This class is the python implementation of the BSCLL1 algorithm in paper:
    Ding, Meng, et al. "Hyperspectral super-resolution via interpretable block-term tensor modeling." IEEE Journal of Selected Topics in Signal Processing 15.3 (2020): 641-656.
    We further add a soft sum-to-one constraint to fix the scale ambiguity.
    More details please refer to https://github.com/MengDing56/Code_SCLL1_HSR.
    '''

    def __init__(self, data, R, Lambda=0.01, maxIter=1200, e=1e-9):
        '''
        Initialize the BSCLL1_soft_constraint class.

        input:
            data: the data dictionary, which should at least contain the {SRI_M, HSI, MSI, Q}
            R: the number of endmembers.
            Lambda: the weight of the soft sum-to-one constraint.
            maxIter: the maximum number of iterations.
            e: the tolerance for convergence.
        '''

        #===== store the parameters =====#
        self.R = R                  # number of materials
        self.Lambda = Lambda        # weight of the soft sum-to-one constraint (default:0.01)
        self.alpha = 0.1           # weight for norm regularization on C (default:1e-2)    
        self.eta = 5*1e-3           # weight for low-rank regularization (default:5*1e-3)
        self.p = 0.5                # parameter in the low-rank regularization (default:0.5)
        self.q = 0.5                # parameter in the low-rank regularization (default:0.5)
        self.theta = 1e-4           # weight for the TV smoothness regularization (default:1e-4)
        self.tau = 1                # parameter in the TV smoothness regularization (default:1)
        self.epsilon = 1e-3         # parameter in the TV smoothness regularization (default:1e-3)
        self.maxIter = maxIter      # maximum number of iterations (default:1200)
        self.e = e                  # tolerance for convergence (default:1e-9)


        #===== load the data =====#
        self.MSI = data['MSI']
        self.HSI = data['HSI']
        self.PM = data['Q']
        self.IH, self.JH, self.KH = self.HSI.shape
        self.IM, self.JM, self.KM = self.MSI.shape

        self.YM3 = self.MSI.reshape((-1, self.KM), order='F') #unfold the MSI (mode-3 unfolding)
        self.YH3 = self.HSI.reshape((-1, self.KH), order='F') #unfold the HSI (mode-3 unfolding)

        #===== Precompute the matrices and parameters used in the algorithm =====#
        print('Precomputing the matrices and parameters...')

        #Construct horizontal and vertical discrete gradient operators for TV regularization
        H_1 = np.identity(self.IM)
        H_roll = np.roll(H_1, 1, axis=1)
        H = sparse.csr_matrix(H_1 - H_roll)

        self.Hx = sparse.kron(H,sparse.identity(self.JM))
        self.Hy = sparse.kron(sparse.identity(self.JM),H)

        #Construct the all one vectors used in the soft sum-to-one constraint
        self.row_1 = np.ones((R,1))  
        self.col_1_M = np.ones((self.IM*self.JM,1))
        self.col_1_H = np.ones((self.IH*self.JH,1))

        #Pre-compute the largest singular values for step size estimation
        self.sgv_PM = np.linalg.svd(np.dot(self.PM.T,self.PM), full_matrices=False, compute_uv=False)[0]
        self.sgv_HxT = sparse.linalg.svds(self.Hx.T,return_singular_vectors=False,k=1)[0]
        self.sgv_HyT = sparse.linalg.svds(self.Hy.T,return_singular_vectors=False,k=1)[0]
        self.sgv_Hx = sparse.linalg.svds(self.Hx,return_singular_vectors=False,k=1)[0]
        self.sgv_Hy = sparse.linalg.svds(self.Hy,return_singular_vectors=False,k=1)[0]

    
    #===== Define objective used in the method =====#
    def Objective(self,CH,SM3,SH3):
        '''
        Compute the value of the objective function

        input:
            CH: the estimated endmembers (shape: KH * R).
            SM3: the unfolded estimated abundance maps of the MSI (shape: IMJM * R).
            SH3: the unfolded estimated abundance maps of the HSI (shape: IHJH * R).

        output:
            obj: the objective value. 
        '''

        #regression term for HSI
        term_regression_HSI = 0.5 * np.linalg.norm(self.YH3 - SH3.dot(CH.T),'fro')**2


        #regression term for MSI
        term_regression_MSI = 0.5 * np.linalg.norm(self.YM3 - SM3.dot((self.PM.dot(CH)).T),'fro')**2


        #low-rank regularization term
        term_low_rank =0
        for r in range(self.R):
            SM = SM3[:,r].reshape((self.IM,self.JM),order='F')
            SH = SH3[:,r].reshape((self.IH,self.JH),order='F')

            #low-rank regularization term for MSI
            term_low_rank += np.trace((SM.dot(SM.T) + self.tau*np.identity(SM.shape[0]))**(self.p/2))

            #low-rank regularization term for HSI 
            term_low_rank += np.trace((SH.dot(SH.T) + self.tau*np.identity(SH.shape[0]))**(self.p/2))


        #TV smoothness regularization term (only for MSI)
        term_smoothness = 0
        for r in range(self.R):
            qr = SM3[:,r]

            #TV smoothness regularization term for the row direction
            term_smoothness += np.sum((((sparse.csr_matrix.dot(self.Hx,qr))**2)+self.epsilon)**(self.q/2))

            #TV smoothness regularization term for the column direction
            term_smoothness += np.sum((((sparse.csr_matrix.dot(self.Hy,qr))**2)+self.epsilon)**(self.q/2))

        #norm C regularization term
        term_norm_C = 0.5* np.linalg.norm(CH, ord='fro')**2

        #soft sum-to-one constraint term
        term_sum2one = 0.5 * (np.linalg.norm(SM3.dot(self.row_1) - self.col_1_M)**2 + np.linalg.norm(SH3.dot(self.row_1) - self.col_1_H)**2)

        #calculate the objective value
        obj = term_regression_HSI + term_regression_MSI + self.eta * term_low_rank + self.theta * term_smoothness + self.Lambda * term_sum2one + self.alpha * term_norm_C

        return obj


    #===== Compute the gradient and step size for CH optimization =====#
    def Grad_C(self, CH, SM3, SH3):
        '''
        Compute the gradient and maximum step size for the gradient descent of CH matrix

        input:
            CH: the estimated endmembers (shape: KH * R).
            SM3: the unfolded estimated abundance maps of the MSI (shape: IMJM * R).
            SH3: the unfolded estimated abundance maps of the HSI (shape: IHJH * R).

        output:
            grad: the gradient of the objective function with respect to CH.
            step_size: the maximum step size for the gradient descent.
        '''

        #Compute the terms in the gradient
        grad_regression_H =  CH.dot(SH3.T).dot(SH3) - (self.YH3.T).dot(SH3)
        
        grad_regression_M = (self.PM.T).dot(self.PM).dot(CH).dot(SM3.T).dot(SM3) - (self.PM.T).dot(self.YM3.T).dot(SM3)
        
        grad = grad_regression_H + grad_regression_M + self.alpha * CH

        #calculate the step size which is bounded by the inverse of the largest singular value of the hessian
        sgv_H = np.linalg.svd(np.dot(SH3.T,SH3), full_matrices=False, compute_uv=False)[0]
        sgv_M = np.linalg.svd(np.dot(SM3.T,SM3), full_matrices=False, compute_uv=False)[0]

        step_size = 1/(sgv_H + self.sgv_PM * sgv_M + self.alpha)

        return grad, step_size
    


    #===== Compute the gradient and step size for SM3 optimization (More complex, we divide it to servel blocks) =====#
    def Grad_SM_regression(self,CH,SM3):
        '''
        Compute the gradient and maximum singular value (for hessian matrix) on regression term w.r.t. SM3.

        input:
            CH: the estimated endmembers (shape: KH * R).
            SM3: the unfolded estimated abundance maps of the MSI (shape: IMJM * R).

        output:
            grad_regression: the gradient of the regression term w.r.t. SM3.
            sgv_regression: the maximum singular value for hessian matrix on regression term w.r.t. SM3.
        '''

        #Compute the gradient
        grad_regression = (SM3.dot(CH.T).dot(self.PM.T) - self.YM3).dot(self.PM).dot(CH)

        #Compute the singular value
        sgv_CH = np.linalg.svd(CH, full_matrices=False, compute_uv=False)[0]
        sgv_regression = self.sgv_PM * (sgv_CH**2)

        return grad_regression, sgv_regression
    


    def Grad_SM_low_rank(self, SM3):
        '''
        Compute the gradient and maximum singular value (for hessian matrix) on low-rank regularization term w.r.t. SM3.

        input:
            SM3: the unfolded estimated abundance maps of the MSI (shape: IMJM * R).

        output:
            grad_low_rank: the gradient of the low-rank regularization w.r.t. SM3.
            sgv_low_rank: the maximum singular value for hessian matrix on low-rank regularization term w.r.t. SM3.
        '''

        grad_low_rank = np.zeros((self.IM*self.JM,self.R))
        sgv_low_rank = np.zeros(self.R)
        for r in range(self.R):
            #Construct W_r matrix
            SMr = (SM3[:,r]).reshape((self.IM,self.JM),order='F')

            base = np.dot(SMr,SMr.T) + self.tau*np.identity(SMr.shape[0]) # term inside W_r
            eigenvalues, eigenvectors = np.linalg.eigh(base) 
            Wr = eigenvectors.dot(np.diag(eigenvalues**((self.p-2)/2))).dot(eigenvectors.T) #construct W_r matrix by raising eigenvalues to a fractional power.

            #Compute the gradient
            grad_low_rank[:,r] = self.p * np.dot(Wr,SMr).reshape(self.IM*self.JM,order='F')

            #Compute the singular value
            sgv_low_rank[r] = self.p * np.linalg.svd(Wr, full_matrices=False, compute_uv=False)[0]


        return grad_low_rank, np.max(sgv_low_rank)
    


    def Grad_SM_smoothness_row(self,SM3):
        '''
        Compute the gradient and maximum singular value (for Ur, r = 1...R) on smoothness regularization (row direction) w.r.t. SM3.

        input:
            SM3: the unfolded estimated abundance maps of the MSI (shape: IMJM * R).

        output:
            grad_smoothness_row: the gradient of the smoothness regularization (row direction) w.r.t. SM3.
            sgv_U: the maximum singular values (for Ur, r = 1...R) on smoothness regularization (row direction) term w.r.t. SM3.
        '''

        grad_smoothness_row = np.zeros((self.IM*self.JM,self.R))
        sgv_U = np.zeros(self.R)

        for r in range(self.R):
            qr = SM3[:,r]
            #Construct U_r matrix
            Ur3 = (((sparse.csr_matrix.dot(self.Hx,qr))**2)+self.epsilon)**((self.q-2)/2)
            Ur = sparse.diags(Ur3) #Construct U_r matrix by Diagonalizing the vector Ur3

            #Compute the gradient
            grad_smoothness_row[:,r] = self.q * sparse.csr_matrix.dot(sparse.csr_matrix.dot(sparse.csr_matrix.dot(self.Hx.T,Ur),self.Hx),qr)

            #Compute the singular value
            sgv_U[r] = self.q * np.max(Ur3)

        return grad_smoothness_row, sgv_U
    


    def Grad_SM_smoothness_col(self,SM3):
        '''
        Compute the gradient and maximum singular value (for V_r, r = 1...R) on smoothness regularization (column direction) w.r.t. SM3.

        input:
            SM3: the unfolded estimated abundance maps of the MSI (shape: IMJM * R).

        output:
            grad_smoothness_col: the gradient of the smoothness regularization (column direction) w.r.t. SM3.
            sgv_V: the maximum singular values (for V_r, r = 1....R) on smoothness regularization (column direction) term w.r.t. SM3.
        '''

        grad_smoothness_col = np.zeros((self.IM*self.JM,self.R))
        sgv_V = np.zeros(self.R)

        for r in range(self.R):
            qr = SM3[:,r]
            #Construct V_r matrix
            Vr3 = (((sparse.csr_matrix.dot(self.Hy,qr))**2)+self.epsilon)**((self.q-2)/2)
            Vr = sparse.diags(Vr3)

            #Compute the gradient
            grad_smoothness_col[:,r] = self.q * sparse.csr_matrix.dot(sparse.csr_matrix.dot(sparse.csr_matrix.dot(self.Hy.T,Vr),self.Hy),qr)
            
            #Compute the singular value
            sgv_V[r] = self.q * np.max(Vr3)

        return grad_smoothness_col, sgv_V
    


    def Grad_SM_sum2one(self,SM3):
        '''
        Compute the gradient and maximum singular value (for hessian) on soft sum-to-one constraint w.r.t. SM3.

        input:
            SM3: the unfolded estimated abundance maps of the MSI (shape: IMJM * R).

        output:
            grad_sum2one: the gradient of the soft sum-to-one constraint w.r.t. SM3.
            sgv_sum2one: the maximum singular value (for hessian matrix) on soft sum-to-one constraint w.r.t. SM3.
        '''

        #Compute the gradient
        grad_sum2one = (SM3.dot(self.row_1) - self.col_1_M).dot(self.row_1.T)

        #Compute the singular value
        sgv_sum2one = 1

        return grad_sum2one, sgv_sum2one
     

    def Grad_SM(self,CH,SM3):
        '''
        Compute the gradient and step size for gradient descent w.r.t. SM3.

        input:
            CH: the estimated endmembers (shape: KH * R).
            SM3: the unfolded estimated abundance maps of the MSI (shape: IMJM * R).

        output:
            grad: the gradient of the objective function w.r.t. SM3.
            step_size: the maximum step size for the gradient descent.
        '''

        #Take out the gradient and singular value for every term
        grad_regression, sgv_regression = self.Grad_SM_regression(CH,SM3)
        grad_low_rank, sgv_low_rank = self.Grad_SM_low_rank(SM3)
        grad_smoothness_row, sgv_U = self.Grad_SM_smoothness_row(SM3)
        grad_smoothness_col, sgv_V = self.Grad_SM_smoothness_col(SM3)
        grad_sum2one, sgv_sum2one = self.Grad_SM_sum2one(SM3)

        #Compute the gradient
        grad = grad_regression + self.eta * grad_low_rank + self.theta * (grad_smoothness_row + grad_smoothness_col) + self.Lambda * grad_sum2one

        #Compute the step size
        sgv_smoothness_row = np.max(self.sgv_HxT * sgv_U * self.sgv_Hx)
        sgv_smoothness_col = np.max(self.sgv_HyT * sgv_V * self.sgv_Hy)

        sgv = sgv_regression + self.eta * sgv_low_rank + self.theta * (sgv_smoothness_row + sgv_smoothness_col) + self.Lambda * sgv_sum2one

        step_size = 1/sgv

        return grad, step_size
    

    #===== Compute the gradient and step size for SH3 optimization (More complex, we divide it to servel blocks) =====#
    def Grad_SH_regression(self,CH,SH3):
        '''
        Compute the gradient and maximum singular value (for hessian matrix) on regression term w.r.t. SH3.

        input:
            CH: the estimated endmembers (shape: KH * R).
            SH3: the unfolded estimated abundance maps of the HSI (shape: IHJH * R).

        output:
            grad_regression: the gradient of the regression term w.r.t. SH3.
            sgv_regression: the maximum singular value for hessian matrix on regression term w.r.t. SH3.
        '''

        #Compute the gradient
        grad_regression = (SH3.dot(CH.T)- self.YH3).dot(CH)

        #Compute the singular value
        sgv_regression = np.linalg.svd(np.dot(CH.T,CH), full_matrices=False, compute_uv=False)[0]

        return grad_regression, sgv_regression
    

    def Grad_SH_low_rank(self, SH3):
        '''
        Compute the gradient and maximum singular value (for hessian matrix) on low-rank regularization term w.r.t. SH3.

        input:
            SH3: the unfolded estimated abundance maps of the HSI (shape: IHJH * R).

        output:
            grad_low_rank: the gradient of the low-rank regularization w.r.t. SH3.
            sgv_low_rank: the maximum singular value for hessian matrix on low-rank regularization term w.r.t. SH3.
        '''

        grad_low_rank = np.zeros((self.IH*self.JH,self.R))
        sgv_low_rank = np.zeros(self.R)
        for r in range(self.R):
            SHr = (SH3[:,r]).reshape((self.IH,self.JH),order='F')

            #Construct W_r matrix
            base = np.dot(SHr,SHr.T) + self.tau*np.identity(SHr.shape[0]) # term inside W_r
            eigenvalues, eigenvectors = np.linalg.eigh(base)
            Wr = eigenvectors.dot(np.diag(eigenvalues**((self.p-2)/2))).dot(eigenvectors.T) #construct W_r matrix by raising eigenvalues to a fractional power.

            #Compute the gradient
            grad_low_rank[:,r] = self.p * np.dot(Wr,SHr).reshape(self.IH*self.JH,order='F')

            #Compute the singular value
            sgv_low_rank[r] = self.p * np.linalg.svd(Wr, full_matrices=False, compute_uv=False)[0]

        return grad_low_rank, np.max(sgv_low_rank)
    


    def Grad_SH_sum2one(self,SH3):
        '''
        Compute the gradient and maximum singular value (for hessian matrix) on soft sum-to-one constraint w.r.t. SH3.

        input:
            SH3: the unfolded estimated abundance maps of the HSI (shape: IHJH * R).

        output:
            grad_sum2one: the gradient of the soft sum-to-one constraint w.r.t. SH3.
            sgv_sum2one: the maximum singular value (for hessian matrix) on soft sum-to-one constraint w.r.t. SH3.
        '''

        #Compute the gradient
        grad_sum2one = (SH3.dot(self.row_1) - self.col_1_H).dot(self.row_1.T)

        #Compute the singular value
        sgv_sum2one = 1

        return grad_sum2one, sgv_sum2one
    

    def Grad_SH(self,CH,SH3):
        '''
        Compute the gradient and step size for gradient descent w.r.t. SH3.

        input:
            CH: the estimated endmembers (shape: KH * R).
            SH3: the unfolded estimated abundance maps of the HSI (shape: IHJH * R).

        output:
            grad: the gradient of the objective function w.r.t. SH3.
            step_size: the maximum step size for the gradient descent.
        '''

        #Take out the gradient and singular value for every term
        grad_regression, sgv_regression = self.Grad_SH_regression(CH,SH3)
        grad_low_rank, sgv_low_rank = self.Grad_SH_low_rank(SH3)
        grad_sum2one, sgv_sum2one = self.Grad_SH_sum2one(SH3)

        #Compute the gradient
        grad = grad_regression + self.eta * grad_low_rank + self.Lambda * grad_sum2one

        #Compute the step size
        sgv = sgv_regression + self.eta * sgv_low_rank + self.Lambda * sgv_sum2one

        step_size = 1/sgv

        return grad, step_size
    

    #===== The main function (projected gradient descent) of the BSCLL1 algorithm =====#
    def PGD_BSCLL1(self):
        '''
        The main function of the BSCLL1 algorithm. 
        We use projected gradient descent to optimize the objective function.
        We also apply the Nesterov's acceleration technique to speed up the convergence.

        output:
            SM3: the estimated abundance maps of the MSI (shape: IMJM * R).
            SH3: the estimated abundance maps of the HSI (shape: IHJH * R).
            CH: the estimated endmembers (shape: KH * R).
        '''

        #===== Initialize variables =====#
        CH = np.random.rand(self.KH,self.R)
        SM3 = np.random.rand(self.IM*self.JM,self.R)
        SH3 = np.random.rand(self.IH*self.JH,self.R)

        obj = 0.01

        #===== Initialize the variables for Nesterov's acceleration =====#
        CH_old = CH
        SM3_old = SM3
        SH3_old = SH3

        CH_N = CH
        SM_N = SM3
        SH_N = SH3

        gamma_CH = 1
        gamma_SM = 1
        gamma_SH = 1

        print('Start the optimization...')

        for i in tqdm(range(self.maxIter)):

            #===== Update CH =====#
            #compute the gradient and step size
            grad_CH, step_CH = self.Grad_C(CH_N,SM_N,SH_N)

            #record the old value to compute the momentum
            CH_old = CH

            #update the C (gradient descent + projection)
            CH = np.maximum(CH_N - step_CH * grad_CH,0)

            #update the momentum
            gamma_CH_next = (1+np.sqrt(1+4*gamma_CH**2))/2
            CH_N = CH + ((gamma_CH-1)/gamma_CH_next)*(CH-CH_old)


            #===== Update SM3 =====#
            #compute the gradient and step size
            grad_SM, step_SM = self.Grad_SM(CH,SM_N)

            #record the old value to compute the momentum
            SM3_old = SM3

            #update the SM3 (gradient descent + projection)
            SM3 = np.maximum(SM_N - step_SM * grad_SM,0)

            #update the momentum
            gamma_SM_next = (1+np.sqrt(1+4*gamma_SM**2))/2
            SM_N = SM3 + ((gamma_SM-1)/gamma_SM_next)*(SM3-SM3_old)


            #===== Update SH3 =====#
            #compute the gradient and step size
            grad_SH, step_SH = self.Grad_SH(CH,SH_N)

            #record the old value to compute the momentum
            SH3_old = SH3

            #update the SH3 (gradient descent + projection)
            SH3 = np.maximum(SH_N - step_SH * grad_SH,0)

            #update the momentum
            gamma_SH_next = (1+np.sqrt(1+4*gamma_SH**2))/2
            SH_N = SH3 + ((gamma_SH-1)/gamma_SH_next)*(SH3-SH3_old)


            #===== Update the gamma values =====#
            gamma_CH = gamma_CH_next
            gamma_SM = gamma_SM_next
            gamma_SH = gamma_SH_next


            #===== Compute the objective value =====#
            obj_old = obj
            obj = self.Objective(CH,SM3,SH3)

            #compute the relative change of the objective value
            dis = np.abs(obj - obj_old)/np.abs(obj_old) 

            #check convergence
            if dis < self.e:
                print('Converged at iteration:',i)
                break

            print('Iter:',i,'; objective:',obj,'; relative change:',dis)
        
        return SM3, SH3, CH















def Evaluation_and_Visualization(S,ST,C,flag_eva,flag_visual,result_root,size_M,size_H,R):
    [I,J] = size_M
    [IH,JH] = size_H

    fig,ax = plt.subplots(2,R,figsize=(30,11))
    S3 = S.reshape((I,J,R),order='F')
    ST3 = ST.reshape((IH,JH,R),order='F')
    for i in range(R):
        ax[0,i].imshow(S3[:,:,i],vmin=0,vmax=1)
        ax[1,i].imshow(ST3[:,:,i],vmin=0,vmax=1)
        ax[0,i].axis('off')
        ax[1,i].axis('off')
        str_temp = 'r = '+str(i+1)
        ax[0,i].set_title(str_temp,fontsize=20)

    plt.tight_layout()
    fig.text(0.01, 0.75, 'Abun_M', va='center', ha='center', rotation='vertical', fontsize=20)
    fig.text(0.01, 0.25, 'Abun_H', va='center', ha='center', rotation='vertical', fontsize=20)

    plt.savefig(result_root+'Abun_Stage_1.jpg',dpi=300)



