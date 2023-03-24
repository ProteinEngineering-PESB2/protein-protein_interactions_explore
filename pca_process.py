from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import numpy as np


class transformation_data():

    def apply_tsne(
        self,
        dataset,
        n_components=3,
        learning_rate='auto', 
        init='random', 
        perplexity=3):

        tsne_instance = TSNE(
            n_components=n_components,
            learning_rate=learning_rate,
            init=init,
            perplexity=perplexity)
         
        transform_data = tsne_instance.fit_transform(dataset)

        return transform_data, tsne_instance

    def apply_pca_linear(
        self, 
        dataset=None, 
        n_components=None, 
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None):
        
        pca_instance = PCA(
            n_components=n_components,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state)

        pca_instance.fit(dataset)
        transform_data = pca_instance.transform(dataset)
        transform_data = np.round(transform_data, 3)

        return transform_data, pca_instance
    
    def apply_kernel_pca(
        self, 
        dataset=None,
        n_components=None, 
        kernel="linear", 
        gamma=None, 
        degree=3,
        coef0=1,
        alpha=1.0,
        eigen_solver="auto",
        tol=0,
        iterated_power="auto",
        random_state=None,
        max_iter=None, 
        n_jobs=-1):

        kernel_pca_instance = KernelPCA(
            n_components=n_components,
            kernel=kernel, 
            gamma=gamma, 
            degree=degree,
            coef0=coef0,
            alpha=alpha,
            eigen_solver=eigen_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
            max_iter=max_iter,
            n_jobs=n_jobs)
        
        kernel_pca_instance.fit(dataset)
        transform_data = kernel_pca_instance.transform(dataset)
        transform_data = np.round(transform_data, 3)

        return transform_data, kernel_pca_instance
    
    def apply_sparce_pca(
        self, 
        dataset=None, 
        alpha=1, 
        ridge_alpha=0.01, 
        max_iter=1000,
        tol=1e-8,
        method="lars",
        n_jobs=-1,
        U_init=None,
        V_init=None,
        random_state=None):

        sparce_instance = SparsePCA(
            alpha=alpha, 
            ridge_alpha=ridge_alpha, 
            max_iter=max_iter, 
            tol=tol, 
            method=method, 
            n_jobs=n_jobs, 
            U_init=U_init, 
            random_state=random_state, 
            V_init=V_init)
        
        sparce_instance.fit(dataset)
        trasnform_data = sparce_instance.transform(dataset)

        return trasnform_data, sparce_instance
    
    def apply_truncate_svd(
        self, 
        dataset=None,
        n_components=2,
        algorithm="randomized",
        n_iter=5,
        random_state=None,
        tol=0):

        truncate_instance = TruncatedSVD(
            n_components=n_components,
            algorithm=algorithm,
            n_iter=n_iter,
            random_state=random_state,
            tol=tol
        )

        truncate_instance.fit(dataset)
        transform_data = truncate_instance.transform(dataset)

        return transform_data, truncate_instance
    
    def apply_non_negative_matrix_factorization(
        self,
        dataset=None,
        n_components=None,
        init=None,
        solver="cd",
        beta_loss="frobenius",
        tol=1e-4,
        max_iter=200,
        random_state=None,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0):

        nmf_instance = NMF(
            n_components=n_components,
            init=init,
            solver=solver,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio
        )

        nmf_instance.fit(dataset)
        transform_data= nmf_instance.transform(dataset)

        return transform_data, nmf_instance

    def apply_incremental_pca(
        self,
        dataset=None,
        n_components=None,
        whiten=False,
        batch_size=None):

        incremental_instance = IncrementalPCA(
            n_components=n_components,
            whiten=whiten,
            batch_size=batch_size
        )

        incremental_instance.fit(dataset)
        transform_data = incremental_instance.transform(dataset)

        return transform_data, incremental_instance
    
    def apply_factor_analysis(
        self,
        dataset=None,
        n_components=None,
        tol=1e-2,
        max_iter=1000,
        noise_variance_init=None,
        svd_method="randomized",
        iterated_power=3,
        rotation=None,
        random_state=0
        ):

        factor_instance = FactorAnalysis(
            n_components=n_components,
            tol=tol,
            max_iter=max_iter,
            noise_variance_init=noise_variance_init,
            svd_method=svd_method,
            iterated_power=iterated_power,
            rotation=rotation,
            random_state=random_state
        )

        factor_instance.fit(dataset)
        transform_data = factor_instance.transform(dataset)

        return transform_data, factor_instance
